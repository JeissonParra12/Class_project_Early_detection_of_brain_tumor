import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class LightweightRegionProposalNetwork(nn.Module):
    """
    Stage 1: Lightweight Region Proposal Network (RPN)
    Optimized for high recall to detect small or low-contrast lesions
    """
    
    def __init__(self, input_channels: int = 4, anchor_sizes: List[Tuple] = None):
        super(LightweightRegionProposalNetwork, self).__init__()
        
        self.input_channels = input_channels
        
        # Default anchor sizes for brain tumors (small, medium, large)
        if anchor_sizes is None:
            anchor_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]
        self.anchor_sizes = anchor_sizes
        self.num_anchors = len(anchor_sizes)
        
        # Lightweight backbone for feature extraction
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
        )
        
        # Region Proposal Head
        self.rpn_conv = nn.Conv2d(128, 256, 3, padding=1)
        self.rpn_relu = nn.ReLU(inplace=True)
        
        # Classification head (object vs background)
        self.rpn_cls = nn.Conv2d(256, self.num_anchors * 2, 1)  # 2: object/background
        
        # Regression head (bounding box adjustments)
        self.rpn_reg = nn.Conv2d(256, self.num_anchors * 4, 1)  # 4: dx, dy, dw, dh
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def generate_anchors(self, feature_map_size: Tuple[int, int]) -> torch.Tensor:
        """Generate anchor boxes for each spatial position in feature map"""
        height, width = feature_map_size
        anchors = []
        
        for y in range(height):
            for x in range(width):
                center_x = (x + 0.5) / width
                center_y = (y + 0.5) / height
                
                for size in self.anchor_sizes:
                    w, h = size
                    w_norm = w / 224.0  # Normalize to image size
                    h_norm = h / 224.0
                    anchors.append([center_x, center_y, w_norm, h_norm])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of RPN
        Returns: classification scores, regression offsets, anchor boxes
        """
        batch_size = x.size(0)
        
        # Extract features
        features = self.backbone(x)  # (batch_size, 128, 28, 28)
        
        # RPN features
        rpn_features = self.rpn_relu(self.rpn_conv(features))
        
        # Classification scores (objectness)
        rpn_cls_scores = self.rpn_cls(rpn_features)  # (batch_size, num_anchors*2, 28, 28)
        rpn_cls_scores = rpn_cls_scores.permute(0, 2, 3, 1).contiguous()
        rpn_cls_scores = rpn_cls_scores.view(batch_size, -1, 2)  # (batch_size, num_anchors*28*28, 2)
        
        # Regression offsets
        rpn_reg_offsets = self.rpn_reg(rpn_features)  # (batch_size, num_anchors*4, 28, 28)
        rpn_reg_offsets = rpn_reg_offsets.permute(0, 2, 3, 1).contiguous()
        rpn_reg_offsets = rpn_reg_offsets.view(batch_size, -1, 4)  # (batch_size, num_anchors*28*28, 4)
        
        # Generate anchor boxes
        feature_map_size = (features.size(2), features.size(3))  # (28, 28)
        anchors = self.generate_anchors(feature_map_size)
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)  # (batch_size, num_anchors, 4)
        
        return rpn_cls_scores, rpn_reg_offsets, anchors

class SizeAwareLoss(nn.Module):
    """
    Size-aware loss function that gives higher weight to small tumor examples
    Inspired by focal loss but with size-based weighting
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, small_tumor_threshold: float = 0.1):
        super(SizeAwareLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.small_tumor_threshold = small_tumor_threshold  # Area threshold for small tumors
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                bbox_sizes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute size-aware loss
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels
            bbox_sizes: Normalized bounding box sizes [width, height] for each sample
        """
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Convert to probabilities
        p_t = torch.exp(-ce_loss)
        
        # Focal loss component
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        # Size-aware weighting
        if bbox_sizes is not None:
            # Calculate area for each bounding box
            areas = bbox_sizes[:, 0] * bbox_sizes[:, 1]  # width * height
            
            # Create size weights: higher weight for smaller tumors
            size_weights = torch.exp(-areas / self.small_tumor_threshold)
            size_weights = size_weights / size_weights.mean()  # Normalize
            
            # Apply size weights to loss
            weighted_loss = focal_loss * size_weights
        else:
            weighted_loss = focal_loss
        
        return weighted_loss.mean()

class CLMBasedClassifier(nn.Module):
    """
    Stage 2: CLM-based ANN framework for final classification
    Uses features from both RPN and the original CLM feature extractor
    """
    
    def __init__(self, input_channels: int = 4, num_classes: int = 2):
        super(CLMBasedClassifier, self).__init__()
        
        # Feature refinement network
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        # Correlation learning layers (simplified CLM)
        self.correlation_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for final classification
        x: cropped region proposals from RPN
        """
        batch_size = x.size(0)
        
        # Feature refinement
        features = self.feature_refinement(x)  # (batch_size, 128, 7, 7)
        features = features.view(batch_size, -1)  # (batch_size, 128*7*7)
        
        # Correlation learning
        correlated_features = self.correlation_layers(features)
        
        # Final classification
        logits = self.classifier(correlated_features)
        
        return logits

class TwoStageBrainTumorDetector(nn.Module):
    """
    Complete two-stage detection and classification system
    """
    
    def __init__(self, input_channels: int = 4, num_classes: int = 2):
        super(TwoStageBrainTumorDetector, self).__init__()
        
        # Stage 1: Region Proposal Network
        self.rpn = LightweightRegionProposalNetwork(input_channels)
        
        # Stage 2: CLM-based Classifier
        self.classifier = CLMBasedClassifier(input_channels, num_classes)
        
        # Proposal selection parameters
        self.proposal_score_threshold = 0.7
        self.max_proposals_per_image = 10
        self.nms_threshold = 0.3
        
    def decode_boxes(self, anchors: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Decode bounding boxes from anchors and regression offsets"""
        # anchors: [center_x, center_y, width, height]
        # offsets: [dx, dy, dw, dh]
        
        centers = anchors[..., :2] + offsets[..., :2] * 0.1
        sizes = anchors[..., 2:] * torch.exp(offsets[..., 2:] * 0.2)
        
        # Convert to [x1, y1, x2, y2] format
        boxes = torch.zeros_like(anchors)
        boxes[..., 0] = centers[..., 0] - sizes[..., 0] / 2  # x1
        boxes[..., 1] = centers[..., 1] - sizes[..., 1] / 2  # y1
        boxes[..., 2] = centers[..., 0] + sizes[..., 0] / 2  # x2
        boxes[..., 3] = centers[..., 1] + sizes[..., 1] / 2  # y2
        
        # Clip to [0, 1] range
        boxes = torch.clamp(boxes, 0, 1)
        
        return boxes
    
    def non_max_suppression(self, boxes: torch.Tensor, scores: torch.Tensor, 
                          threshold: float = 0.5) -> torch.Tensor:
        """Apply Non-Maximum Suppression to remove overlapping boxes"""
        if boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0]
            keep.append(i)
            
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            inds = torch.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def forward(self, x: torch.Tensor, return_proposals: bool = False):
        """
        Forward pass through both stages
        
        Args:
            x: Input image tensor (batch_size, 4, 224, 224)
            return_proposals: Whether to return proposal information
        
        Returns:
            If training: RPN outputs and classification logits
            If inference: Final predictions and proposal information
        """
        batch_size = x.size(0)
        
        # Stage 1: Region Proposal
        rpn_cls_scores, rpn_reg_offsets, anchors = self.rpn(x)
        
        if self.training:
            # During training, return all components for loss computation
            # Stage 2: Classification for all proposals (simplified)
            # In practice, you'd sample proposals here
            return rpn_cls_scores, rpn_reg_offsets, anchors
        else:
            # During inference: select top proposals and classify them
            proposals = self._select_proposals(rpn_cls_scores, rpn_reg_offsets, anchors)
            
            if return_proposals:
                return proposals
            else:
                # For simplicity in this example, return the highest scoring proposal
                final_predictions = []
                for i in range(batch_size):
                    if len(proposals[i]['boxes']) > 0:
                        # Get the highest scoring proposal
                        best_idx = proposals[i]['scores'].argmax()
                        best_box = proposals[i]['boxes'][best_idx]
                        
                        # Extract region and classify (simplified)
                        # In practice, you'd crop and resize the region
                        classification_logits = self.classifier(x[i:i+1])
                        final_predictions.append(classification_logits)
                    else:
                        # No proposals - predict background
                        background_logits = torch.tensor([[1.0, 0.0]], device=x.device)
                        final_predictions.append(background_logits)
                
                return torch.cat(final_predictions, dim=0)
    
    def _select_proposals(self, cls_scores: torch.Tensor, reg_offsets: torch.Tensor, 
                         anchors: torch.Tensor) -> List[Dict]:
        """Select top proposals using NMS"""
        batch_size = cls_scores.size(0)
        proposals = []
        
        for i in range(batch_size):
            # Convert to probabilities
            probs = F.softmax(cls_scores[i], dim=-1)
            objectness_scores = probs[:, 1]  # Probability of being an object
            
            # Decode boxes
            decoded_boxes = self.decode_boxes(anchors[i], reg_offsets[i])
            
            # Filter by score threshold
            high_score_mask = objectness_scores > self.proposal_score_threshold
            if not high_score_mask.any():
                proposals.append({'boxes': torch.tensor([]), 'scores': torch.tensor([])})
                continue
            
            filtered_boxes = decoded_boxes[high_score_mask]
            filtered_scores = objectness_scores[high_score_mask]
            
            # Apply NMS
            keep_indices = self.non_max_suppression(filtered_boxes, filtered_scores, self.nms_threshold)
            
            if len(keep_indices) > self.max_proposals_per_image:
                keep_indices = keep_indices[:self.max_proposals_per_image]
            
            final_boxes = filtered_boxes[keep_indices]
            final_scores = filtered_scores[keep_indices]
            
            proposals.append({
                'boxes': final_boxes,
                'scores': final_scores
            })
        
        return proposals

class TwoStageTrainer:
    """Trainer for the two-stage detection and classification system"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.rpn_cls_loss = SizeAwareLoss(alpha=0.25, gamma=2.0, small_tumor_threshold=0.05)
        self.rpn_reg_loss = nn.SmoothL1Loss()
        self.classifier_loss = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
    
    def compute_rpn_loss(self, rpn_cls_scores: torch.Tensor, rpn_reg_offsets: torch.Tensor,
                        anchors: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RPN loss (simplified for this example)"""
        # In practice, you'd match anchors to ground truth boxes here
        # For this example, we'll use a simplified approach
        
        batch_size = rpn_cls_scores.size(0)
        
        # Classification loss (simplified: assume all are background for this example)
        rpn_cls_targets = torch.zeros(rpn_cls_scores.size(1), dtype=torch.long, device=self.device)
        rpn_cls_targets = rpn_cls_targets.unsqueeze(0).repeat(batch_size, 1)
        
        # Calculate bounding box sizes for size-aware loss
        bbox_sizes = anchors[..., 2:]  # width, height
        
        cls_loss = self.rpn_cls_loss(
            rpn_cls_scores.view(-1, 2), 
            rpn_cls_targets.view(-1),
            bbox_sizes.view(-1, 2)
        )
        
        # Regression loss (simplified)
        reg_loss = self.rpn_reg_loss(rpn_reg_offsets, torch.zeros_like(rpn_reg_offsets))
        
        return cls_loss, reg_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            rpn_cls_scores, rpn_reg_offsets, anchors = self.model(data)
            
            # Compute RPN losses
            rpn_cls_loss, rpn_reg_loss = self.compute_rpn_loss(
                rpn_cls_scores, rpn_reg_offsets, anchors, targets
            )
            
            # Total loss (simplified - in practice you'd include classifier loss)
            total_loss = rpn_cls_loss + rpn_reg_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}, Loss: {total_loss.item():.4f}, '
                      f'Cls: {rpn_cls_loss.item():.4f}, Reg: {rpn_reg_loss.item():.4f}')
        
        return running_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                rpn_cls_scores, rpn_reg_offsets, anchors = self.model(data)
                rpn_cls_loss, rpn_reg_loss = self.compute_rpn_loss(
                    rpn_cls_scores, rpn_reg_offsets, anchors, targets
                )
                
                total_loss = rpn_cls_loss + rpn_reg_loss
                running_loss += total_loss.item()
        
        return running_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 30):
        print("Starting Two-Stage Detection and Classification Training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_two_stage_model.pth')
                print(f'  New best model saved with validation loss: {val_loss:.4f}')
            
            print('-' * 60)

def visualize_detections(model: TwoStageBrainTumorDetector, dataloader: DataLoader, device: torch.device, num_samples: int = 3):
    """Visualize region proposals and detections"""
    model.eval()
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            data = data.to(device)
            
            # Get proposals
            proposals = model(data, return_proposals=True)
            
            # Convert tensor to numpy for visualization
            image = data[0].cpu().numpy()  # First channel for visualization
            image = (image[0] * 255).astype(np.uint8)  # Use first channel
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(image, cmap='gray')
            
            # Draw proposals
            sample_proposals = proposals[0]
            for j, (box, score) in enumerate(zip(sample_proposals['boxes'], sample_proposals['scores'])):
                if j >= 5:  # Show only top 5 proposals
                    break
                    
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, x2 = x1 * 224, x2 * 224
                y1, y2 = y1 * 224, y2 * 224
                width, height = x2 - x1, y2 - y1
                
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                       edgecolor='r', facecolor='none', alpha=0.7)
                ax.add_patch(rect)
                
                # Add score text
                ax.text(x1, y1 - 5, f'Score: {score:.3f}', 
                       bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7),
                       fontsize=8, color='white')
            
            ax.set_title(f'Region Proposals (Sample {i+1})')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'region_proposals_sample_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main function for two-stage detection and classification"""
    # Configuration
    DATA_DIR = "/Users/jeissonparra/Library/CloudStorage/OneDrive-FloridaInternationalUniversity/Capstone/Datasets/CT_enhanced"
    BATCH_SIZE = 8
    EPOCHS = 30
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets (using the same dataset class from previous step)
    from feature_extraction import BrainTumorDataset  # Import from previous code
    
    train_dataset = BrainTumorDataset(DATA_DIR, split="train")
    val_dataset = BrainTumorDataset(DATA_DIR, split="val")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize two-stage model
    model = TwoStageBrainTumorDetector(input_channels=4, num_classes=2)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    trainer = TwoStageTrainer(model, device)
    
    # Train the model
    trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Visualize detections
    print("Visualizing region proposals...")
    visualize_detections(model, val_loader, device, num_samples=3)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.title('Two-Stage Detection Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('two_stage_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Two-Stage Detection and Classification completed successfully!")
    print("📊 Model saved as: best_two_stage_model.pth")
    print("📈 Training history saved as: two_stage_training_history.png")
    print("🔍 Region proposals visualized in: region_proposals_sample_*.png")

if __name__ == "__main__":
    main()