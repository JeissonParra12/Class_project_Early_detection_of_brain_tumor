import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class CorrelationLearningMechanism(nn.Module):
    """
    Correlation Learning Mechanism (CLM) inspired by Woźniak et al. (2023)
    Dynamically filters convolutional layer combinations and evaluates feature correlations
    """
    
    def __init__(self, input_channels: int = 4, num_classes: int = 2):
        super(CorrelationLearningMechanism, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Multi-scale feature extraction branches
        self.branch_configs = self._create_branch_configurations()
        
        # Dynamic convolutional filter banks
        self.conv_filters = nn.ModuleDict()
        self._initialize_conv_filters()
        
        # Calculate the actual output channels from conv_filters
        total_output_channels = 0
        for name in self.conv_filters:
            if 'standard_3x3' in name:
                total_output_channels += 32
            elif 'standard_5x5' in name:
                total_output_channels += 32
            elif 'depthwise_3x3' in name:
                total_output_channels += 32
            elif 'dilated_3x3' in name:
                total_output_channels += 32
            elif 'asymmetric_1x3' in name:
                total_output_channels += 32
        
        print(f"Total output channels from conv filters: {total_output_channels}")
        
        # Correlation learning components - FIXED CHANNEL DIMENSIONS
        self.correlation_net = CorrelationNetwork(total_output_channels, 256)
        self.feature_selector = FastCorrelationFeatureSelector(256, 128)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def _create_branch_configurations(self) -> List[Dict]:
        """Create different branch configurations for dynamic filtering"""
        configs = [
            # Branch 1: Standard convolution
            {'filters': 32, 'kernel_size': 3, 'pool_type': 'max', 'activation': 'relu'},
            # Branch 2: Depth-wise separable convolution
            {'filters': 64, 'kernel_size': 5, 'pool_type': 'avg', 'activation': 'relu'},
            # Branch 3: Dilated convolution for larger receptive field
            {'filters': 32, 'kernel_size': 3, 'dilation': 2, 'pool_type': 'max', 'activation': 'leaky_relu'},
            # Branch 4: Asymmetric convolution
            {'filters': 64, 'kernel_size': (1, 3), 'pool_type': 'avg', 'activation': 'relu'},
        ]
        return configs
    
    def _initialize_conv_filters(self):
        """Initialize different convolutional filter types"""
        # Standard convolutional layers
        self.conv_filters['standard_3x3'] = nn.Conv2d(self.input_channels, 32, 3, padding=1)
        self.conv_filters['standard_5x5'] = nn.Conv2d(self.input_channels, 32, 5, padding=2)
        
        # Depth-wise separable convolutions
        self.conv_filters['depthwise_3x3'] = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels, 3, padding=1, groups=self.input_channels),
            nn.Conv2d(self.input_channels, 32, 1)
        )
        
        # Dilated convolutions
        self.conv_filters['dilated_3x3'] = nn.Conv2d(self.input_channels, 32, 3, padding=2, dilation=2)
        
        # Asymmetric convolutions
        self.conv_filters['asymmetric_1x3'] = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, (1, 3), padding=(0, 1)),
            nn.Conv2d(32, 32, (3, 1), padding=(1, 0))
        )
    
    def _apply_dynamic_pooling(self, x: torch.Tensor, pool_type: str) -> torch.Tensor:
        """Apply dynamic pooling operations"""
        if pool_type == 'max':
            return F.adaptive_max_pool2d(x, (x.size(2)//2, x.size(3)//2))
        elif pool_type == 'avg':
            return F.adaptive_avg_pool2d(x, (x.size(2)//2, x.size(3)//2))
        elif pool_type == 'mixed':
            max_pool = F.adaptive_max_pool2d(x, (x.size(2)//2, x.size(3)//2))
            avg_pool = F.adaptive_avg_pool2d(x, (x.size(2)//2, x.size(3)//2))
            return (max_pool + avg_pool) / 2
        else:
            return x
    
    def _apply_activation(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        """Apply dynamic activation functions"""
        if activation == 'relu':
            return F.relu(x)
        elif activation == 'leaky_relu':
            return F.leaky_relu(x, 0.1)
        elif activation == 'elu':
            return F.elu(x)
        elif activation == 'selu':
            return F.selu(x)
        else:
            return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CLM
        Returns: tuple of (features, classification_logits)
        """
        batch_size = x.size(0)
        
        # Extract features using multiple filter configurations
        branch_outputs = []
        
        for name, filter_module in self.conv_filters.items():
            # Apply convolutional filter
            filtered = filter_module(x)
            
            # Apply pooling based on filter type
            if 'dilated' in name:
                pooled = self._apply_dynamic_pooling(filtered, 'max')
            elif 'depthwise' in name:
                pooled = self._apply_dynamic_pooling(filtered, 'avg')
            else:
                pooled = self._apply_dynamic_pooling(filtered, 'mixed')
            
            # Apply activation
            if 'leaky' in name:
                activated = self._apply_activation(pooled, 'leaky_relu')
            else:
                activated = self._apply_activation(pooled, 'relu')
            
            branch_outputs.append(activated)
        
        # Concatenate all branch outputs
        concatenated_features = torch.cat(branch_outputs, dim=1)
        
        print(f"Concatenated features shape: {concatenated_features.shape}")  # Debug
        
        # Apply correlation learning
        correlated_features = self.correlation_net(concatenated_features)
        
        # Apply fast correlation feature selection
        selected_features = self.feature_selector(correlated_features)
        
        # Global average pooling
        global_features = F.adaptive_avg_pool2d(selected_features, (1, 1))
        global_features = global_features.view(batch_size, -1)
        
        # Classification
        classification_logits = self.classifier(global_features)
        
        return global_features, classification_logits

class CorrelationNetwork(nn.Module):
    """
    Neural network component that evaluates and correlates CNN outputs
    to improve feature relevance and classification confidence
    """
    
    def __init__(self, input_channels: int, output_channels: int):
        super(CorrelationNetwork, self).__init__()
        
        # FIXED: Use the actual input_channels instead of hardcoded 512
        self.correlation_layers = nn.Sequential(
            nn.Conv2d(input_channels, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # Attention mechanism for feature correlation
        self.attention = CorrelationAttention(output_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.correlation_layers(x)
        correlated_features = self.attention(features)
        return correlated_features

class CorrelationAttention(nn.Module):
    """
    Attention mechanism that learns correlations between feature maps
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(CorrelationAttention, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial attention
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        x_spatial = x_channel * spatial_weights
        
        return x_spatial

class FastCorrelationFeatureSelector(nn.Module):
    """
    Fast-correlation filter-based automatic feature selection
    Avoids redundancy in features as mentioned in the research
    """
    
    def __init__(self, input_channels: int, output_channels: int):
        super(FastCorrelationFeatureSelector, self).__init__()
        
        self.selector = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 1),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(input_channels // 2, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # Learnable feature importance weights
        self.feature_importance = nn.Parameter(torch.ones(input_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply feature importance weights
        weighted_x = x * self.feature_importance.view(1, -1, 1, 1)
        
        # Feature selection
        selected_features = self.selector(weighted_x)
        return selected_features

class BrainTumorDataset(Dataset):
    """Dataset class for preprocessed brain tumor CT scans"""
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Collect all processed files
        self.samples = []
        for label in ["tumor", "normal"]:
            label_dir = self.data_dir / split / label
            if label_dir.exists():
                for file_path in label_dir.glob("*.npy"):
                    self.samples.append((file_path, 1 if label == "tumor" else 0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load preprocessed multi-scale data
        data = np.load(file_path)  # Shape: (H, W, 4) - multi-channel
        data = data.transpose(2, 0, 1)  # Convert to (4, H, W)
        
        # Convert to tensor
        data = torch.FloatTensor(data)
        label = torch.LongTensor([label]).squeeze()
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

class FeatureExtractionTrainer:
    """
    Trainer class for the feature extraction and correlation learning step
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        
        # Loss function with class weighting for imbalance
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            features, outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                features, outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        print("Starting Feature Extraction and Correlation Learning Training...")
        
        best_val_accuracy = 0.0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                }, 'best_clm_model.pth')
                print(f'  New best model saved with validation accuracy: {val_acc:.2f}%')
            
            print('-' * 60)
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('clm_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_feature_correlations(model: CorrelationLearningMechanism, dataloader: DataLoader, device: torch.device):
    """
    Analyze feature correlations learned by the CLM
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            features, _ = model(data)
            all_features.append(features.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    
    print(f"Extracted features shape: {all_features.shape}")
    
    # Calculate feature correlations
    correlation_matrix = np.corrcoef(all_features.T)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Feature Correlation Matrix')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_features, all_labels

def main():
    """Main function to run feature extraction and correlation learning"""
    # Configuration
    DATA_DIR = "/Users/jeissonparra/Library/CloudStorage/OneDrive-FloridaInternationalUniversity/Special_Topics_Advanced_Computational_Methods_in_Health_and_Biomedical_Data/Class_project_Early_detection_of_brain_tumor/CT_enhanced"
    BATCH_SIZE = 16
    EPOCHS = 50
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = BrainTumorDataset(DATA_DIR, split="train")
    val_dataset = BrainTumorDataset(DATA_DIR, split="val")
    test_dataset = BrainTumorDataset(DATA_DIR, split="test")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize CLM model
    model = CorrelationLearningMechanism(input_channels=4, num_classes=2)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test a forward pass with a single batch to verify the fix
    print("Testing forward pass with a single batch...")
    with torch.no_grad():
        test_batch, test_targets = next(iter(train_loader))
        test_batch = test_batch.to(device)
        features, outputs = model(test_batch)
        print(f"Forward pass successful! Features shape: {features.shape}, Outputs shape: {outputs.shape}")
    
    # Initialize trainer
    trainer = FeatureExtractionTrainer(model, device)
    
    # Train the model
    trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Analyze feature correlations
    print("Analyzing feature correlations...")
    features, labels = analyze_feature_correlations(model, val_loader, device)
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_clm_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    test_loss, test_accuracy = trainer.validate_epoch(test_loader)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    print("\n✅ Feature Extraction and Correlation Learning completed successfully!")
    print("📊 Model saved as: best_clm_model.pth")
    print("📈 Training history saved as: clm_training_history.png")
    print("🔍 Feature correlations saved as: feature_correlations.png")

if __name__ == "__main__":
    main()