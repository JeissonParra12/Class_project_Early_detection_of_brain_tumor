import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Generates heatmaps showing regions that influence model decisions
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for given input and target class
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class for which to generate heatmap (None for predicted class)
        
        Returns:
            Heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_class] = 1
        output.backward(gradient=one_hot_output)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def overlay_heatmap(self, original_image: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            original_image: Original image (H, W) or (H, W, 3)
            heatmap: Grad-CAM heatmap
            alpha: Transparency for heatmap
            colormap: OpenCV colormap
        
        Returns:
            Image with overlaid heatmap
        """
        # Ensure original image is 2D
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original_image
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_gray.shape[1], original_gray.shape[0]))
        
        # Convert heatmap to colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert original to RGB if grayscale
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_image
        
        # Overlay heatmap
        overlayed = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation of the brain tumor detection model
    Includes both conventional metrics and lesion-level analysis
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.results = {}
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': recall_score(1 - y_true, 1 - y_pred, average='binary')  # TN / (TN + FP)
        }
        
        # ROC curve metrics
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['auc_roc'] = auc(fpr, tpr)
        
        # Precision-Recall curve metrics
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['auc_pr'] = auc(recall, precision)
        
        return metrics
    
    def calculate_lesion_level_metrics(self, predictions: List[Dict], 
                                     ground_truth: List[Dict]) -> Dict[str, float]:
        """
        Calculate lesion-level detection metrics
        Specifically for evaluating small/early-stage tumor detection
        """
        # Simplified lesion-level analysis
        # In practice, you'd have bounding box IoU calculations here
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(predictions, ground_truth):
            # For this example, we'll use a simplified approach
            # In real implementation, you'd calculate IoU between predicted and ground truth boxes
            
            if pred['has_tumor'] and gt['has_tumor']:
                true_positives += 1
            elif pred['has_tumor'] and not gt['has_tumor']:
                false_positives += 1
            elif not pred['has_tumor'] and gt['has_tumor']:
                false_negatives += 1
        
        # Calculate lesion-level sensitivity (recall)
        if true_positives + false_negatives > 0:
            lesion_sensitivity = true_positives / (true_positives + false_negatives)
        else:
            lesion_sensitivity = 0.0
        
        # Calculate lesion-level precision
        if true_positives + false_positives > 0:
            lesion_precision = true_positives / (true_positives + false_positives)
        else:
            lesion_precision = 0.0
        
        # Calculate F1-score for lesions
        if lesion_precision + lesion_sensitivity > 0:
            lesion_f1 = 2 * (lesion_precision * lesion_sensitivity) / (lesion_precision + lesion_sensitivity)
        else:
            lesion_f1 = 0.0
        
        return {
            'lesion_sensitivity': lesion_sensitivity,
            'lesion_precision': lesion_precision,
            'lesion_f1_score': lesion_f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_model(self, dataloader: DataLoader, 
                      lesion_annotations: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            dataloader: DataLoader for evaluation data
            lesion_annotations: Optional lesion-level annotations
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        lesion_predictions = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get model predictions
                if hasattr(self.model, 'module'):
                    # For DataParallel models
                    outputs = self.model.module(data)
                else:
                    outputs = self.model(data)
                
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = outputs.argmax(dim=1)
                
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Tumor class probability
                all_targets.extend(targets.cpu().numpy())
                
                # Store lesion-level predictions
                for i in range(len(data)):
                    lesion_predictions.append({
                        'has_tumor': predicted_classes[i].item() == 1,
                        'confidence': probabilities[i, 1].item(),
                        'image_id': batch_idx * dataloader.batch_size + i
                    })
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate basic metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)
        
        # Calculate lesion-level metrics if annotations provided
        lesion_metrics = {}
        if lesion_annotations:
            lesion_metrics = self.calculate_lesion_level_metrics(lesion_predictions, lesion_annotations)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Comprehensive results
        self.results = {
            'basic_metrics': basic_metrics,
            'lesion_metrics': lesion_metrics,
            'confusion_matrix': cm,
            'predictions': {
                'true': y_true,
                'pred': y_pred,
                'prob': y_prob
            },
            'classification_report': classification_report(y_true, y_pred, 
                                                          target_names=['Normal', 'Tumor'])
        }
        
        return self.results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Tumor'],
                   yticklabels=['Normal', 'Tumor'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, save_path: str = None):
        """Generate comprehensive evaluation report"""
        if not self.results:
            print("No evaluation results available. Run evaluate_model() first.")
            return
        
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE BRAIN TUMOR DETECTION EVALUATION REPORT")
        report.append("=" * 60)
        
        # Basic metrics
        basic_metrics = self.results['basic_metrics']
        report.append("\nBASIC CLASSIFICATION METRICS:")
        report.append("-" * 40)
        for metric, value in basic_metrics.items():
            report.append(f"{metric.replace('_', ' ').title():<20}: {value:.4f}")
        
        # Lesion metrics
        if self.results['lesion_metrics']:
            lesion_metrics = self.results['lesion_metrics']
            report.append("\nLESION-LEVEL DETECTION METRICS:")
            report.append("-" * 40)
            for metric, value in lesion_metrics.items():
                if isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title():<20}: {value:.4f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title():<20}: {value}")
        
        # Classification report
        report.append("\nDETAILED CLASSIFICATION REPORT:")
        report.append("-" * 40)
        report.append(self.results['classification_report'])
        
        report.append("\n" + "=" * 60)
        
        full_report = "\n".join(report)
        print(full_report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
        
        return full_report

class CrossValidator:
    """
    Cross-validation for robustness and generalizability assessment
    """
    
    def __init__(self, model_class: nn.Module, dataset: Dataset, device: torch.device, n_splits: int = 5):
        self.model_class = model_class
        self.dataset = dataset
        self.device = device
        self.n_splits = n_splits
        self.cv_results = []
    
    def perform_cross_validation(self, epochs: int = 10, batch_size: int = 8):
        """Perform k-fold cross-validation"""
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Get targets for stratification
        targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(self.dataset)), targets)):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{self.n_splits}")
            print(f"{'='*50}")
            
            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_subsampler)
            val_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=val_subsampler)
            
            # Initialize model
            model = self.model_class(input_channels=4, num_classes=2)
            model.to(self.device)
            
            # Train model (simplified training for demonstration)
            # In practice, use your full training pipeline
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Simple training loop
            for epoch in range(epochs):
                model.train()
                for data, targets in train_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            evaluator = ComprehensiveEvaluator(model, self.device)
            results = evaluator.evaluate_model(val_loader)
            
            fold_metrics.append(results['basic_metrics'])
            
            print(f"Fold {fold + 1} Results:")
            for metric, value in results['basic_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        # Aggregate results
        self.cv_results = fold_metrics
        return self.aggregate_cv_results()
    
    def aggregate_cv_results(self) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        metrics_df = pd.DataFrame(self.cv_results)
        
        summary = {
            'mean': metrics_df.mean(),
            'std': metrics_df.std(),
            'min': metrics_df.min(),
            'max': metrics_df.max()
        }
        
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print("="*60)
        for metric in metrics_df.columns:
            print(f"{metric}: {summary['mean'][metric]:.4f} ± {summary['std'][metric]:.4f} "
                  f"(min: {summary['min'][metric]:.4f}, max: {summary['max'][metric]:.4f})")
        
        return summary

def visualize_grad_cam_explanations(model: nn.Module, dataloader: DataLoader, device: torch.device, 
                                  num_samples: int = 5, target_layer: nn.Module = None):
    """
    Generate and visualize Grad-CAM explanations for model predictions
    """
    if target_layer is None:
        # Try to find a suitable convolutional layer
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
    
    if target_layer is None:
        print("No suitable convolutional layer found for Grad-CAM")
        return
    
    grad_cam = GradCAM(model, target_layer)
    
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for data, targets in dataloader:
            if samples_processed >= num_samples:
                break
                
            for i in range(len(data)):
                if samples_processed >= num_samples:
                    break
                
                # Get single sample
                sample = data[i:i+1].to(device)
                target = targets[i].item()
                
                # Get prediction
                output = model(sample)
                prediction = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, prediction].item()
                
                # Generate Grad-CAM heatmap
                heatmap = grad_cam.generate_heatmap(sample, prediction)
                
                # Get original image (first channel)
                original_image = sample[0, 0].cpu().numpy()  # First channel
                original_image = (original_image * 255).astype(np.uint8)
                
                # Overlay heatmap
                overlayed_image = grad_cam.overlay_heatmap(original_image, heatmap)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(original_image, cmap='gray')
                axes[0].set_title(f'Original CT Scan\nTrue: {"Tumor" if target == 1 else "Normal"}')
                axes[0].axis('off')
                
                # Heatmap
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap\n(Activation Regions)')
                axes[1].axis('off')
                
                # Overlay
                axes[2].imshow(overlayed_image)
                axes[2].set_title(f'Overlay\nPred: {"Tumor" if prediction == 1 else "Normal"} '
                                f'(Conf: {confidence:.3f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'grad_cam_explanation_{samples_processed + 1}.png', 
                          dpi=300, bbox_inches='tight')
                plt.show()
                
                samples_processed += 1
                
                print(f"Sample {samples_processed}: "
                      f"True: {'Tumor' if target == 1 else 'Normal'}, "
                      f"Pred: {'Tumor' if prediction == 1 else 'Normal'}, "
                      f"Confidence: {confidence:.3f}")

def main():
    """Main function for comprehensive evaluation and explainability"""
    # Configuration
    DATA_DIR = "/Users/jeissonparra/Library/CloudStorage/OneDrive-FloridaInternationalUniversity/Capstone/Datasets/CT_enhanced"
    MODEL_PATH = "best_two_stage_model.pth"  # Or your trained model path
    BATCH_SIZE = 16
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model (using the TwoStageBrainTumorDetector from previous step)
    from two_stage_detection import TwoStageBrainTumorDetector, BrainTumorDataset
    
    model = TwoStageBrainTumorDetector(input_channels=4, num_classes=2)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("⚠️  Model file not found. Using untrained model for demonstration.")
    
    model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = BrainTumorDataset(DATA_DIR, split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Comprehensive Evaluation
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    evaluator = ComprehensiveEvaluator(model, device)
    results = evaluator.evaluate_model(test_loader)
    
    # Generate reports and visualizations
    evaluator.generate_comprehensive_report("evaluation_report.txt")
    
    # Plot metrics
    y_true = results['predictions']['true']
    y_prob = results['predictions']['prob']
    
    evaluator.plot_confusion_matrix(results['confusion_matrix'], "confusion_matrix.png")
    evaluator.plot_roc_curve(y_true, y_prob, "roc_curve.png")
    evaluator.plot_precision_recall_curve(y_true, y_prob, "precision_recall_curve.png")
    
    # Grad-CAM Explanations
    print("\n" + "="*60)
    print("GRAD-CAM EXPLANATIONS")
    print("="*60)
    
    # Find a suitable layer for Grad-CAM (using the RPN backbone)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'backbone' in name:
            target_layer = module
            break
    
    if target_layer:
        visualize_grad_cam_explanations(model, test_loader, device, 
                                      num_samples=5, target_layer=target_layer)
    else:
        print("Could not find suitable layer for Grad-CAM")
    
    # Cross-Validation (optional - can be time-consuming)
    print("\n" + "="*60)
    print("CROSS-VALIDATION FOR ROBUSTNESS ASSESSMENT")
    print("="*60)
    
    # Note: This can be time-consuming, so we'll use a smaller subset for demonstration
    full_dataset = BrainTumorDataset(DATA_DIR, split="train")
    
    # Use smaller subset for faster demonstration
    from torch.utils.data import Subset
    subset_indices = list(range(min(100, len(full_dataset))))  # Use first 100 samples
    subset_dataset = Subset(full_dataset, subset_indices)
    
    cross_validator = CrossValidator(TwoStageBrainTumorDetector, subset_dataset, device, n_splits=3)
    cv_results = cross_validator.perform_cross_validation(epochs=5, batch_size=8)
    
    print("\n✅ Comprehensive Evaluation Completed!")
    print("📊 Evaluation report saved as: evaluation_report.txt")
    print("📈 Visualizations saved as: confusion_matrix.png, roc_curve.png, precision_recall_curve.png")
    print("🔍 Grad-CAM explanations saved as: grad_cam_explanation_*.png")
    print("📋 Cross-validation results printed above")

if __name__ == "__main__":
    main()