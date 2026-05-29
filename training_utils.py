# Training Utilities: Confusion Matrices, Graphs, and Bias Testing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
import os

class TrainingMetrics:
    """Handle confusion matrices, graphs, and bias analysis."""
    
    def __init__(self, model_name: str, output_dir: str = "training_results"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model_output_dir = os.path.join(output_dir, model_name)
        
        # Create output directories
        os.makedirs(self.model_output_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def record_epoch(self, train_loss, train_acc, test_loss, test_acc):
        """Record metrics for each epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['test_loss'].append(test_loss)
        self.history['test_acc'].append(test_acc)
    
    def plot_training_curves(self):
        """Plot training/validation loss and accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curve
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{self.model_name} - Training/Test Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, self.history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title(f'{self.model_name} - Training/Test Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.model_output_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved training curves: {output_path}")
        return output_path
    
    def plot_confusion_matrix(self, y_true, y_pred, dataset_name: str = "Test"):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                   xticklabels=['Truth', 'Deception'],
                   yticklabels=['Truth', 'Deception'])
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_name} - {dataset_name} Confusion Matrix', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.model_output_dir, f'confusion_matrix_{dataset_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm, output_path
    
    def generate_classification_report(self, y_true, y_pred, dataset_name: str = "Test"):
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, 
                                      target_names=['Truth', 'Deception'],
                                      output_dict=True)
        
        return report
    
    def plot_dataset_bias_comparison(self, results_dict: dict):
        """Plot performance comparison across datasets (RLDD vs Dolos bias analysis)."""
        datasets = list(results_dict.keys())
        accuracies = [results_dict[ds]['accuracy'] for ds in datasets]
        precisions = [results_dict[ds]['precision'] for ds in datasets]
        recalls = [results_dict[ds]['recall'] for ds in datasets]
        f1_scores = [results_dict[ds]['f1'] for ds in datasets]
        
        x = np.arange(len(datasets))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rects1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        rects2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        rects3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_name} - Bias Analysis (Dataset Comparison)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for rects in [rects1, rects2, rects3, rects4]:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(self.model_output_dir, 'bias_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved bias analysis graph: {output_path}")
        return output_path
    
    def save_detailed_report(self, y_true, y_pred, dataset_name: str = "Test", 
                            additional_metrics: dict = None):
        """Save a detailed text report with all metrics."""
        cm = confusion_matrix(y_true, y_pred)
        report = self.generate_classification_report(y_true, y_pred, dataset_name)
        
        output_path = os.path.join(self.model_output_dir, f'report_{dataset_name.lower()}.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"{self.model_name.upper()} - {dataset_name.upper()} EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Confusion Matrix
            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 70 + "\n")
            f.write(f"                Predicted Truth    Predicted Deception\n")
            f.write(f"Actual Truth     {cm[0, 0]:>15}    {cm[0, 1]:>15}\n")
            f.write(f"Actual Deception {cm[1, 0]:>15}    {cm[1, 1]:>15}\n\n")
            
            # Classification Report
            f.write("CLASSIFICATION METRICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n")
            f.write("-" * 70 + "\n")
            
            for class_name in ['Truth', 'Deception']:
                metrics = report[class_name]
                f.write(f"{class_name:<20} {metrics['precision']:<15.4f} "
                       f"{metrics['recall']:<15.4f} {metrics['f1-score']:<15.4f}\n")
            
            f.write("-" * 70 + "\n")
            f.write(f"{'Weighted Avg':<20} {report['weighted avg']['precision']:<15.4f} "
                   f"{report['weighted avg']['recall']:<15.4f} "
                   f"{report['weighted avg']['f1-score']:<15.4f}\n\n")
            
            # Overall Accuracy
            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
            f.write(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            
            # Additional metrics if provided
            if additional_metrics:
                f.write("ADDITIONAL METRICS:\n")
                f.write("-" * 70 + "\n")
                for key, value in additional_metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key:<40}: {value:.4f}\n")
                    else:
                        f.write(f"{key:<40}: {value}\n")
        
        print(f"  ✓ Saved detailed report: {output_path}")
        return output_path

# Helper function to evaluate model and create visualizations
def evaluate_model_with_bias_analysis(model, test_loaders_dict, device, model_name: str,
                                      output_dir: str = "training_results"):
    """
    Evaluate model on multiple datasets and perform bias analysis.
    
    Args:
        model: PyTorch model
        test_loaders_dict: Dict of {'dataset_name': DataLoader} for bias testing
        device: torch device
        model_name: Name of the model
        output_dir: Output directory for results
    
    Returns:
        results_dict: Dict with performance metrics per dataset
    """
    import torch
    
    metrics_handler = TrainingMetrics(model_name, output_dir)
    results_dict = {}
    
    model.eval()
    with torch.no_grad():
        for dataset_name, test_loader in test_loaders_dict.items():
            all_preds = []
            all_labels = []
            
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Generate confusion matrix
            cm, cm_path = metrics_handler.plot_confusion_matrix(all_labels, all_preds, dataset_name)
            
            # Generate classification report
            report = metrics_handler.generate_classification_report(all_labels, all_preds, dataset_name)
            
            # Calculate metrics
            accuracy = (all_labels == all_preds).mean()
            precision = report['Deception']['precision']  # Deception = class 1
            recall = report['Deception']['recall']
            f1 = report['Deception']['f1-score']
            
            results_dict[dataset_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'report': report
            }
            
            # Save detailed report
            metrics_handler.save_detailed_report(all_labels, all_preds, dataset_name)
    
    return metrics_handler, results_dict
