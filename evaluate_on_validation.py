# Final Validation Evaluation Script
# Evaluates all trained models on the held-out validation set
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from training_utils import TrainingMetrics
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architectures
from train_acoustic_enhanced import AcousticBrain
from train_linguistic_enhanced import LinguisticBrain
from train_visual_enhanced import VisualBrain
from train_fusion_enhanced import FusionBrain

def load_model(model_class, checkpoint_path, device):
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return None
    
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_validation_features(data_dir: str = "data", modality: str = "acoustic"):
    """Load validation features for a modality."""
    val_features_file = f"{data_dir}/validation_{modality}_x.npy"
    val_labels_file = f"{data_dir}/validation_{modality}_y.npy"
    
    if not os.path.exists(val_features_file):
        print(f"Warning: Validation features not found: {val_features_file}")
        return None, None
    
    features = np.load(val_features_file)
    labels = np.load(val_labels_file)
    
    return features, labels

def evaluate_acoustic_model(model, val_features, val_labels, device):
    """Evaluate acoustic model on validation set."""
    print("\n" + "=" * 70)
    print("ACOUSTIC MODEL VALIDATION")
    print("=" * 70)
    
    val_tensor = torch.FloatTensor(val_features).to(device)
    
    with torch.no_grad():
        outputs = model(val_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return preds, val_labels

def evaluate_linguistic_model(model, val_features, val_labels, device):
    """Evaluate linguistic model on validation set."""
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "=" * 70)
    print("LINGUISTIC MODEL VALIDATION")
    print("=" * 70)
    
    # Scale validation features using training data for reference
    # Note: In production, you'd save the scaler from training
    scaler = StandardScaler()
    val_features_scaled = scaler.fit_transform(val_features)
    
    val_tensor = torch.FloatTensor(val_features_scaled).to(device)
    
    with torch.no_grad():
        outputs = model(val_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return preds, val_labels

def evaluate_visual_model(model, val_features, val_labels, device):
    """Evaluate visual model on validation set."""
    print("\n" + "=" * 70)
    print("VISUAL MODEL VALIDATION")
    print("=" * 70)
    
    val_tensor = torch.FloatTensor(val_features).to(device)
    
    with torch.no_grad():
        outputs = model(val_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return preds, val_labels

def evaluate_fusion_model(model, val_acoustic, val_linguistic, val_visual, val_labels, device):
    """Evaluate fusion model on validation set."""
    print("\n" + "=" * 70)
    print("FUSION MODEL VALIDATION")
    print("=" * 70)
    
    acoustic_tensor = torch.FloatTensor(val_acoustic).to(device)
    linguistic_tensor = torch.FloatTensor(val_linguistic).to(device)
    visual_tensor = torch.FloatTensor(val_visual).to(device)
    
    with torch.no_grad():
        outputs = model(acoustic_tensor, linguistic_tensor, visual_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return preds, val_labels

def run_validation_evaluation():
    """Run complete validation evaluation on all models."""
    
    print("=" * 70)
    print("FINAL VALIDATION EVALUATION")
    print("Evaluating all trained models on held-out validation set")
    print("=" * 70)
    
    device = torch.device("cpu")
    
    # Check if validation features exist
    validation_files = [
        "data/validation_acoustic_x.npy",
        "data/validation_linguistic_x.npy",
        "data/validation_visual_x.npy"
    ]
    
    missing_files = [f for f in validation_files if not os.path.exists(f)]
    
    if missing_files:
        print("\n⚠️  Missing validation feature files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease extract features for validation split using:")
        print("  1. extract_acoustics.py")
        print("  2. extract_linguistic_features.py")
        print("  3. extract_visuals.py")
        print("\nWith input file: data/dolos_validation_split.csv")
        return
    
    # Load validation data
    print("\n📂 Loading validation data...")
    val_acoustic_features, val_acoustic_labels = load_validation_features("data", "acoustic")
    val_linguistic_features, val_linguistic_labels = load_validation_features("data", "linguistic")
    val_visual_features, val_visual_labels = load_validation_features("data", "visual")
    
    if all([val_acoustic_features is not None, val_linguistic_features is not None, val_visual_features is not None]):
        print(f"  ✓ Acoustic: {len(val_acoustic_labels)} samples")
        print(f"  ✓ Linguistic: {len(val_linguistic_labels)} samples")
        print(f"  ✓ Visual: {len(val_visual_labels)} samples")
    
    results = {}
    
    # Evaluate Acoustic Model
    acoustic_checkpoint = "checkpoints/acoustic_brain_final.pth"
    if os.path.exists(acoustic_checkpoint):
        print(f"\n🔍 Loading acoustic model from {acoustic_checkpoint}...")
        acoustic_model = load_model(AcousticBrain, acoustic_checkpoint, device)
        if acoustic_model:
            preds, labels = evaluate_acoustic_model(acoustic_model, val_acoustic_features, val_acoustic_labels, device)
            metrics_handler = TrainingMetrics("AcousticBrain")
            cm, cm_path = metrics_handler.plot_confusion_matrix(labels, preds, "Validation")
            report = metrics_handler.generate_classification_report(labels, preds, "Validation")
            metrics_handler.save_detailed_report(labels, preds, "Validation")
            
            accuracy = (labels == preds).mean()
            results['Acoustic'] = {
                'accuracy': accuracy,
                'precision': report['Deception']['precision'],
                'recall': report['Deception']['recall'],
                'f1': report['Deception']['f1-score']
            }
    else:
        print(f"⚠️  Acoustic checkpoint not found: {acoustic_checkpoint}")
    
    # Evaluate Linguistic Model
    linguistic_checkpoint = "checkpoints/linguistic_brain_final.pth"
    if os.path.exists(linguistic_checkpoint):
        print(f"\n🔍 Loading linguistic model from {linguistic_checkpoint}...")
        linguistic_model = load_model(LinguisticBrain, linguistic_checkpoint, device)
        if linguistic_model:
            preds, labels = evaluate_linguistic_model(linguistic_model, val_linguistic_features, val_linguistic_labels, device)
            metrics_handler = TrainingMetrics("LinguisticBrain")
            cm, cm_path = metrics_handler.plot_confusion_matrix(labels, preds, "Validation")
            report = metrics_handler.generate_classification_report(labels, preds, "Validation")
            metrics_handler.save_detailed_report(labels, preds, "Validation")
            
            accuracy = (labels == preds).mean()
            results['Linguistic'] = {
                'accuracy': accuracy,
                'precision': report['Deception']['precision'],
                'recall': report['Deception']['recall'],
                'f1': report['Deception']['f1-score']
            }
    else:
        print(f"⚠️  Linguistic checkpoint not found: {linguistic_checkpoint}")
    
    # Evaluate Visual Model
    visual_checkpoint = "checkpoints/visual_brain_final.pth"
    if os.path.exists(visual_checkpoint):
        print(f"\n🔍 Loading visual model from {visual_checkpoint}...")
        visual_model = load_model(VisualBrain, visual_checkpoint, device)
        if visual_model:
            preds, labels = evaluate_visual_model(visual_model, val_visual_features, val_visual_labels, device)
            metrics_handler = TrainingMetrics("VisualBrain")
            cm, cm_path = metrics_handler.plot_confusion_matrix(labels, preds, "Validation")
            report = metrics_handler.generate_classification_report(labels, preds, "Validation")
            metrics_handler.save_detailed_report(labels, preds, "Validation")
            
            accuracy = (labels == preds).mean()
            results['Visual'] = {
                'accuracy': accuracy,
                'precision': report['Deception']['precision'],
                'recall': report['Deception']['recall'],
                'f1': report['Deception']['f1-score']
            }
    else:
        print(f"⚠️  Visual checkpoint not found: {visual_checkpoint}")
    
    # Evaluate Fusion Model
    fusion_checkpoint = "checkpoints/fusion_brain_final.pth"
    if os.path.exists(fusion_checkpoint):
        print(f"\n🔍 Loading fusion model from {fusion_checkpoint}...")
        fusion_model = load_model(FusionBrain, fusion_checkpoint, device)
        if fusion_model:
            preds, labels = evaluate_fusion_model(
                fusion_model, val_acoustic_features, val_linguistic_features, 
                val_visual_features, val_acoustic_labels, device
            )
            metrics_handler = TrainingMetrics("FusionBrain")
            cm, cm_path = metrics_handler.plot_confusion_matrix(labels, preds, "Validation")
            report = metrics_handler.generate_classification_report(labels, preds, "Validation")
            metrics_handler.save_detailed_report(labels, preds, "Validation")
            
            accuracy = (labels == preds).mean()
            results['Fusion'] = {
                'accuracy': accuracy,
                'precision': report['Deception']['precision'],
                'recall': report['Deception']['recall'],
                'f1': report['Deception']['f1-score']
            }
    else:
        print(f"⚠️  Fusion checkpoint not found: {fusion_checkpoint}")
    
    # Print summary and create comparison graph
    if results:
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 70)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        # Create comparison graph
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        precisions = [results[m]['precision'] for m in models]
        recalls = [results[m]['recall'] for m in models]
        f1_scores = [results[m]['f1'] for m in models]
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance on Validation Set', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = "validation_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison graph saved: {output_path}")
        print("=" * 70)
    else:
        print("\n⚠️  No models were evaluated. Check checkpoints directory.")

if __name__ == "__main__":
    run_validation_evaluation()
