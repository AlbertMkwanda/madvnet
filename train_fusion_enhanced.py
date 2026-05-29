# Enhanced Fusion Training with Bias Testing and Visualization
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from training_utils import TrainingMetrics, evaluate_model_with_bias_analysis

# Fusion Model combining acoustic, linguistic, and visual features
class FusionBrain(nn.Module):
    def __init__(self, acoustic_size=692, linguistic_size=775, visual_size=512):
        super(FusionBrain, self).__init__()
        
        # Acoustic pathway
        self.acoustic_path = nn.Sequential(
            nn.Linear(acoustic_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128)
        )
        
        # Linguistic pathway
        self.linguistic_path = nn.Sequential(
            nn.Linear(linguistic_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128)
        )
        
        # Visual pathway
        self.visual_path = nn.Sequential(
            nn.Linear(visual_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Fusion layer (concatenate all pathways)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),  # 128 from each modality = 384
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Final Output: [Truth, Deception]
        )

    def forward(self, acoustic, linguistic, visual):
        # Process each modality
        acoustic_out = self.acoustic_path(acoustic)
        linguistic_out = self.linguistic_path(linguistic)
        visual_out = self.visual_path(visual)
        
        # Concatenate
        fused = torch.cat([acoustic_out, linguistic_out, visual_out], dim=1)
        
        # Fusion processing
        output = self.fusion_layer(fused)
        
        return output


class FusionDataLoader:
    """Custom dataloader for fusion model that combines three modalities."""
    
    def __init__(self, acoustic_data, linguistic_data, visual_data, labels, batch_size=32, shuffle=False):
        self.acoustic_data = acoustic_data
        self.linguistic_data = linguistic_data
        self.visual_data = visual_data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(labels))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            acoustic_batch = torch.FloatTensor(self.acoustic_data[batch_indices])
            linguistic_batch = torch.FloatTensor(self.linguistic_data[batch_indices])
            visual_batch = torch.FloatTensor(self.visual_data[batch_indices])
            labels_batch = torch.LongTensor(self.labels[batch_indices])
            
            yield acoustic_batch, linguistic_batch, visual_batch, labels_batch
    
    def __len__(self):
        return len(self.labels) // self.batch_size


def prepare_fusion_dataloaders(data_dir: str = "data"):
    """Prepare dataloaders for fusion model with 4 splits and StandardScaler normalization."""
    
    # Load training data for all modalities
    x_train_acoustic = np.load(f"{data_dir}/acoustic_train_x.npy")
    x_train_linguistic = np.load(f"{data_dir}/linguistic_train_x.npy")
    x_train_visual = np.load(f"{data_dir}/visual_train_x.npy")
    y_train = np.load(f"{data_dir}/acoustic_train_y.npy")
    
    # Fit StandardScaler on training data ONLY for each modality
    scaler_acoustic = StandardScaler()
    x_train_acoustic_scaled = scaler_acoustic.fit_transform(x_train_acoustic)
    
    scaler_linguistic = StandardScaler()
    x_train_linguistic_scaled = scaler_linguistic.fit_transform(x_train_linguistic)
    
    scaler_visual = StandardScaler()
    x_train_visual_scaled = scaler_visual.fit_transform(x_train_visual)
    
    # Load RLDD test set and apply scalers
    x_test_rldd_acoustic = np.load(f"{data_dir}/acoustic_test_rldd_x.npy")
    x_test_rldd_linguistic = np.load(f"{data_dir}/linguistic_test_rldd_x.npy")
    x_test_rldd_visual = np.load(f"{data_dir}/visual_test_rldd_x.npy")
    y_test_rldd = np.load(f"{data_dir}/acoustic_test_rldd_y.npy")
    
    x_test_rldd_acoustic_scaled = scaler_acoustic.transform(x_test_rldd_acoustic)
    x_test_rldd_linguistic_scaled = scaler_linguistic.transform(x_test_rldd_linguistic)
    x_test_rldd_visual_scaled = scaler_visual.transform(x_test_rldd_visual)
    
    # Load Dolos test set and apply scalers
    x_test_dolos_acoustic = np.load(f"{data_dir}/acoustic_test_dolos_x.npy")
    x_test_dolos_linguistic = np.load(f"{data_dir}/linguistic_test_dolos_x.npy")
    x_test_dolos_visual = np.load(f"{data_dir}/visual_test_dolos_x.npy")
    y_test_dolos = np.load(f"{data_dir}/acoustic_test_dolos_y.npy")
    
    x_test_dolos_acoustic_scaled = scaler_acoustic.transform(x_test_dolos_acoustic)
    x_test_dolos_linguistic_scaled = scaler_linguistic.transform(x_test_dolos_linguistic)
    x_test_dolos_visual_scaled = scaler_visual.transform(x_test_dolos_visual)
    
    # Load Dolos validation set and apply scalers
    x_val_dolos_acoustic = np.load(f"{data_dir}/acoustic_val_dolos_x.npy")
    x_val_dolos_linguistic = np.load(f"{data_dir}/linguistic_val_dolos_x.npy")
    x_val_dolos_visual = np.load(f"{data_dir}/visual_val_dolos_x.npy")
    y_val_dolos = np.load(f"{data_dir}/acoustic_val_dolos_y.npy")
    
    x_val_dolos_acoustic_scaled = scaler_acoustic.transform(x_val_dolos_acoustic)
    x_val_dolos_linguistic_scaled = scaler_linguistic.transform(x_val_dolos_linguistic)
    x_val_dolos_visual_scaled = scaler_visual.transform(x_val_dolos_visual)
    
    # Create custom dataloaders
    train_loader = FusionDataLoader(
        x_train_acoustic_scaled, x_train_linguistic_scaled, x_train_visual_scaled, y_train,
        batch_size=32, shuffle=True
    )
    
    test_rldd_loader = FusionDataLoader(
        x_test_rldd_acoustic_scaled, x_test_rldd_linguistic_scaled, x_test_rldd_visual_scaled, y_test_rldd,
        batch_size=32, shuffle=False
    )
    
    test_dolos_loader = FusionDataLoader(
        x_test_dolos_acoustic_scaled, x_test_dolos_linguistic_scaled, x_test_dolos_visual_scaled, y_test_dolos,
        batch_size=32, shuffle=False
    )
    
    val_dolos_loader = FusionDataLoader(
        x_val_dolos_acoustic_scaled, x_val_dolos_linguistic_scaled, x_val_dolos_visual_scaled, y_val_dolos,
        batch_size=32, shuffle=False
    )
    
    datasets = {
        'train': ('Combined Train', len(y_train)),
        'test_rldd': ('RLDD Test', len(y_test_rldd)),
        'test_dolos': ('Dolos Test', len(y_test_dolos)),
        'val_dolos': ('Dolos Validation', len(y_val_dolos))
    }
    
    loaders = {
        'train': train_loader,
        'test_rldd': test_rldd_loader,
        'test_dolos': test_dolos_loader,
        'val_dolos': val_dolos_loader
    }
    
    return loaders, datasets


def train_fusion_with_bias_testing():
    print("=" * 70)
    print("FUSION TRAINING WITH BIAS ANALYSIS")
    print("=" * 70)
    
    # Check all data files exist
    required_files = [
        "data/acoustic_train_x.npy",
        "data/linguistic_train_x.npy",
        "data/visual_train_x.npy"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found.")
            return

    # Prepare dataloaders
    print("\n[*] Preparing fusion dataloaders...")
    loaders, datasets = prepare_fusion_dataloaders()
    
    for split, info in datasets.items():
        print(f"  [OK] {info[0]}: {info[1]} samples")
    
    # Initialize model
    device = torch.device("cpu")
    model = FusionBrain().to(device)
    print(f"\n[MODEL] Fusion model initialized on {device}")
    print(f"   - Acoustic pathway: 692 → 256 → 128")
    print(f"   - Linguistic pathway: 775 → 256 → 128")
    print(f"   - Visual pathway: 512 → 256 → 128")
    print(f"   - Fusion layer: 384 → 256 → 128 → 2")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # Initialize metrics handler
    metrics_handler = TrainingMetrics("FusionBrain")
    
    # Training loop
    num_epochs = 1000
    print(f"\n[TRAIN] Starting training ({num_epochs} epochs)...\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for acoustic, linguistic, visual, labels in loaders['train']:
            acoustic = acoustic.to(device)
            linguistic = linguistic.to(device)
            visual = visual.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(acoustic, linguistic, visual)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase on RLDD test set
        model.eval()
        test_loss_rldd = 0.0
        test_correct_rldd = 0
        test_total_rldd = 0
        
        with torch.no_grad():
            for acoustic, linguistic, visual, labels in loaders['test_rldd']:
                acoustic = acoustic.to(device)
                linguistic = linguistic.to(device)
                visual = visual.to(device)
                labels = labels.to(device)
                
                outputs = model(acoustic, linguistic, visual)
                loss = criterion(outputs, labels)
                
                test_loss_rldd += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total_rldd += labels.size(0)
                test_correct_rldd += (predicted == labels).sum().item()
        
        test_loss_rldd /= test_total_rldd
        test_acc_rldd = test_correct_rldd / test_total_rldd
        
        # Validation phase on Dolos test set
        test_loss_dolos = 0.0
        test_correct_dolos = 0
        test_total_dolos = 0
        
        with torch.no_grad():
            for acoustic, linguistic, visual, labels in loaders['test_dolos']:
                acoustic = acoustic.to(device)
                linguistic = linguistic.to(device)
                visual = visual.to(device)
                labels = labels.to(device)
                
                outputs = model(acoustic, linguistic, visual)
                loss = criterion(outputs, labels)
                
                test_loss_dolos += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total_dolos += labels.size(0)
                test_correct_dolos += (predicted == labels).sum().item()
        
        test_loss_dolos /= test_total_dolos
        test_acc_dolos = test_correct_dolos / test_total_dolos
        
        # Record metrics (use RLDD for primary tracking)
        metrics_handler.record_epoch(train_loss, train_acc, test_loss_rldd, test_acc_rldd)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"RLDD Test Acc: {test_acc_rldd:.4f} | Dolos Test Acc: {test_acc_dolos:.4f}")
    
    print(f"\n[OK] Training completed!")
    
    # Save model
    model_path = "checkpoints/fusion_brain_final.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved: {model_path}")
    
    # Generate visualizations and reports
    print("\n[PLOT] Generating visualizations and reports...")
    
    # Plot training curves
    metrics_handler.plot_training_curves()
    
    # Evaluate on all datasets and perform bias analysis
    print("\n[BIAS] Performing bias analysis across datasets...")
    
    # Create evaluation loaders dict with all 3 test sets
    eval_loaders = {
        'RLDD_Test': loaders['test_rldd'],
        'Dolos_Test': loaders['test_dolos'],
        'Dolos_Validation': loaders['val_dolos']
    }
    
    metrics_handler, results_dict = evaluate_model_with_bias_analysis(
        model, eval_loaders, device, "FusionBrain"
    )
    
    # Plot bias comparison
    if len(results_dict) > 1:
        metrics_handler.plot_dataset_bias_comparison(results_dict)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nResults by dataset:")
    for dataset, metrics in results_dict.items():
        print(f"\n{dataset}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    print(f"\n[RESULTS] Results saved to: training_results/FusionBrain/")
    print("=" * 70)

if __name__ == "__main__":
    train_fusion_with_bias_testing()
