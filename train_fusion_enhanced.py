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
    """Prepare dataloaders for fusion model."""
    
    # Load all modalities
    x_train_acoustic = np.load(f"{data_dir}/acoustic_train_x.npy")
    x_test_acoustic = np.load(f"{data_dir}/acoustic_test_x.npy")
    
    x_train_linguistic = np.load(f"{data_dir}/linguistic_train_x.npy")
    x_test_linguistic = np.load(f"{data_dir}/linguistic_test_x.npy")
    
    x_train_visual = np.load(f"{data_dir}/visual_train_x.npy")
    x_test_visual = np.load(f"{data_dir}/visual_test_x.npy")
    
    y_train = np.load(f"{data_dir}/acoustic_train_y.npy")
    y_test = np.load(f"{data_dir}/acoustic_test_y.npy")
    
    # Scale linguistic features
    scaler = StandardScaler()
    x_train_linguistic_scaled = scaler.fit_transform(x_train_linguistic)
    x_test_linguistic_scaled = scaler.transform(x_test_linguistic)
    
    # Create custom dataloaders
    train_loader = FusionDataLoader(
        x_train_acoustic, x_train_linguistic_scaled, x_train_visual, y_train,
        batch_size=32, shuffle=True
    )
    
    test_loader = FusionDataLoader(
        x_test_acoustic, x_test_linguistic_scaled, x_test_visual, y_test,
        batch_size=32, shuffle=False
    )
    
    datasets = {
        'train': ('RLDD Train', len(y_train)),
        'test': ('RLDD Test', len(y_test))
    }
    
    return train_loader, test_loader, datasets


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
    print("\n📂 Preparing fusion dataloaders...")
    train_loader, test_loader, datasets = prepare_fusion_dataloaders()
    
    for split, info in datasets.items():
        print(f"  ✓ {info[0]}: {info[1]} samples")
    
    # Initialize model
    device = torch.device("cpu")
    model = FusionBrain().to(device)
    print(f"\n🧠 Fusion model initialized on {device}")
    print(f"   - Acoustic pathway: 692 → 256 → 128")
    print(f"   - Linguistic pathway: 775 → 256 → 128")
    print(f"   - Visual pathway: 512 → 256 → 128")
    print(f"   - Fusion layer: 384 → 256 → 128 → 2")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # Initialize metrics handler
    metrics_handler = TrainingMetrics("FusionBrain")
    
    # Training loop
    num_epochs = 100
    print(f"\n🚀 Starting training ({num_epochs} epochs)...\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for acoustic, linguistic, visual, labels in train_loader:
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
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for acoustic, linguistic, visual, labels in test_loader:
                acoustic = acoustic.to(device)
                linguistic = linguistic.to(device)
                visual = visual.to(device)
                labels = labels.to(device)
                
                outputs = model(acoustic, linguistic, visual)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_loss /= test_total
        test_acc = test_correct / test_total
        
        # Record metrics
        metrics_handler.record_epoch(train_loss, train_acc, test_loss, test_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    print(f"\n✓ Training completed!")
    
    # Save model
    model_path = "checkpoints/fusion_brain_final.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Generate visualizations and reports
    print("\n📊 Generating visualizations and reports...")
    
    # Plot training curves
    metrics_handler.plot_training_curves()
    
    # Evaluate final performance
    print("\n🔍 Evaluating final performance...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for acoustic, linguistic, visual, labels in test_loader:
            acoustic = acoustic.to(device)
            linguistic = linguistic.to(device)
            visual = visual.to(device)
            
            outputs = model(acoustic, linguistic, visual)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Generate confusion matrix and report
    cm, cm_path = metrics_handler.plot_confusion_matrix(all_labels, all_preds, "RLDD_Test")
    report = metrics_handler.generate_classification_report(all_labels, all_preds, "RLDD_Test")
    
    accuracy = (all_labels == all_preds).mean()
    precision = report['Deception']['precision']
    recall = report['Deception']['recall']
    f1 = report['Deception']['f1-score']
    
    metrics_handler.save_detailed_report(all_labels, all_preds, "RLDD_Test")
    
    # Print summary
    print("\n" + "=" * 70)
    print("FUSION TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nTest Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n📁 Results saved to: training_results/FusionBrain/")
    print("=" * 70)

if __name__ == "__main__":
    train_fusion_with_bias_testing()
