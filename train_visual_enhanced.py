# Enhanced Visual Training with Bias Testing and Visualization
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from training_utils import TrainingMetrics, evaluate_model_with_bias_analysis

# Original VisualBrain model with residual connection (unchanged)
class VisualBrain(nn.Module):
    def __init__(self, input_size=512):
        super(VisualBrain, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        self.fc_out = nn.Linear(512, 2)  # Final Output: [Truth, Deception]

    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second layer with residual connection
        residual = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x + residual  # Residual connection
        
        # Dropout
        x = self.dropout(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Output layer
        x = self.fc_out(x)
        
        return x


def load_dataset_split(split_file: str, label_col: str = 'label'):
    """Load a dataset split CSV file."""
    if not os.path.exists(split_file):
        print(f"Warning: {split_file} not found")
        return None
    
    df = pd.read_csv(split_file)
    
    # Normalize labels
    labels = df[label_col].apply(lambda x: 0 if 'truth' in str(x).lower() else 1).values
    
    return df, labels


def prepare_dataloaders(visual_data_dir: str = "data"):
    """Prepare train, test, and validation dataloaders."""
    
    # Load original train/test splits
    x_train = np.load(f"{visual_data_dir}/visual_train_x.npy")
    y_train = np.load(f"{visual_data_dir}/visual_train_y.npy")
    x_test = np.load(f"{visual_data_dir}/visual_test_x.npy")
    y_test = np.load(f"{visual_data_dir}/visual_test_y.npy")

    # Load Dolos splits if available
    dolos_train_csv = f"{visual_data_dir}/dolos_train_split.csv"
    dolos_test_csv = f"{visual_data_dir}/dolos_test_split.csv"
    dolos_val_csv = f"{visual_data_dir}/dolos_validation_split.csv"
    
    loaders = {}
    datasets = {}
    
    # Create train loader
    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    loaders['train'] = DataLoader(train_ds, batch_size=32, shuffle=True)
    datasets['train'] = ('RLDD Train', len(y_train))
    
    # Create test loader
    test_ds = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
    loaders['test_rldd'] = DataLoader(test_ds, batch_size=32)
    datasets['test_rldd'] = ('RLDD Test', len(y_test))
    
    # Load and create Dolos splits if available
    if os.path.exists(dolos_test_csv):
        dolos_test_df, dolos_test_labels = load_dataset_split(dolos_test_csv)
        print("  ℹ️  Dolos test split found (features need extraction)")
    
    if os.path.exists(dolos_val_csv):
        dolos_val_df, dolos_val_labels = load_dataset_split(dolos_val_csv)
        print("  ℹ️  Dolos validation split found (features need extraction)")
    
    return loaders, datasets


def train_visual_with_bias_testing():
    print("=" * 70)
    print("VISUAL TRAINING WITH BIAS ANALYSIS")
    print("=" * 70)
    
    # Ensure the data directory exists
    if not os.path.exists("data/visual_train_x.npy"):
        print("Error: .npy files not found. Please run the extraction script first.")
        return

    # Prepare data loaders
    print("\n📂 Preparing data loaders...")
    loaders, datasets = prepare_dataloaders()
    
    for split, info in datasets.items():
        print(f"  ✓ {info[0]}: {info[1]} samples")
    
    # Initialize model
    device = torch.device("cpu")
    model = VisualBrain(input_size=512).to(device)
    print(f"\n🧠 Model initialized on {device}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Initialize metrics handler
    metrics_handler = TrainingMetrics("VisualBrain")
    
    # Training loop
    num_epochs = 100
    print(f"\n🚀 Starting training ({num_epochs} epochs)...\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in loaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(loaders['train'])
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in loaders['test_rldd']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_loss /= len(loaders['test_rldd'])
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
    model_path = "checkpoints/visual_brain_final.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Generate visualizations and reports
    print("\n📊 Generating visualizations and reports...")
    
    # Plot training curves
    metrics_handler.plot_training_curves()
    
    # Evaluate on all datasets and perform bias analysis
    print("\n🔍 Performing bias analysis across datasets...")
    
    # Create evaluation loaders dict
    eval_loaders = {
        'RLDD_Test': loaders['test_rldd']
    }
    
    metrics_handler, results_dict = evaluate_model_with_bias_analysis(
        model, eval_loaders, device, "VisualBrain"
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
    
    print(f"\n📁 Results saved to: training_results/VisualBrain/")
    print("=" * 70)

if __name__ == "__main__":
    train_visual_with_bias_testing()
