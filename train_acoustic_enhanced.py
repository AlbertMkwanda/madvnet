# Enhanced Acoustic Training with Bias Testing and Visualization
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from training_utils import TrainingMetrics, evaluate_model_with_bias_analysis
import matplotlib.pyplot as plt

# Original AcousticBrain model (unchanged)
class AcousticBrain(nn.Module):
    def __init__(self, input_size=692):
        super(AcousticBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 2)  # Final Output: [Truth, Deception]
        )

    def forward(self, x):
        return self.network(x)


def load_dataset_split(split_file: str, label_col: str = 'label'):
    """Load a dataset split CSV file."""
    if not os.path.exists(split_file):
        print(f"Warning: {split_file} not found")
        return None
    
    df = pd.read_csv(split_file)
    
    # Normalize labels
    labels = df[label_col].apply(lambda x: 0 if 'truth' in str(x).lower() else 1).values
    
    return df, labels


def prepare_dataloaders(acoustic_data_dir: str = "data"):
    """Prepare train, test, and validation dataloaders."""
    
    # Load original train/test splits (for backward compatibility)
    x_train = np.load(f"{acoustic_data_dir}/acoustic_train_x.npy")
    y_train = np.load(f"{acoustic_data_dir}/acoustic_train_y.npy")
    x_test_rldd = np.load(f"{acoustic_data_dir}/acoustic_test_x.npy")
    y_test_rldd = np.load(f"{acoustic_data_dir}/acoustic_test_y.npy")

    # Load Dolos splits if available
    dolos_train_csv = f"{acoustic_data_dir}/dolos_train_split.csv"
    dolos_test_csv = f"{acoustic_data_dir}/dolos_test_split.csv"
    dolos_val_csv = f"{acoustic_data_dir}/dolos_validation_split.csv"
    
    loaders = {}
    datasets = {}
    
    # Create train loader
    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    loaders['train'] = DataLoader(train_ds, batch_size=32, shuffle=True)
    datasets['train'] = ('Combined Train', len(y_train))
    
    # Create RLDD test loader
    test_ds_rldd = TensorDataset(torch.FloatTensor(x_test_rldd), torch.LongTensor(y_test_rldd))
    loaders['test_rldd'] = DataLoader(test_ds_rldd, batch_size=32)
    datasets['test_rldd'] = ('RLDD Test', len(y_test_rldd))
    
    print("  ℹ️  Note: Dolos test/validation splits require feature extraction")
    print("       Continuing with RLDD test set for now...")
    
    return loaders, datasets


def train_acoustic_with_bias_testing():
    print("=" * 70)
    print("ACOUSTIC TRAINING WITH BIAS ANALYSIS")
    print("=" * 70)
    
    # Ensure the data directory exists
    if not os.path.exists("data/acoustic_train_x.npy"):
        print("Error: .npy files not found. Please run the extraction script first.")
        return

    # Prepare data loaders
    print("\n📂 Preparing data loaders...")
    loaders, datasets = prepare_dataloaders()
    
    for split, info in datasets.items():
        print(f"  ✓ {info[0]}: {info[1]} samples")
    
    # Initialize model
    device = torch.device("cpu")
    model = AcousticBrain(input_size=692).to(device)
    print(f"\n🧠 Model initialized on {device}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Initialize metrics handler
    metrics_handler = TrainingMetrics("AcousticBrain")
    
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
    model_path = "checkpoints/acoustic_brain_final.pth"
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
        model, eval_loaders, device, "AcousticBrain"
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
    
    print(f"\n📁 Results saved to: training_results/AcousticBrain/")
    print("=" * 70)

if __name__ == "__main__":
    train_acoustic_with_bias_testing()
