# Enhanced Linguistic Training with Bias Testing and Visualization
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from training_utils import TrainingMetrics, evaluate_model_with_bias_analysis

# Original LinguisticBrain model (unchanged)
class LinguisticBrain(nn.Module):
    def __init__(self, input_size=775):
        super(LinguisticBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
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


def prepare_dataloaders(linguistic_data_dir: str = "data"):
    """Prepare train, RLDD test, Dolos test, and validation dataloaders with separate scaling."""
    
    # Load training data (combined RLDD + Dolos)
    x_train = np.load(f"{linguistic_data_dir}/linguistic_train_x.npy")
    y_train = np.load(f"{linguistic_data_dir}/linguistic_train_y.npy")

    # IMPORTANT: Fit StandardScaler ONLY on training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # Load RLDD test set
    x_test_rldd = np.load(f"{linguistic_data_dir}/linguistic_test_rldd_x.npy")
    y_test_rldd = np.load(f"{linguistic_data_dir}/linguistic_test_rldd_y.npy")
    x_test_rldd_scaled = scaler.transform(x_test_rldd)
    
    # Load Dolos test set
    x_test_dolos = np.load(f"{linguistic_data_dir}/linguistic_test_dolos_x.npy")
    y_test_dolos = np.load(f"{linguistic_data_dir}/linguistic_test_dolos_y.npy")
    x_test_dolos_scaled = scaler.transform(x_test_dolos)
    
    # Load Dolos validation set
    x_val_dolos = np.load(f"{linguistic_data_dir}/linguistic_val_dolos_x.npy")
    y_val_dolos = np.load(f"{linguistic_data_dir}/linguistic_val_dolos_y.npy")
    x_val_dolos_scaled = scaler.transform(x_val_dolos)
    
    loaders = {}
    datasets = {}
    
    # Create train loader (shuffle for training)
    train_ds = TensorDataset(torch.FloatTensor(x_train_scaled), torch.LongTensor(y_train))
    loaders['train'] = DataLoader(train_ds, batch_size=16, shuffle=True)
    datasets['train'] = ('RLDD+Dolos Train', len(y_train))
    
    # Create RLDD test loader
    test_rldd_ds = TensorDataset(torch.FloatTensor(x_test_rldd_scaled), torch.LongTensor(y_test_rldd))
    loaders['test_rldd'] = DataLoader(test_rldd_ds, batch_size=16, shuffle=False)
    datasets['test_rldd'] = ('RLDD Test', len(y_test_rldd))
    
    # Create Dolos test loader
    test_dolos_ds = TensorDataset(torch.FloatTensor(x_test_dolos_scaled), torch.LongTensor(y_test_dolos))
    loaders['test_dolos'] = DataLoader(test_dolos_ds, batch_size=16, shuffle=False)
    datasets['test_dolos'] = ('Dolos Test', len(y_test_dolos))
    
    # Create Dolos validation loader
    val_dolos_ds = TensorDataset(torch.FloatTensor(x_val_dolos_scaled), torch.LongTensor(y_val_dolos))
    loaders['val_dolos'] = DataLoader(val_dolos_ds, batch_size=16, shuffle=False)
    datasets['val_dolos'] = ('Dolos Validation', len(y_val_dolos))
    
    return loaders, datasets


def train_linguistic_with_bias_testing():
    print("=" * 70)
    print("LINGUISTIC TRAINING WITH BIAS ANALYSIS")
    print("=" * 70)
    
    # Ensure the data directory exists
    if not os.path.exists("data/linguistic_train_x.npy"):
        print("Error: .npy files not found. Please run the extraction script first.")
        return

    # Prepare data loaders
    print("\n[*] Preparing data loaders...")
    loaders, datasets = prepare_dataloaders()
    
    for split, info in datasets.items():
        print(f"  [OK] {info[0]}: {info[1]} samples")
    
    # Initialize model
    device = torch.device("cpu")
    model = LinguisticBrain(input_size=775).to(device)
    print(f"\n[MODEL] Model initialized on {device}")

    # Loss and Optimizer with weight decay and label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # Initialize metrics handler
    metrics_handler = TrainingMetrics("LinguisticBrain")
    
    # Training loop
    num_epochs = 1500
    print(f"\n[TRAIN] Starting training ({num_epochs} epochs)...\n")
    
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
        
        # Validation phase - Test on RLDD
        model.eval()
        test_loss_rldd = 0.0
        test_correct_rldd = 0
        test_total_rldd = 0
        
        with torch.no_grad():
            for inputs, labels in loaders['test_rldd']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss_rldd += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total_rldd += labels.size(0)
                test_correct_rldd += (predicted == labels).sum().item()
        
        test_loss_rldd /= len(loaders['test_rldd'])
        test_acc_rldd = test_correct_rldd / test_total_rldd
        
        # Test on Dolos
        test_loss_dolos = 0.0
        test_correct_dolos = 0
        test_total_dolos = 0
        
        with torch.no_grad():
            for inputs, labels in loaders['test_dolos']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss_dolos += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total_dolos += labels.size(0)
                test_correct_dolos += (predicted == labels).sum().item()
        
        test_loss_dolos /= len(loaders['test_dolos'])
        test_acc_dolos = test_correct_dolos / test_total_dolos
        
        # Record metrics (use RLDD as primary for training curve)
        metrics_handler.record_epoch(train_loss, train_acc, test_loss_rldd, test_acc_rldd)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"RLDD Test Acc: {test_acc_rldd:.4f} | "
                  f"Dolos Test Acc: {test_acc_dolos:.4f}")
    
    print(f"\n[OK] Training completed!")
    
    # Save model
    model_path = "checkpoints/linguistic_brain_final.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved: {model_path}")
    
    # Generate visualizations and reports
    print("\n[*] Generating visualizations and reports...")
    
    # Plot training curves
    metrics_handler.plot_training_curves()
    
    # Evaluate on all datasets and perform bias analysis
    print("\n[BIAS] Performing bias analysis across datasets...")
    
    # Create evaluation loaders dict with all test sets
    eval_loaders = {
        'RLDD_Test': loaders['test_rldd'],
        'Dolos_Test': loaders['test_dolos'],
        'Dolos_Validation': loaders['val_dolos']
    }
    
    metrics_handler, results_dict = evaluate_model_with_bias_analysis(
        model, eval_loaders, device, "LinguisticBrain"
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
    
    print(f"\n[RESULTS] Results saved to: training_results/LinguisticBrain/")
    print("=" * 70)

if __name__ == "__main__":
    train_linguistic_with_bias_testing()
