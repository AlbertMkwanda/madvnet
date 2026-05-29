import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os


# 1. Architecture: Intelligent MLP for Statistical Acoustic Features
class AcousticBrain(nn.Module):
    def __init__(self, input_size=692):
        super(AcousticBrain, self).__init__()
        # We use a deeper network to process the Mean, Std, Max, and Min stats
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout to prevent overfitting on 692 features

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


def train_acoustic():
    # Ensure the data directory exists
    if not os.path.exists("data/acoustic_train_x.npy"):
        print("Error: .npy files not found. Please run the extraction script first.")
        return

    # Load the 692-dimensional features
    x_train = np.load("data/acoustic_train_x.npy")
    y_train = np.load("data/acoustic_train_y.npy")
    x_test = np.load("data/acoustic_test_x.npy")
    y_test = np.load("data/acoustic_test_y.npy")

    # Convert to PyTorch Tensors
    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Targeted at your i7-4600U setup
    device = torch.device("cpu")
    model = AcousticBrain(input_size=692).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Lower learning rate (0.0005) for smoother convergence with high-dim data
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)

    print(f"--- Training Intelligent Acoustic Brain (Input: {x_train.shape[1]} features) ---")

    best_acc = 0
    for epoch in range(10000):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation Phase
        model.eval()
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / len(y_test)
        print(f"Epoch {epoch + 1:02d} | Loss: {total_loss / len(train_loader):.4f} | Test Acc: {accuracy:.2f}%")

        # Save the best version of the model
        if accuracy > best_acc:
            best_acc = accuracy
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/acoustic_brain_final.pth")

    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")
    print("Model saved to models/acoustic_brain_final.pth")


if __name__ == "__main__":
    train_acoustic()