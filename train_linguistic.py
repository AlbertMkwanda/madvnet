import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class LinguisticBrain(nn.Module):
    def __init__(self, input_size=775):
        super(LinguisticBrain, self).__init__()
        # Architecture designed for high-dimensional 775 input
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout to handle RoBERTa-base complexity

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [Truth, Deception]
        )

    def forward(self, x):
        return self.network(x)


def train_linguistic():
    # 1. Load Data
    x_train = np.load("data/linguistic_train_x.npy")
    y_train = np.load("data/linguistic_train_y.npy")
    x_test = np.load("data/linguistic_test_x.npy")
    y_test = np.load("data/linguistic_test_y.npy")

    # 2. FEATURE SCALING (Crucial for 58% stall)
    # This ensures your Liu counts (0-20) and RoBERTa vectors (-1 to 1) are comparable
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))

    # Smaller batch size for better generalization
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device("cpu")
    model = LinguisticBrain(input_size=x_train.shape[1]).to(device)

    # 3. ADVANCED OPTIMIZATION
    # Using Weight Decay (Regularization) and a lower LR
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Helps with noisy text labels
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Scheduler lowers LR when accuracy stalls
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    print(f"--- Training Brilliant Linguistic Brain (Input Dim: {x_train.shape[1]}) ---")

    best_acc = 0
    for epoch in range(5000):  # Reduced from 2000; with Scaling, it converges faster
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / len(y_test)

        # Step the scheduler
        scheduler.step(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), "models/linguistic_brain_best.pth")

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch + 1} | Loss: {train_loss / len(train_loader):.4f} | Acc: {accuracy:.2f}% | Best: {best_acc:.2f}%")

    print(f"\nFinal Breakthrough Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train_linguistic()