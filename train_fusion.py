import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class WeightedFusionBrain(nn.Module):
    def __init__(self, audio_dim=692, visual_dim=512, text_dim=775):
        super(WeightedFusionBrain, self).__init__()
        # Learnable parameters for the Softmax Gate
        # Initialized to favor Audio (71%) and Visual (69%) over Text (59%)
        self.modality_logits = nn.Parameter(torch.tensor([1.5, 1.2, 0.4]))

        self.network = nn.Sequential(
            nn.Linear(audio_dim + visual_dim + text_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 2)
        )

    def forward(self, a, v, t):
        # 1. Calculate Softmax Weights (they will always sum to 1.0)
        # This is the "Automatic Optimizer"
        soft_weights = F.softmax(self.modality_logits, dim=0)

        # 2. Apply the weights
        weighted_a = a * soft_weights[0]
        weighted_v = v * soft_weights[1]
        weighted_t = t * soft_weights[2]

        # 3. Concatenate and pass through the classifier
        combined = torch.cat((weighted_a, weighted_v, weighted_t), dim=1)
        return self.network(combined)


def train_weighted_fusion():
    # Load separate modalities
    ax_train = np.load("data/acoustic_train_x.npy")
    vx_train = np.load("data/visual_train_x.npy")
    lx_train = np.load("data/linguistic_train_x.npy")
    y_train = np.load("data/acoustic_train_y.npy")

    ax_test = np.load("data/acoustic_test_x.npy")
    vx_test = np.load("data/visual_test_x.npy")
    lx_test = np.load("data/linguistic_test_x.npy")
    y_test = np.load("data/acoustic_test_y.npy")

    # Scaling
    scaler_a = StandardScaler();
    ax_train = scaler_a.fit_transform(ax_train);
    ax_test = scaler_a.transform(ax_test)
    scaler_v = StandardScaler();
    vx_train = scaler_v.fit_transform(vx_train);
    vx_test = scaler_v.transform(vx_test)
    scaler_l = StandardScaler();
    lx_train = scaler_l.fit_transform(lx_train);
    lx_test = scaler_l.transform(lx_test)

    train_ds = TensorDataset(torch.FloatTensor(ax_train), torch.FloatTensor(vx_train),
                             torch.FloatTensor(lx_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    test_ds = TensorDataset(torch.FloatTensor(ax_test), torch.FloatTensor(vx_test),
                            torch.FloatTensor(lx_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=16)

    model = WeightedFusionBrain()
    # Using a slightly higher weight decay to prevent the weights from becoming too extreme
    optimizer = optim.AdamW(model.parameters(), lr=0.00003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("--- Training Auto-Optimized Fusion Brain ---")
    best_acc = 0
    for epoch in range(1000):  # 1000 is usually enough with 775 features
        model.train()
        for a_b, v_b, t_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(a_b, v_b, t_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for a_b, v_b, t_b, y_b in test_loader:
                outputs = model(a_b, v_b, t_b)
                _, pred = torch.max(outputs, 1)
                correct += (pred == y_b).sum().item()

        acc = 100 * correct / len(y_test)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/fusion_brain_best.pth")

        if epoch % 5 == 0:
            # Calculate current softmax distribution for printing
            with torch.no_grad():
                w = F.softmax(model.modality_logits, dim=0)
            print(f"Epoch {epoch} | Acc: {acc:.2f}% | Trust Distribution -> A:{w[0]:.2f} V:{w[1]:.2f} T:{w[2]:.2f}")

    print(f"\nBreakthrough Fusion Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train_weighted_fusion()