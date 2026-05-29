import torch.nn as nn

class AcousticBrain(nn.Module):
    def __init__(self, input_dim=40, num_classes=3):
        super(AcousticBrain, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        return self.classifier(x)

    def get_features(self, x):
        # This returns the 64-dimensional feature vector for Fusion
        x = self.relu(self.layer1(x))
        return self.relu(self.layer2(x))