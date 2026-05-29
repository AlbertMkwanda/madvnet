import torch
import torch.nn as nn


class DeceptionFusion(nn.Module):
    def __init__(self, v_dim=512, t_dim=768, num_classes=3):
        """
        v_dim: Output dimension of MADVNet feature layer (usually 512 for ResNet-based 3D CNNs)
        t_dim: Output dimension of LieXBerta (768 for standard BERT-base)
        """
        super(DeceptionFusion, self).__init__()

        # Concatenated dimension
        self.input_dim = v_dim + t_dim

        # The "Decision Making" layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Helps prevent overfitting during retraining
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, v_features, t_features):
        # v_features: [batch, 512]
        # t_features: [batch, 768]

        # Merge the vectors
        combined = torch.cat((v_features, t_features), dim=1)

        # Final classification
        logits = self.fusion_layer(combined)
        return logits