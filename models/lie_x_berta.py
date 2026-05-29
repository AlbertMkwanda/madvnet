import torch
import torch.nn as nn
from transformers import AutoModel

class LieXBerta(nn.Module):
    def __init__(self, num_classes=3): # Changed to 3
        super(LieXBerta, self).__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes) # Outputs 3 categories

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.classifier(x)

    def get_features(self, input_ids, attention_mask):
        # Use self.roberta to match your __init__
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return output.pooler_output  # Returns the 768-dimensional vector