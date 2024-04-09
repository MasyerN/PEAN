from torch.nn import functional as F
import torch
from torch import nn


class MaxPoolingMIL(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.embedding = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(.2)
        self.ln = nn.LayerNorm(1024)
        self.classifier = nn.Linear(1024, cfgs["num_classes"])
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.ln(x)
        y = self.classifier(x)
        # y = F.sigmoid(y)
        return y
    
    def inference(self, x):
        x = x.reshape([-1, 1024])
        y = torch.softmax(self.forward(x), 1)
        pred = y.argmax(dim=-1)
        classes, counts = torch.unique(pred, return_counts=True)
        pred = classes[torch.argmax(counts)]
        return pred