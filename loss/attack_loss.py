import torch
from torch import nn


class AttackLoss(nn.Module):
    def __init__(self, threshold=0.03, weight=1e-1):
        super().__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, x1, x2):
        distance = torch.pairwise_distance(x1, x2, p=2)
        loss = torch.where(distance > self.threshold, distance / self.threshold, 0.0)
        return self.weight * torch.mean(loss, dtype=torch.float)
