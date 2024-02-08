import torch
from torch import nn


class DiffLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x1, x2):
        return self.weight * torch.sum(torch.abs(x1 - x2))
