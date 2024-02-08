import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, x1, x2, y):
        """
        Shapes:
        -------
        x1: [B,C]
        x2: [B,C]
        y: [B,1]

        Returns:
        --------
        loss: [B,1]]
        """
        distance = torch.pairwise_distance(x1, x2, p=2)
        loss = self.alpha * (1 - y) * distance ** 2 + self.beta * y * (
                    torch.max(torch.zeros_like(distance), self.margin - distance) ** 2)
        return torch.mean(loss, dtype=torch.float)
