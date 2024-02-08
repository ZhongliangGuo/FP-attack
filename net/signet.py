import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            # input size = [155, 220, 1]
            nn.Conv2d(1, 96, 11),  # size = [145,210]
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [72, 105]
            nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'),  # size = [72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [36, 52]
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2),  # size = [18, 26]
            nn.Dropout2d(p=0.3),
            nn.Flatten(1, -1),  # 18*26*256
            nn.Linear(18 * 26 * 256, 1024),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
        )

    def forward(self, x):
        return self.features(x)


class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.encoder = Encoder()

    def forward(self, x1, x2):
        return self.encoder(x1), self.encoder(x2)


def glorot_init_uniform(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    pass
