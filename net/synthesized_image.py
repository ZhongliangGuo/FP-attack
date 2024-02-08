import torch
from torch import nn


class SynthesizedImage(nn.Module):
    def __init__(self, init_img_tensor, img_shape=(1, 1, 155, 220), **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))
        if init_img_tensor is not None:
            self.weight.data.copy_(init_img_tensor.data)

    def forward(self):
        return self.weight
