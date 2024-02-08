from torchvision import transforms
from PIL import ImageOps
import torch


def postprocess(tensor):
    return ImageOps.invert(transforms.ToPILImage()(tensor[0]))


def calculate_attack_acc(embed1, embed2, threshold, label):
    distance = torch.pairwise_distance(embed1, embed2, p=2)
    if label == 0:
        if distance < threshold:
            return 1
        else:
            return 0
    elif label == 1:
        if distance > threshold:
            return 1
        else:
            return 0
    else:
        raise ValueError('label is {}, not 0 or 1'.format(label))
