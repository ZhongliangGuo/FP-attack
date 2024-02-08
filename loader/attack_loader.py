import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists

from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((155, 220)),
    ImageOps.invert,
    transforms.ToTensor(),
])


class GeneralDataset(Dataset):
    def __init__(self, label_path: str, data_dir: str, img_transform=None):
        if not exists(label_path):
            raise ValueError('train/test data pairs csv file not found in the data_dir.')
        else:
            print('Use existed data pairs csv file')
        self.img_transform = img_transform
        self.data_dir = data_dir
        self.df = pd.read_csv(join(label_path), header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x1, x2, label, correctly_pred = self.df.iloc[index]
        x1 = Image.open(join(self.data_dir, x1)).convert('L')
        x2 = Image.open(join(self.data_dir, x2)).convert('L')
        if self.img_transform:
            x1 = self.img_transform(x1)
            x2 = self.img_transform(x2)
        return x1, x2, label, correctly_pred


class AttackDataset(Dataset):
    def __init__(self, label_path: str, data_dir: str, img_transform=None, attacked_label=0):
        if not exists(label_path):
            raise ValueError('train/test data pairs csv file not found in the data_dir.')
        else:
            print('Use existed data pairs csv file')
        self.img_transform = img_transform
        self.data_dir = data_dir
        self.df = pd.read_csv(join(label_path), header=None)
        condition = (self.df.iloc[:, -2] == attacked_label) & (self.df.iloc[:, -1] == 1)
        self.df = self.df.loc[condition, [0, 1]].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x1, x2 = self.df.iloc[index]
        x1 = Image.open(join(self.data_dir, x1)).convert('L')
        x2 = Image.open(join(self.data_dir, x2)).convert('L')
        if self.img_transform:
            x1 = self.img_transform(x1)
            x2 = self.img_transform(x2)
        return x1, x2


def get_attack_loader(label_path: str, batch_size=1, img_transform=image_transform,
                      data_dir='path-to-data',
                      shuffle=False, attacked_label=0):
    data = AttackDataset(label_path, data_dir, img_transform, attacked_label)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=12, pin_memory=True)
    return loader


def get_data_loader(label_path: str, batch_size=1, img_transform=image_transform,
                    data_dir='path-to-data',
                    shuffle=False):
    data = GeneralDataset(label_path, data_dir, img_transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=12, pin_memory=True)
    return loader


class OriginalDataset(Dataset):
    def __init__(self, label_path: str, data_dir: str, img_transform=None):
        if not exists(label_path):
            raise ValueError('train/test data pairs csv file not found in the data_dir.')
        else:
            print('Use existed data pairs csv file')
        self.img_transform = img_transform
        self.data_dir = data_dir
        self.df = pd.read_csv(join(label_path), header=None)
        print(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x1, x2, label = self.df.iloc[index]
        x1 = Image.open(join(self.data_dir, x1)).convert('L')
        x2 = Image.open(join(self.data_dir, x2)).convert('L')
        if self.img_transform:
            x1 = self.img_transform(x1)
            x2 = self.img_transform(x2)
        return x1, x2, label

def get_ori_loader(label_path: str, batch_size=1, img_transform=image_transform,
                    data_dir='path-to-data',
                    shuffle=False):
    data = OriginalDataset(label_path, data_dir, img_transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=12, pin_memory=True)
    return loader


if __name__ == "__main__":
    pass
