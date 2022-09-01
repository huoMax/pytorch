import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

# 自定义数据集，没有额外的标签文件，从数据文件的文件命中自定义标签
class CustomTrainDataset(Dataset):
    def __init__(self, img_root_path, data_transform=None, label_transform=None):
        self.img_root_path = img_root_path
        self.labels = list(os.listdir(img_root_path))
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = read_image(os.path.join(self.img_root_path, label))

        if self.label_transform:
            label = self.label_transform(label)
        if self.data_transform:
            img = self.data_transform(img)

        return img, label