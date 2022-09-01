import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
from torchvision.io import read_image

# 从CSV文件中获取标签信息
class TrainDataset(Dataset):
    def __init__(self, labels_file, img_root_path, data_transforms=None, label_transforms=None):
        self.labels = pd.read_csv(labels_file)
        self.img_root_path = img_root_path
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root_path, self.labels.iloc[idx, 0])
        label = self.labels.iloc[idx, 1]
        img = read_image(img_path)

        if self.data_transforms:
            img = self.data_transforms(label)
        if self.label_transforms:
            label = self.label_transforms(label)

        return img, label
