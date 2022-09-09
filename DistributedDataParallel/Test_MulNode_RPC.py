import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torch.multiprocessing as mp

# 初始化环境变量
os.environ['MASTER_ADDR'] = "localhost"
os.environ["MASTER_PORT"] = "23357"
os.environ["GLOO_SOCKET_IFNAME"] = "WLAN"
os.environ["TP_SOCKET_IFNAME"] = "WLAN"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 56)
        self.fc4 = nn.Linear(56, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x_rref):
        x = x_rref.to_here()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class Net2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x_rref):
        x = x_rref.to_here()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Net3(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(84, 56)
        self.fc2 = nn.Linear(56, 10)

    def forward(self, x_rref):
        x = x_rref.to_here()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x