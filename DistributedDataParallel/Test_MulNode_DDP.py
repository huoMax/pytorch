import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# 初始化环境变量
os.environ['MASTER_ADDR'] = "localhost"
os.environ["MASTER_PORT"] = "23357"
# os.environ["GLOO_SOCKET_IFNAME"] = "WLAN"

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
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def run(rank, world_size):
    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 建立DDP模型
    model = Net()
    ddp_model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    # 数据采样
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, sampler=train_sampler)

    # 训练
    for epoch in range(10):
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.time()
        print("rank-%d, epoch-%d: [start_time: %f,end_time: %f\n" % (rank, epoch, start_time, end_time))

def main():
    world_size = 6
    mp.spawn(run,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__=="__main__":
    main()

