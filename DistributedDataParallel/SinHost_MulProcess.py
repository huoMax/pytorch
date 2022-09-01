"""

"""

import os
import time
import torch
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
from torch.multiprocessing import Process

# 设置环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


def main(rank):
    # 初始化进程组, 设置4个进程
    dist.init_process_group("gloo", rank=rank, world_size=2)

    # 定义 transforms
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # 选择 Pytorch 自带的 MNIST 分类数据集
    # 加载训练集
    data_set_train = torchvision.datasets.MNIST("../data", train=True, transform=trans, target_transform=None, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set_train) # 数据集分类器, 这里相当于数据并行中不同进程对应 data_set_test 的子集
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set_train, batch_size=256, sampler=train_sampler)
    print("训练集大小: {}".format(len(data_set_train)))

    # 加载测试集
    data_set_test = torchvision.datasets.MNIST("../data", train=False, transform=trans, target_transform=None, download=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(data_set_test)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_set_test, batch_size=256, sampler=test_sampler)
    print("测试集大小: {}".format(len(data_set_test)))

    # 使用 ResNet 模型
    # 定义模型
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

    # 将模型进行分布式预处理
    net = torch.nn.parallel.DistributedDataParallel(net)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    # 训练
    print("start train Model...     time: {}".format(time.time()))
    for epoch in range(1):
        for i, data in enumerate(data_loader_train):
            images, labels = data

            # 清除梯度积累
            opt.zero_grad()

            # forward
            outputs = net(images)

            # backward
            loss = criterion(outputs, labels)
            loss.backward()

            # optimizer
            opt.step()

            if i % 10 == 0:
                print("loss: {}     time: {}    loop: {}    rank: {}".format(loss.item(), time.time(), i, rank))

    if rank == 0:
        torch.save(net, "./ModelSave/SinHost_MulProcess.pth")


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=main, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()