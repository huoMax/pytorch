"""
https://zhuanlan.zhihu.com/p/358974461

首先，看一下如何实现多机的一个进程占用一张卡的使用，需要注意的位置：

1. dist.init_process_group里面的rank需要根据node以及GPU的数量计算；

2. world_size的大小=节点数 x GPU 数量。

3. ddp 里面的device_ids需要指定对应显卡。

这里因为做实验相关，树莓派没有GPU，因此将其改为CPU，即多台节点，每个节点一个进程，运行以下命令：
#节点1
python demo.py --world_size=2 --rank=0 --master_addr="192.168.0.1" --master_port=22335

#节点2
python demo.py --world_size=2 --rank=1 --master_addr="192.168.0.1" --master_port=22335
"""

import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
import argparse
import os
os.environ["MASTER_ADDR"] = "192.168.159.140"
os.environ["MASTER_PORT"] = "22336"
os.environ["GLOO_SOCKET_IFNAME"] = "ens33"

# 解析命令参数
# parser = argparse.ArgumentParser()
# parser.add_argument("--rank", default=0, type=int)
# parser.add_argument("--world_size", default=1, type=int)
# parser.add_argument("--master_addr", default="127.0.0.1", type=str)
# parser.add_argument("--master_port", default="12355", type=str)
# args = parser.parse_args()


def run(rank, world_size):
    # 初始化进程组, 设置两个节点，各1个进程
    print("test2")
    dist.init_process_group("gloo",
                            init_method='tcp://192.168.159.140:22336',
                            rank=rank,
                            world_size=world_size)

    # 定义 transforms
    print("test1")
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # 选择 Pytorch 自带的 MNIST 分类数据集
    # 加载训练集
    data_set_train = torchvision.datasets.MNIST("../data", train=True, transform=trans, target_transform=None, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set_train) # 数据集分类器, 这里相当于数据并行中不同进程对应 data_set_test 的子集
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set_train,
                                                    batch_size=256,
                                                    sampler=train_sampler,
                                                    num_workers=2,
                                                    pin_memory=True)
    print("训练集大小: {}".format(len(data_set_train)))

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
        torch.save(net, "./ModelSave/MinHost_SinProcess.pth")


if __name__ == "__main__":
    rank = 0
    world_size = 2
    run(rank, world_size)