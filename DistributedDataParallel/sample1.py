import os
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    # 设计多进程通信环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'

    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# 销毁进程组
def cleanup():
    dist.destroy_process_group()

# 定义一个简单的模型
class TonyModel(nn.Module):
    def __init__(self):
        super(TonyModel, self).__init__()
        self.net1 = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print("Now running ddp demo in rank{}".format(rank))
    setup(rank, world_size)

    model = TonyModel()
    ddp_model = DDP(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    output = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5)
    criterion(output, labels).backward()
    optimizer.step()

    cleanup()

def compared_line_run():
    model = TonyModel()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    output = model(torch.randn(20, 10))
    labels = torch.randn(20, 5)
    criterion(output, labels).backward()
    optimizer.step()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    print(time.time())
    run_demo(demo_fn=demo_basic, world_size=4)
    print(time.time())
    print(time.time())
    for i in range(4):
        compared_line_run()
    print(time.time())
