import os
import threading
import time
import argparse

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torchvision.models.resnet import ResNet, Bottleneck
num_classes = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int)
parser.add_argument("--rank", type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)
parser.add_argument("--backend", default='ens33', type=str)
parser.add_argument("--tcp_backend", default='ens33', type=str)
args = parser.parse_args()

def SetEnvironmentalVariable():
    """
    Set distributed rpc framework and distributed data parallel environmental variable
    """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['GLOO_SOCKET_IFNAME'] = args.backend
    os.environ['TP_SOCKET_IFNAME']  = args.tcp_backend

def model1(ResNet):


def model2(ResNet):
    pass


def run_master():
    pass

def run_worker(rank, world_size):
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        run_master()
    else:
        rpc.init_rpc(
            "work{}".format(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        pass




if __name__ == "__main__":
    SetEnvironmentalVariable()
    start_time = time.time()
    run_worker(args.rank, args.world_size)
    end_time = time.time()
