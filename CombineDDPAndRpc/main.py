import random

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim

from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16

class HybridModel(torch.nn.Module):
    """
    The model consists of a sparse part and a dense part
    1. The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel
    2. The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """
    def __init__(self, remote_emb_module):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16,8))

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices,offsets)
        return self.fc(emb_lookup)

def _run_trainer(remote_emb_module, rank):
    """
    每个 trainer 前向传播时，在参数服务器上进行嵌入查找并运行nn.Linear
    在反向传递期间，DDP负责聚合密集部分(nn.Linear)的梯度，分布式autograd确保梯度更新传播到参数服务器。
    """
    model = HybridModel(remote_emb_module)

    # 从 embedding table 接受参数
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # model.fc.parameters() 仅包含 local parameters
    # 注意：不可以在此处调用 model.parameters()
    # 因为会导致调用到 remote_emb_module.parameters()，
    # 而 remote_emb_module 并不支持 parameters()
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # generate offsets
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8)
            yield indices, offsets_tensor, target

    # Train for 100 epochs
    for epoch in range(100):
        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Tun distributed optimizer
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training done for epoch {}".format(epoch))




def run_worker(rank, world_size):
    """
    A wrapper function that initializes RPC, calls the function, ans shuts down RPC
    """
    # 使用不同的端口初始化RPC(init_rpc)和DDP(init_process_group)，避免端口冲突
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 是主节点(master)，Rank 3 是参数服务器(ps)，Rank 0, 1 是 trainers
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # run the training loop on trainers
        futs = []
        for trainers_rank in [0,1]:
            trainers_name = "trainer{}".format(trainers_rank)
            fut = rpc.rpc_async(
                trainers_name, _run_trainer, args=(remote_emb_module, trainers_rank)
            )
            futs.append(fut)
        for fut in futs:
            fut.wait()

    elif rank <= 1:
        # 初始化DDP进程组(init_process_group)
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # 初始化RPC
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

    # Trainer just waits for RPCs from master
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        pass

     # block until all rpcs finish
    rpc.shutdown()

if __name__ == "__main__":
    # 2 trainers(FC layers), 1 parameter server, 1 master
    world_size = 4
    mp.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
