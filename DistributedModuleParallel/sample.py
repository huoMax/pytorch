"""
本实例出自Pytorch官网教程 ’GETTING STARTED WITH DISTRIBUTED RPC FRAMEWORK‘ 文章
使用RNN模型来展示: 使用RPC API进行分布式模型并行训练
"""

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote

def _call_method(method, rref, *args, **kwargs):
    """a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    """a helper function to run method to the owner of rref and fetch back the result using RPC
    """
    return rpc.rpc_sync(rref.owner(), _call_method, args=[method, rref] + list(args), kwargs=kwargs)

def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs

class EmbeddingTable(nn.Module):
    """Encoding layers of the RNNModel
    """
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return self.drop(self.encoder(input))

class Decoder(nn.Module):
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))

# ps是参数服务器
class RNNModel(nn.Module):
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        return decoded, hidden

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        remote_params.extend(_parameter_rrefs(self.rnn))
        remote_params.extend(_remote_method, self.decoder_rref)
        return remote_params

def run_trainer():
    batch = 5
    ntoken = 10
    ninp = 2

    nhid = 3
    nindices = 3
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = RNNModel('ps', ntoken, ninp, nhid, nlayers)

    opt = Distri





