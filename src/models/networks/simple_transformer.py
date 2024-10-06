"""
请基于pytorch从底层帮我实现一个transformer源代码，输入一个字符串，输出一个字符串
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.utils import *
# import torch.nn.MultiheadAttention as MHA
from torch.nn import MultiheadAttention as MHA


# 1 调用pytorch的MHA
x = torch.rand((2, 2000000, 512))
mattn = MHA(embed_dim=512, num_heads=8)
@print_run_time("mha")
def mTF():
    output = mattn(x, x, x)
    return output
output = mTF()

# 2 手动实现多头注意力
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, 
            embed_dim, 
            num_heads,
            batch_first
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )
        def forward(self, x, x_mask):
            out, _ = self.mha(x,x,x, key_padding_mask=x_mask)
            return out

def positional_encoding(length, embed_dim):
    dim = embed_dim//2

    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)

    angle = 1 / (10000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)

    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed

# 3 手动实现transformer - Harvard
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline

# 4 手动实现transformer - OpenAI
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
      
        super().__init__()
        
        assert k % heads == 0
        
        self.k, self.heads = k, heads
        # Note the assert: the embedding dimension needs to be divisible by the number of heads. 
        # 嵌入维度需要能被头部的数量整除

        # These compute the queries, keys and values for all
        # heads
        self.tokeys    = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues  = nn.Linear(k, k, bias=False)

        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x)
        keys    = self.tokeys(x)   
        values  = self.tovalues(x)

        # 重塑张量以添加一个在head迭代的维度
        s = k // h
        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

         # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # -- dot has size (b*h, t, t) containing raw weights

        # scale the dot product
        dot = dot / (s ** (1/2))
        
        # normalize 
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights

        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
    
        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
        nn.Linear(k, 4 * k),
        nn.ReLU(),
        nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)