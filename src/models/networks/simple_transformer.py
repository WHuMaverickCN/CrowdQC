"""
请基于pytorch从底层帮我实现一个transformer源代码，输入一个字符串，输出一个字符串
"""
'''
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
'''

# 4 手动实现transformer - scratch
import torch
from torch import nn
import torch.nn.functional as F
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import random, tqdm, sys, math, gzip

from src.models import former
from src.models.former import util
from src.models.former.util import d, here
from src.io.data_load_utils import BiasDataset

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
NUM_CLS = 2
# TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
# LABEL = data.Field(sequential=False)
def collate_batch(batch):
    text_list, label_list = [], []
        
    for _text, _label in batch:
        # 将特征 tensor 转换为 long 类型
        text_list.append(_text.long())  # 转换为整型
        label_list.append(_label.long())  # 标签也转换为整型

    # padding 处理，保证序列长度一致（根据你的需求，也可以不使用）
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.long)
        
    return text_list, label_list

def _get_train_dataset():
    # Define the path to your data directory
    data_dir = "/home/gyx/data/cqc/processed/fit1011/"
    # Create the dataset
    dataset = BiasDataset(data_dir)
    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,collate_fn=collate_batch)
    return train_loader,test_loader

def go(arg):
    '''
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    '''
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging
    '''
    # 这部分代码是用于测试RNN模型的
    ### 以下部分section1-6是基于pb代码，在pytorch 2.0.0之后的版本上基于DataLoader重构的

    # 1. 定义分词器
    tokenizer = get_tokenizer('basic_english')

    # 2. 定义函数来生成训练数据的token序列
    def yield_tokens(data_iter):
        for label, text in data_iter:
            yield tokenizer(text)

    # 3. 加载 IMDB 数据集并创建词汇表
    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # 4. 定义文本和标签处理函数
    def process_text(text):
        tokenized_text = tokenizer(text)
        indexed_text = [vocab[token] for token in tokenized_text]
        return torch.tensor(indexed_text, dtype=torch.long)

    def process_label(label):
        return 1 if label == 'pos' else 0

    # 5. 加载数据集并处理成 PyTorch 格式
    def collate_batch(batch):
        text_list, label_list = [], []
        for _label, _text in batch:
            label_list.append(process_label(_label))
            text_list.append(process_text(_text))
        
        # padding 处理
        text_list = pad_sequence(text_list, batch_first=True)
        label_list = torch.tensor(label_list, dtype=torch.float)
        return text_list, label_list

    # 使用 DataLoader 加载和批处理数据
    train_iter, test_iter = IMDB(split=('train', 'test'))
    batch_size = 8

    # 使用 DataLoader 代替 BucketIterator
    train_dataloader = DataLoader(list(train_iter), 
                            batch_size=batch_size, 
                            shuffle=True, 
                            collate_fn=collate_batch)
    test_dataloader = DataLoader(list(test_iter), 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=collate_batch) 

    # 6. 打印一个 batch 的内容
    for text_batch, label_batch in train_dataloader:
        print("Text batch shape:", text_batch.shape)
        print("Label batch shape:", label_batch.shape)
        break
    '''

    train_dataloader,test_dataloader = _get_train_dataset()
    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = former.CTransformer(emb=arg.embedding_size, \
                                heads=arg.num_heads, \
                                depth=arg.depth, \
                                seq_length=mx, \
                                num_tokens=arg.vocab_size,\
                                num_classes=NUM_CLS, \
                                max_pool=arg.max_pool)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)

    # training loop
    seen = 0
    for e in range(arg.num_epochs):

        print(f'\n epoch {e}')
        model.train(True)

        for batch in tqdm.tqdm(train_dataloader):

            opt.zero_grad()

            # input = batch.text[0]
            # label = batch.label - 1
            
            #重构
            input, label = batch[0].to(device), (batch[1]).to(device)

            if input.size(1) > mx:
                input = input[:, :mx]
            out = model(input)

            label = label.long()
            loss = F.nll_loss(out, label)

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            sch.step()

            seen += input.size(0)
            tbw.add_scalar('classification/train-loss', float(loss.item()), seen)

        with torch.no_grad():
            model.train(False)
            tot, cor= 0.0, 0.0

            for batch in tqdm.tqdm(test_dataloader):

                # input = batch.text[0]
                # label = batch.label - 1
                input, label = batch[0].to(device), (batch[1]).to(device)

                if input.size(1) > mx:
                    input = input[:, :mx]
                out = model(input).argmax(dim=1)

                tot += float(input.size(0))
                cor += float((label == out).sum().item())

            acc = cor / tot
            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            tbw.add_scalar('classification/test-loss', float(loss.item()), e)
    input()

if __name__ == "src.models.networks.simple_transformer":#作为模块运行

    parser = ArgumentParser()

    # parser.add_argument("-e", "--num-epochs",
    #                     dest="num_epochs",
    #                     help="Number of epochs.",
    #                     default=80, type=int)

    # parser.add_argument("-b", "--batch-size",
    #                     dest="batch_size",
    #                     help="The batch size.",
    #                     default=4, type=int)

    # parser.add_argument("-l", "--learn-rate",
    #                     dest="lr",
    #                     help="Learning rate",
    #                     default=0.0001, type=float)

    # parser.add_argument("-T", "--tb_dir", dest="tb_dir",
    #                     help="Tensorboard logging directory",
    #                     default='./runs')

    # parser.add_argument("-f", "--final", dest="final",
    #                     help="Whether to run on the real test set (if not included, the validation set is used).",
    #                     action="store_true")

    # parser.add_argument("--max-pool", dest="max_pool",
    #                     help="Use max pooling in the final classification layer.",
    #                     action="store_true")

    # parser.add_argument("-E", "--embedding", dest="embedding_size",
    #                     help="Size of the character embeddings.",
    #                     default=128, type=int)

    # parser.add_argument("-V", "--vocab-size", dest="vocab_size",
    #                     help="Number of words in the vocabulary.",
    #                     default=150_000, type=int)

    # parser.add_argument("-M", "--max", dest="max_length",
    #                     help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
    #                     default=512, type=int)

    # parser.add_argument("-H", "--heads", dest="num_heads",
    #                     help="Number of attention heads.",
    #                     default=8, type=int)

    # parser.add_argument("-d", "--depth", dest="depth",
    #                     help="Depth of the network (nr. of self-attention layers)",
    #                     default=6, type=int)

    # parser.add_argument("-r", "--random-seed",
    #                     dest="seed",
    #                     help="RNG seed. Negative for random",
    #                     default=1, type=int)

    # parser.add_argument("--lr-warmup",
    #                     dest="lr_warmup",
    #                     help="Learning rate warmup.",
    #                     default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args(args=[])

    print('OPTIONS ', options)

    go(options)


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
