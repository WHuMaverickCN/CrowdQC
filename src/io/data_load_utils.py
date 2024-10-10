import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
# import spacy
# import torchtext
# from torchtext import data
# from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import math
import copy
from datetime import datetime
from torchvision import transforms
# from janome.tokenizer import Tokenizer

# 此处是专门用于读取训练数据，调用torch中DataLoader的位置。
class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, 
                    idx):
        pass