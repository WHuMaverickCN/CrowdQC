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
# from janome.tokenizer import Tokenizer

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')