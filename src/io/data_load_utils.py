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
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BiasDataset(Dataset):
    def __init__(self, 
                 data_dir
                #  batch_size=32, 

    ):
        self.data = []
        self.labels = []
        self.load_data(data_dir)
        # self.collate_fn = self.collate_batch

    def load_data(self, data_dir,feature_num=6):
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                slice_bias_path = os.path.join(folder_path, 'slice_bias.json')
                veh2recons_bias_path = os.path.join(folder_path, 'veh2recons_bias.json')

                with open(slice_bias_path, 'r') as f:
                    slice_bias = json.load(f)
                with open(veh2recons_bias_path, 'r') as f:
                    veh2recons_bias = json.load(f)

                if len(slice_bias) < feature_num:
                    print(f"Skipping {folder} because slice_bias has fewer than 6 elements.")
                    continue

                # Get the first 20 values from veh2recons_bias
                veh2recons_values = list(veh2recons_bias.values())[:feature_num]
                self.data.append(veh2recons_values)

                # Calculate the mean of the first 20 values from slice_bias
                slice_values = list(slice_bias.values())[:feature_num]
                mean_slice_value = sum(slice_values) / len(slice_values)
                label = 1 if mean_slice_value < 1 else 0
                self.labels.append(label)
                
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
