import numpy as np
import os
import os.path as ops

def config(args):
    args.data_dir = 'Inference'
    args.dataset_name = '/home/gyx/data/cqc/processed/fit0925/'
    args.save_path = './inference/'
    args.world_size = 4 #有限使用四张卡训练