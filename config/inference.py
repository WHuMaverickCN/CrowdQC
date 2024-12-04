import numpy as np
import os
import os.path as ops

def config(args):
    args.model_name = 'ClassFormer'
    
    args.data_dir = 'Inference'
    # args.dataset_name = '/home/gyx/data/cqc/processed/fit1011/'
    args.dataset_name = '/home/gyx/data/cqc/processed/clips_class_mode/'
    args.save_path = './inference/'
    args.world_size = 4 # 优先使用四张卡训练

    # args.no_cuda = False
    # for the case only running evaluation
    args.evaluate = False
    args.evaluate_case = False
    args.evaluate = True
    args.evaluate_case = True
    # 是否使用Tensorboard
    args.no_tb = True
    
    # ddp init
    args.use_slurm = False
    # args.local_rank = 0

    # ddp related
    args.dist = True
    args.sync_bn = True

    args.cudnn = True
    # args.port = 29666

    