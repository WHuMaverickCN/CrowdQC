import argparse
def define_args_______():
    parser = argparse.ArgumentParser(description='Crowd_Quality_Inference')
    #路径设置 Path Settings
    parser.add_argument('--config',\
            default='',\
            type=str, help='config file path')
    
    parser.add_argument('--data_dir', \
                        type=str, \
                        default= '/home/gyx/data/cqc/processed/fit0925/',\
                        help='The path of dataset json files (annotations & raw reconstructed geojson files)')
    
    parser.add_argument('--save_path', \
                        type=str, \
                        default= './inference/',\
                        help='directory to save output')
    
    # General model settings 
    #  - Transformer Classify model settings
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--world_size', type=int, default = 1)
    parser.add_argument("-e", "--num-epochs",dest="num_epochs",help="Number of epochs.",default=80, type=int)
    parser.add_argument("-b", "--batch-size",dest="batch_size",help="The batch size.",default=4, type=int)
    parser.add_argument("-l", "--learn-rate",dest="lr",help="Learning rate",default=0.0001, type=float)
    parser.add_argument("-T", "--tb_dir", dest="tb_dir",help="Tensorboard logging directory",default='./runs')
    parser.add_argument("-f", "--final", dest="final",help="Whether to run on the real test set (if not included, the validation set is used).",action="store_true")
    parser.add_argument("--max-pool", dest="max_pool",help="Use max pooling in the final classification layer.",action="store_true")
    parser.add_argument("-E", "--embedding", dest="embedding_size",help="Size of the character embeddings.",default=128, type=int)
    parser.add_argument("-V", "--vocab-size", dest="vocab_size",help="Number of words in the vocabulary.",default=150_000, type=int)
    parser.add_argument("-M", "--max", dest="max_length",help="Max sequence length. Longer sequences are clipped (-1 for no limit).",default=512, type=int)
    parser.add_argument("-H", "--heads", dest="num_heads",help="Number of attention heads.",default=8, type=int)
    parser.add_argument("-d", "--depth", dest="depth",help="Depth of the network (nr. of self-attention layers)",default=6, type=int)
    parser.add_argument("-r", "--random-seed",dest="seed",help="RNG seed. Negative for random",default=1, type=int)
    parser.add_argument("--lr-warmup",dest="lr_warmup",help="Learning rate warmup.",default=10_000, type=int)
    parser.add_argument("--gradient-clipping",dest="gradient_clipping",help="Gradient clipping.",default=1.0, type=float)
    

    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local-rank", type=int, default=1, help="Local rank passed from torchrun")
    
#     parser.add_argument("--local_rank", type=int)
#     parser.add_argument("--local_rank", type=int, default = 0)
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--nodes', type=int, default = 1)
    # parser.add_argument('--dataset_dir', type=str, help='The path of dataset  files (images)')
    return parser

def define_args():
    parser = argparse.ArgumentParser(description='Crowd_Quality_Inference')
    parser.add_argument('--local-rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--nodes', type=int, default = 1)
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    

    #default model settings
    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=12, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=150_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--distributed', action='store_true')
    return parser