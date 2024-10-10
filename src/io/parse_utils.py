import argparse
def define_args():
    parser = argparse.ArgumentParser(description='Crowd_Quality_Inference')
    #路径设置
    
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
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--world_size', type=int, default = 1)
    
    
    # parser.add_argument('--dataset_dir', type=str, help='The path of dataset  files (images)')
    return parser