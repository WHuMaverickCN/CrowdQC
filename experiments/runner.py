import os

import torch
from src.utils import mkdir_if_missing
from src.io.data_load_utils import ReconstructionDataset

class Runner:
    def __init__(self, args):
        self.args = args

        
        # if args.proc_id == 0:
        # Check GPU availability
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用GPU 2
        os.environ['WORLD_SIZE'] = str(args.world_size)
        if not args.no_cuda and not torch.cuda.is_available():
            raise Exception("No gpu available for usage")
        if int(os.getenv('WORLD_SIZE', 1)) >= 1:
            print("Let's use", int(os.getenv('WORLD_SIZE', 1)), "GPUs!")
            torch.cuda.empty_cache()

        # 创建推理结果路径

        mkdir_if_missing(args.save_path)
        from src.models.networks import simple_transformer
        # from src.models.networks import feature_extractor
    
    def train(self):
        pass
    
    def test(self):
        pass

    def _get_train_dataset(self):
        args = self.args
        train_dataset = ReconstructionDataset(args.dataset_dir, \
                                            args.data_dir + 'training/',\
                                            args, \
                                            data_aug=True, \
                                            save_std=True, \
                                            seg_bev=args.seg_bev)
        
        # train_loader, train_sampler = get_loader(train_dataset, args)

        # return train_dataset, train_loader, train_sampler