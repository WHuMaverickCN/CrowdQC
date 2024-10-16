import os
import math
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from src.utils import mkdir_if_missing
from src.io.data_load_utils import BiasDataset

from src.models import former
from src.models.former import util
from src.models.former.util import d, here
from src.io.data_load_utils import BiasDataset


# Used for converting between nats and bits
LOG2E = math.log2(math.e)
NUM_CLS = 2

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

        # Get Dataset
        if args.proc_id == 0:#单卡处理的情况
            print("Loading Dataset ...")
            self.train_dataloader,\
            self.test_dataloader = self._get_train_dataset()
        else:#多卡处理的情况
            pass
        # 创建推理结果路径
        # train_dataloader,test_dataloader = self._get_train_dataset()

        mkdir_if_missing(args.save_path)
        tbw = SummaryWriter(log_dir=args.tb_dir) 

        if args.model_name == 'ClassFormer':
            self.criterion = torch.nn.NLLLoss()
        
        if args.proc_id == 0:#单卡处理的情况
            print("Init Done")


        # from src.models.networks import simple_transformer
        # from src.models.networks import feature_extractor
    
    def train(self):
        args = self.args
        train_dataloader = self.train_dataloader
        
        # Define the model
        if args.model_name == 'ClassFormer':
            model, \
            optimizer, \
            scheduler, \
            best_epoch, \
            lowest_loss, \
            best_f1_epoch, \
            best_val_f1 = self._get_model_ddp()
        
        criterion = self.criterion
        # criterion = self.criterion
        if not args.no_cuda and torch.cuda.is_available():
            device = torch.device("cuda", args.local_rank)
            criterion = criterion.to(device)
            model.to(device)

        for e in range(args.num_epochs):
            for batch in tqdm.tqdm(train_dataloader):
                optimizer.zero_grad()
                input, \
                label = batch[0].to(device), (batch[1]).to(device)
                if input.size(1) > self.mx:
                    input = input[:, :self.mx]
                out = model(input)

                label = label.long()
                loss = criterion(out, label)
                
                loss.backward()
                




    def validate(self):
        args = self.args
        pass

    def eval(self):
        args = self.args
        pass
    
    def _get_train_dataset(self):
        # Define the path to your data directory
        # data_dir = self.data
        data_dir = "/home/gyx/data/cqc/processed/fit1011/"
        # Create the dataset
        dataset = BiasDataset(data_dir)
        # Split the dataset into training and testing sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=self.collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,collate_fn=self.collate_batch)
        return train_loader,test_loader

    def _get_test_dataset(self):
        data_dir = "/home/gyx/data/cqc/processed/fit1011/"


        pass

    @staticmethod
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
    
    def _get_model_ddp(self):
        args = self.args
        train_dataloader = self.train_dataloader
        if args.max_length < 0:
            mx = max([input.text[0].size(1) for input in train_dataloader])
            mx = mx * 2
            self.mx = mx
            print(f'- maximum sequence length: {mx}')
        else:
            mx = args.max_length
            self.mx = mx
        model = former.CTransformer(emb=args.embedding_size, \
                                        heads=args.num_heads, \
                                        depth=args.depth, \
                                        seq_length=mx, \
                                        num_tokens=args.vocab_size,\
                                        num_classes=NUM_CLS, \
                                        max_pool=args.max_pool)
            
        # 批量的标准化
        if args.sync_bn and args.proc_id == 0:
            print("Converting Sync BatchNorm")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.no_cuda:
            device = torch.device('cuda',args.local_rank)
            if args.model_name == 'ClassFormer':
                model = model.to(device)
        
        """
            first load param to model, then model = DDP(model)
        """
        # Logging setup
        best_epoch = 0
        lowest_loss = np.inf
        best_f1_epoch = 0
        best_val_f1 = -1e-5
        optim_saved_state = None
        schedule_saved_state = None

        # # resume model
        # args.resume = first_run(args.save_path)
        # if args.resume:
        #     model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, \
        #         optim_saved_state, schedule_saved_state = self.resume_model(args, model)
        # elif args.pretrained and args.proc_id == 0:
        #     if args.model_name == "PersFormer":
        #         path = 'models/pretrain/model_pretrain.pth.tar'
        #         if os.path.isfile(path):
        #             checkpoint = torch.load(path)
        #             model.load_state_dict(checkpoint['state_dict'])
        #             print("Use pretrained model in {} to start training".format(path))
        #         else:
        #             raise Exception("No pretrained model found in {}".format(path))
        # elif args.pretrained and args.model_name == "GenLaneNet":
        #     checkpoint = torch.load(args.pretrained_feat_model)
        #     model1 = self.load_my_state_dict(model1, checkpoint['state_dict'])
        #     model1.eval()  # do not back propagate to model1

        # dist.barrier()

        if args.distributed:
            if args.model_name == "ClassFormer":
                model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        
        '''
            Define optimizer after DDP init
        '''
        if args.model_name == "ClassFormer":
            optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (args.lr_warmup / args.batch_size), 1.0))

        if args.model_name == "ClassFormer":
            return model,   \
                    optimizer, \
                    scheduler, \
                    best_epoch, \
                    lowest_loss, \
                    best_f1_epoch, \
                    best_val_f1