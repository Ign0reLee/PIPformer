import json, os
import numpy as np

import torch
import torch.nn as nn


from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from libs.datalibs.datasets import Dataset
from libs.datalibs.compose import  NPatchDrop, Normalization, MaskDrop
from libs.trainer.train_utils import find_free_port


class Base_Trainer():

    def __init__(self, pathDB, ckpt_dir, log_dir, patch_size, img_size, N, batchSize, sharedFilePath):
        # Set Variable's
        self.pathDB = pathDB
        self.world_size = 2
        self.nr         = 0
        self.patch_size = patch_size
        self.log_dir    = log_dir
        self.img_size   = img_size
        self.N          = N
        self.batchSize  = batchSize
        self.ckpt_dir   = ckpt_dir
        self.numWorkers = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()//2
        self.gpu_count = torch.cuda.device_count()
        self.sharedFilePath = sharedFilePath

        # Run Initialize Function
        self.__checkDirectory__()
        
    
    def __checkDirectory__(self):
        assert os.path.exists(self.pathDB), f"Please Input Check Your Dataset Path!"

        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
    
    def initialize(self, gpu, size):
        # For Debugging If, You want. using below it
        # os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ['NCCL_DEBUG_SUBSYS']="ALL"
        # For Online Learning, In My enviroment, can't using it. if want to use chang, init_method.
        # os.environ["MASTER_ADDR"] = "127.0.0.1"
        # os.environ["MASTER_PORT"] = "29500"

        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['NCCL_IB_DISABLE']= '1'
        os.environ['LOCAL_RANK'] = str(gpu)
        torch.cuda.set_device(gpu)

        # Setting Model with Json Parsing
        dist.init_process_group(backend='nccl', init_method='file://'+self.sharedFilePath+'/sharedfile', rank=gpu, world_size=size)

    
    def makeDatasets(self, ddp=True):
        self.transformTrain     = transforms.Compose([Normalization(mean=0.5, std=0.5),NPatchDrop(self.patch_size)])
        self.dataset            = Dataset(data_dir=self.pathDB, transforms=self.transformTrain,
                                        patchSize=self.patch_size, size=(self.img_size, self.img_size), N=self.N)

        self.datasetTrain, self.datasetVal = torch.utils.data.random_split(self.dataset, [int(len(self.dataset) * 0.8), int(len(self.dataset) * 0.2)])

        if ddp:
            self.datasetTrainSampler= DistributedSampler(self.datasetTrain)
            self.datasetValSampler  = DistributedSampler(self.datasetVal)

            self.loaderTrain        = DataLoader(self.datasetTrain, batch_size=self.batchSize, shuffle=False, num_workers=self.numWorkers, 
                                                pin_memory=True, sampler=self.datasetTrainSampler)
            self.loaderVal          = DataLoader(self.datasetVal, batch_size=self.batchSize, shuffle=False, num_workers=self.numWorkers, 
                                                pin_memory=True, sampler=self.datasetValSampler)
        else:
            self.loaderTrain        = DataLoader(self.datasetTrain, batch_size=self.batchSize, shuffle=True, num_workers=self.numWorkers)
            self.loaderVal          = DataLoader(self.datasetVal, batch_size=self.batchSize, shuffle=True, num_workers=self.numWorkers)

    def makeEvalDatasets(self, maskPath=None):
        if maskPath is not None:
            self.transformTrain = transforms.Compose([Normalization(mean=0.5, std=0.5), MaskDrop()])
        else:
            self.transformTrain     = transforms.Compose([Normalization(mean=0.5, std=0.5),NPatchDrop(self.patch_size)])
        self.dataset            = Dataset(data_dir=self.pathDB, transforms=self.transformTrain,
                                        patchSize=self.patch_size, size=(self.img_size, self.img_size), N=self.N, maskPath = maskPath)
        self.loaderTrain        = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=False, num_workers=self.numWorkers, pin_memory=True)
    
    def makeTensorBoard(self):

        self.summaryWriter = SummaryWriter(log_dir=self.log_dir)
    
    
    
    

            
            
            

        
