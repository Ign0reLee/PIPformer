import os,sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import collections.abc as container_abcs

# from apex.parallel import DistributedDataParallel as DDP 
from torch.nn.parallel import DistributedDataParallel as DDP

from libs.models.GEL2 import GEL2
from libs.trainer.train_utils import save_gen, load_gen, init_process, find_free_port
from libs.trainer.base_trainer import Base_Trainer

from tqdm import tqdm

r"""
Sample Model
This Model using just Generator with L2 Loss
Model's Name: PPformers
"""

class PatchPainting_GEL2(Base_Trainer):
    
    def __init__(self, jsonData, sharedFilePath,restart=False, startEpoch=0):
        
        self.jsonData = jsonData
        self.restart = restart
        self.startEpoch = startEpoch
        self.sharedFilePath = sharedFilePath
    
    def train(self, gpu, size):

        print(f"Now Initialize GPU : {gpu} | Number Of GPU : {size}")
        self.initialize(gpu, size)
        self.gan = GEL2(self.jsonData, gpu) 

        # Initialize Trainer's 
        Base_Trainer.__init__(self, self.gan.pathDB, self.gan.ckptDir, self.gan.logDir, self.gan.patch_size, self.gan.img_size, self.gan.N, self.gan.batchSize, self.sharedFilePath)  
        self.name = self.gan.name + "_GEL2"
        if not os.path.exists(os.path.join(self.ckpt_dir, self.name)): os.mkdir(os.path.join(self.ckpt_dir, self.name))
        
        if self.restart:
            self.gan.model, self.gan.optimizers, self.startEpoch = load_gen(self.ckpt_dir, self.gan.model, self.gan.optimizers)

        self.gan.model = DDP(self.gan.model, device_ids=[gpu])
        self.makeDatasets()
        self.makeTensorBoard()
        print(f"DDP Rank {gpu} Run...")


        for epoch in range(self.startEpoch, self.gan.epochs):
            self.datasetTrainSampler.set_epoch(epoch)
            self.gan.model.train()

            for data in self.loaderTrain:
                inputImg = data["inputs"].cuda(gpu, non_blocking=True)
                realImg  = data["real"].cuda(gpu, non_blocking=True)
                if self.gan.genLocal:
                    dropIndex = data["indexMap"].cuda(gpu, non_blocking=True)
                else:
                    dropIndex = None
                self.gan.oneTrainStep(inputImg, realImg, dropIndex)

            # Print and Save Log
            if epoch % 100 == 0 and gpu == 0:
                nowLoss = np.mean(self.gan.lossLog)
                nowAcc  = np.mean(self.gan.accLog)

                stringLog = f"Train {epoch} / {self.gan.epochs} Epochs | PSNR Acc {nowAcc} | LOSS {nowLoss}"
                self.summaryWriter.add_scalar("Train Loss", nowLoss, epoch)
                self.summaryWriter.add_scalar("Train PSNR Acc", nowAcc, epoch)

                if self.gan.genLocal:
                    stringLog += f" | Global MSE {np.mean(self.gan.globalLoss)} | Local MSE {np.mean(self.gan.localLoss)}"
                    self.summaryWriter.add_scalar("Train Global MSE", np.mean(self.gan.globalLoss), epoch)
                    self.summaryWriter.add_scalar("Train Local MSE", np.mean(self.gan.localLoss), epoch)
                    self.gan.globalLoss= []
                    self.gan.localLoss = []

                self.gan.lossLog = []
                self.gan.accLog  = []
                print(stringLog)
            
            if epoch % self.gan.saveIter == 0:
                self.checkPoint(gpu, self.gan.ckptDir, self.gan.model, self.gan.optimizers, epoch, self.name)
                
                # Evaluation Every saveIter and gpu 0
                if gpu  == 0:
                    self.gan.model.eval()
                    with torch.no_grad():
                        for index, data in enumerate(self.loaderVal):
                            inputImg = data["inputs"].cuda(gpu, non_blocking=True)
                            realImg  = data["real"].cuda(gpu, non_blocking=True)

                            if self.gan.genLocal:
                                dropIndex = data["indexMap"].cuda(gpu, non_blocking=True)
                            else:
                                dropIndex = None
                            outImg, attn = self.gan.oneValStep(inputImg, realImg, dropIndex, False)
                            
                            # Save Every First Value Image
                            if index == 0:
                                self.summaryWriter.add_image("inputs",  self.gan.makeVisual(inputImg), epoch, dataformats="NHWC")
                                self.summaryWriter.add_image("outputs", self.gan.makeVisual(outImg), epoch, dataformats="NHWC")
                                self.summaryWriter.add_image("reals",   self.gan.makeVisual(realImg), epoch, dataformats="NHWC")
                                for attn_name in attn.keys():
                                    self.gan.makeAttnVisual(self.summaryWriter, attn[attn_name].to("cpu").detach().numpy()[0], epoch, attn_name)

                        # Print and Save Log
                        nowLoss = np.mean(self.gan.lossVal)
                        nowAcc  = np.mean(self.gan.accVal)
                        stringLog = f"Validation {epoch} / {self.gan.epochs} Epochs | PSNR Acc {nowAcc} | LOSS {nowLoss}"
                        self.summaryWriter.add_scalar("Validation Loss", nowLoss, epoch)
                        self.summaryWriter.add_scalar("Validation PSNR Acc", nowAcc, epoch)

                        if self.gan.genLocal:
                            stringLog+= f" | Global MSE {np.mean(self.gan.globalVal)} | Local MSE {np.mean(self.gan.localVal)}"
                            self.summaryWriter.add_scalar("Validation Global MSE", np.mean(self.gan.globalVal), epoch)
                            self.summaryWriter.add_scalar("Validation Local MSE", np.mean(self.gan.localVal), epoch)
                            self.gan.globalVal = []
                            self.gan.localVal = []
                        
                        self.gan.lossVal = []
                        self.gan.accVal  = []
                        print(stringLog)

    def eval(self, vis=False, saveImage=False, mix=True, epoch=None, maskPath=None):
        r"""
        For Evaluation Step, when using this doesn't using Multi GPU
        """

        if saveImage:
            # Save Image setting
            savePath = os.path.join(".","outputs")
            Index    = 0

            # Save Image Path Check
            if not os.path.exists(savePath):
                os.mkdir(savePath)
                os.mkdir(os.path.join(savePath, "inputs"))
                os.mkdir(os.path.join(savePath, "outputs"))
                os.mkdir(os.path.join(savePath, "real"))
            
        if vis is True:
            # Visualization Image Setting
            self.log_dir += "/Evaluation"
            print(self.log_dir)

            # Save Visualization Path Check
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            self.makeTensorBoard()


        gpu = 0
        self.gan = GEL2(self.jsonData, gpu) 

        # Initialize Trainer's 
        Base_Trainer.__init__(self, self.gan.pathDB, self.gan.ckptDir, self.gan.logDir, self.gan.patch_size, self.gan.img_size, self.gan.N, self.gan.batchSize, self.sharedFilePath)  
        self.name = self.gan.name + "_GEL2"
        self.makeEvalDatasets(maskPath)
        
        
        self.gan.model, self.gan.optimizers, self.startEpoch = load_gen(self.ckpt_dir, self.gan.model, self.gan.optimizers, self.name, epoch=epoch)
        print(f"Load Done.. Model Name : {self.name} | {self.startEpoch}")
        print(f"Path :{self.gan.pathDB}")

        self.gan.model.eval()
        with torch.no_grad():
            for index, data in tqdm(enumerate(self.loaderTrain), desc="Run Evaluation...", total=len(self.loaderTrain)):
                inputImg = data["inputs"].cuda(gpu)
                realImg  = data["real"].cuda(gpu)
                if self.gan.genLocal:
                    dropIndex = data["indexMap"].cuda(gpu)
                else:
                    dropIndex = None
                outImg, _ = self.gan.oneValStep(inputImg, realImg, dropIndex, mix)

                if vis:
                    self.summaryWriter.add_image("inputs",  self.gan.makeVisual(inputImg), index, dataformats="NHWC")
                    self.summaryWriter.add_image("outputs", self.gan.makeVisual(outImg), index, dataformats="NHWC")
                    self.summaryWriter.add_image("reals",   self.gan.makeVisual(realImg), index, dataformats="NHWC")
                
                if saveImage:
                    for iImg, oImg, rImg in zip(self.gan.makeVisual(inputImg),  self.gan.makeVisual(outImg), self.gan.makeVisual(realImg)):
                        plt.imsave(os.path.join(savePath, "inputs", f"{Index}.png"), iImg)
                        plt.imsave(os.path.join(savePath, "outputs", f"{Index}.png"), oImg)
                        plt.imsave(os.path.join(savePath, "real", f"{Index}.png"), rImg)
                        Index += 1
                    

            # Print and Save Log
            nowLoss = np.mean(self.gan.lossVal)
            nowAcc  = np.mean(self.gan.accVal)
            stringLog = f"Evaluation End : {self.startEpoch} Epochs | PSNR Acc {nowAcc} | LOSS {nowLoss}"
            if self.gan.genLocal:
                stringLog+= f" | Global MSE {np.mean(self.gan.globalVal)} | Local MSE {np.mean(self.gan.localVal)}"
            
            print(stringLog)
    
    def visual_init(self, epoch=None):
        r"""
        For Inference Mode, runing when using visual.py 
        This step just Initialize and Load Model
        """
        gpu = 0
        self.gan = GEL2(self.jsonData, gpu) 

        # Initialize Trainer's 
        Base_Trainer.__init__(self, self.gan.pathDB, self.gan.ckptDir, self.gan.logDir, self.gan.patch_size, self.gan.img_size, self.gan.N, self.gan.batchSize, self.sharedFilePath)  
        self.name = self.gan.name + "_GEL2"
        self.makeEvalDatasets()

        self.gan.model, self.gan.optimizers, self.startEpoch = load_gen(self.ckpt_dir, self.gan.model, self.gan.optimizers, self.name, epoch=epoch)
        self.gan.model.eval()

             

    def checkPoint(self, gpu, ckptDir, model, optim, epoch, name):

        if gpu == 0:
            save_gen(ckptDir, model, optim, epoch, name)
        
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(os.path.join(ckptDir, name, f"{name}_{epoch}.pth"), map_location=mapLocation)
        model.module.load_state_dict(dict_model["netG"])
    
    def runTrain(self):

        gpus = torch.cuda.device_count()
        mp.spawn(self.train, args=(gpus,), nprocs=gpus)
    
    def runEval(self, vis=False, si=False,mix=True, epcoh=None, maskPath=None):
        self.eval(vis, si, mix, epoch=epcoh, maskPath=maskPath)


if __name__ == "__main__":

    size = torch.cuda.device_count()
    testTrainer = PatchPainting_GEL2("TrainParameterExample.json")
    testTrainer.runTrain()
    # testTrainer = mp.spawn(, args=(size,"TrainParameterExample.json"), nprocs=size)
    # testTrainer.runTrain()



            
        

        

        