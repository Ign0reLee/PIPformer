import os,sys, gc
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
 
from libs.models.PPransGAN import PPransGAN
from libs.trainer.train_utils import save, load
from libs.trainer.base_trainer import Base_Trainer

from tqdm import tqdm
from ptflops import get_model_complexity_info

import GPUtil





r"""
Optimize Patch-Based Painting using GAN architecture with Transformers
Model's Name: Patch-based Painting Transformers Generative Adversarial Networks
Base Loss : WGAN-GP
Reference Code : 
    https://github.com/facebookresearch/pytorch_GAN_zoo (WGAN-GP)
    https://github.com/godisboy/SN-GAN (SN-PatchGAN)
"""

class PPransGANtrainer(Base_Trainer):
    
    def __init__(self, jsonData, sharedFilePath,restart=False, startEpoch=0, ddp=False):

        torch.cuda.max_memory_allocated()
        self.jsonData = jsonData
        self.restart = restart
        self.startEpoch = startEpoch
        self.sharedFilePath = sharedFilePath
    
    def train(self):

        self.gan = PPransGAN(self.jsonData, gpu= "cuda:0" if torch.cuda.is_available() else "cpu") 

        # Initialize Trainer's 
        Base_Trainer.__init__(self, self.gan.pathDB, self.gan.ckptDir, self.gan.logDir, self.gan.patch_size, self.gan.img_size, self.gan.N, self.gan.batchSize, self.sharedFilePath)  
        self.name = self.gan.name + "_PPransGAN"
        if not os.path.exists(os.path.join(self.ckpt_dir, self.name)): os.mkdir(os.path.join(self.ckpt_dir, self.name))
        dummy_size = (3, 256, 256)
        netGmacs, netGparams = get_model_complexity_info(self.gan.netG, dummy_size, as_strings=False, print_per_layer_stat=True, verbose=True)
        netDmacs, netDparams = get_model_complexity_info(self.gan.netD, dummy_size, as_strings=False, print_per_layer_stat=True, verbose=True)
        print(f'computational netG complexity: {float(netGmacs)/2} FLOPs')
        print(f'number of netG parameters: {netGparams} prams')
        print(f'computational netD complexity: {float(netDmacs)/2} FLOPs')
        print(f'number of netD parameters: {netDparams} prams')
        del dummy_size

        
        if self.restart:
            self.gan.netG, self.gan.netD, self.gan.optimG, self.gan.optimD, self.startEpoch = load(os.path.join(self.gan.ckptDir, self.name), self.gan.netG, self.gan.netD, self.gan.optimG, self.gan.optimD)
            self.startEpoch += 1
            print(f"Load Done.. Step : {self.startEpoch}")
            
        self.gan.netG = nn.DataParallel(self.gan.netG)
        self.gan.netD = nn.DataParallel(self.gan.netD)
        # self.gan.perceptualNet = nn.DataParallel(self.gan.perceptualNet)

        self.makeDatasets(ddp=False)
        self.makeTensorBoard()


        for epoch in range(self.startEpoch, self.gan.epochs):

            self.gan.netG.train()
            self.gan.netD.train()
            torch.cuda.empty_cache()
            gc.collect()

            for data in self.loaderTrain:
                # GPUtil.showUtilization()
                torch.cuda.empty_cache()
                gc.collect()

                inputImg = data["inputs"].cuda()
                realImg  = data["real"].cuda()
                if self.gan.genLocal:
                    dropIndex = data["indexMap"].cuda()
                else:
                    dropIndex = None
                
                self.gan.oneTrainStep(inputImg, realImg, dropIndex)

            # Print and Save Log
            if epoch % 10 == 0:
                loss  = np.mean(self.gan.loss["LOSS"])
                lossG = np.mean(self.gan.loss["G"])
                lossD = np.mean(self.gan.loss["D"])
                acc   = np.mean(self.gan.acc["PSNR"])

                stringLog = f"Train {epoch} / {self.gan.epochs} Epochs | LOSS {loss} | PSNR Acc {acc} | G {lossG} | D {lossD}"
                self.summaryWriter.add_scalar("Train Loss", loss, epoch)
                self.summaryWriter.add_scalar("Train G Loss", lossG, epoch)
                self.summaryWriter.add_scalar("Train D Loss", lossD, epoch)
                self.summaryWriter.add_scalar("Train PSNR Acc", acc, epoch)
                self.loss = {"LOSS":[], "G":[], "D":[]}
                self.acc  = {"PSNR":[], "valPSNR":[]}

                if self.gan.genLocal:
                    globalG = np.mean(self.gan.loss["GlobalG"])
                    localG  = np.mean(self.gan.loss["LocalG"])
                    globalD = np.mean(self.gan.loss["GlobalD"])
                    localD  = np.mean(self.gan.loss["LocalD"])
                    stringLog += f" | Global G {globalG} | Local G {localG} | Global D {globalD} | Local D {localD}"
                    self.summaryWriter.add_scalar("Train Global G Loss", globalG, epoch)
                    self.summaryWriter.add_scalar("Train Global D Loss", globalD, epoch)
                    self.summaryWriter.add_scalar("Train Local G Loss", localG, epoch)
                    self.summaryWriter.add_scalar("Train Local D Loss", localD, epoch)
                    self.gan.loss = {"LOSS":[], "G":[], "D":[], "GlobalG" : [], "GlobalD": [], "LocalG":[], "LocalD":[]}

                print(stringLog)
            
            if epoch % self.gan.saveIter == 0:
                
                self.gan.netG.eval()
                self.gan.netD.eval()

                with torch.no_grad():
                    for index, data in enumerate(self.loaderVal):
                        torch.cuda.empty_cache()
                        gc.collect()
                        inputImg = data["inputs"].cuda()
                        realImg  = data["real"].cuda()

                        if self.gan.genLocal:
                            dropIndex = data["indexMap"].cuda()
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
                    acc   = np.mean(self.gan.acc["valPSNR"])
                    stringLog = f"Validation {epoch} / {self.gan.epochs} Epochs | PSNR Acc {acc}"
                    self.summaryWriter.add_scalar("Validation PSNR Acc", acc, epoch)
                    self.gan.acc["valPSNR"] = []
                    print(stringLog)
                
                save(self.gan.ckptDir, self.gan.netG, self.gan.netD, self.gan.optimG, self.gan.optimD, epoch, self.name)

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
        self.gan = PPransGAN(self.jsonData, gpu="cuda:0" if torch.cuda.is_available() else "cpu") 

        # Initialize Trainer's 
        Base_Trainer.__init__(self, self.gan.pathDB, self.gan.ckptDir, self.gan.logDir, self.gan.patch_size, self.gan.img_size, self.gan.N, self.gan.batchSize, self.sharedFilePath)  
        self.name = self.gan.name + "_PPransGAN"
        self.makeEvalDatasets(maskPath)
        
        
        self.gan.netG, self.gan.netD, self.gan.optimG, self.gan.optimD, self.startEpoch = load(self.gan.netG, self.gan.netD, self.gan.optimG, self.gan.optimD, self.startEpoch)
        print(f"Load Done.. Model Name : {self.name} | {self.startEpoch}")
        print(f"Path :{self.gan.pathDB}")

        self.gan.netG.eval()
        self.gan.netD.eval()

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
            nowAcc  = np.mean(self.gan.acC["valPSNR"])
            stringLog = f"Evaluation End : {self.startEpoch} Epochs | PSNR Acc {nowAcc}"
            print(stringLog)
    
    def visual_init(self, epoch=None):
        r"""
        For Inference Mode, runing when using visual.py 
        This step just Initialize and Load Model
        """
        gpu = 0
        self.gan = PPransGAN(self.jsonData, gpu="cuda:0" if torch.cuda.is_available() else "cpu") 

        # Initialize Trainer's 
        Base_Trainer.__init__(self, self.gan.pathDB, self.gan.ckptDir, self.gan.logDir, self.gan.patch_size, self.gan.img_size, self.gan.N, self.gan.batchSize, self.sharedFilePath)  
        self.name = self.gan.name +"_PPransGAN"
        self.makeEvalDatasets()

        self.gan.netG, self.gan.netD, self.gan.optimG, self.gan.optimD, self.startEpoch = load(self.gan.netG, self.gan.netD, self.gan.optimG, self.gan.optimD, self.startEpoch)
        self.gan.netG.eval()
        self.gan.netD.eval()

    
    def runTrain(self):
        self.train()
    
    def runEval(self, vis=False, si=False,mix=True, epcoh=None, maskPath=None):
        self.eval(vis, si, mix, epoch=epcoh, maskPath=maskPath)


if __name__ == "__main__":

    size = torch.cuda.device_count()
    testTrainer = PPransGANtrainer("TrainParameterExample.json")
    testTrainer.runTrain()
    # testTrainer = mp.spawn(, args=(size,"TrainParameterExample.json"), nprocs=size)
    # testTrainer.runTrain()



            
        

        

        