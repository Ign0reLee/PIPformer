
import numpy as np

import torch
import torch.nn as nn

from libs.models.baseGAN import baseGAN
from libs.models.metrics import PSNR
from libs.transformers import GenerateTransformer


class GEL2(baseGAN):
    def __init__(self, jsonPath, gpu):
        baseGAN.__init__(self, jsonPath)

        self.gpu = gpu
        self.model =  GenerateTransformer(self.n_layers, self.emb_size, self.num_heads,
                                        self.dff, self.in_channels, self.randomNoise,
                                        self.patch_size, self.img_size, self.rate, self.ffn_rate).cuda(gpu)
        # self.model.generator.positions.cuda(gpu)
        self.optimizers = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr= self.lr, betas=[0, 0.99])
        self.loss_fn = nn.MSELoss().cuda(gpu)
        self.metrics = PSNR().cuda(gpu)

        # Set Log
        self.lossLog = []
        self.lossVal = []  
        self.accLog  = []
        self.accVal  = []

        if self.genLocal:
            self.globalLoss = []
            self.localLoss  = []
            self.globalVal = []
            self.localVal  = []   
        
    
    def oneTrainStep(self, inputImg, realImg, index):

        self.optimizers.zero_grad()

        output, _ = self.model(inputImg)
        # output = torch.clamp(output, min=0, max=1)
        loss   = self.loss_fn(output, realImg)
        acc    = self.metrics(output, realImg)
        if index is not None: 
            
            localOutput = torch.mul(output, index)
            localReal  = torch.mul(realImg, index)
                                
            self.globalLoss += [loss.item()]
            localMSEloss    = self.loss_fn(localOutput, localReal)
            loss = (self.genGlobalrate * loss) + (self.genLocalrate * localMSEloss)

            self.localLoss += [localMSEloss.item()]

        self.lossLog += [loss.item()]
        self.accLog  += [acc.item()]

        loss.backward(retain_graph=True)
        self.optimizers.step()
    
    def oneValStep(self, inputImg, realImg, index=None, postpro=True):

        output, attn_weight = self.model(inputImg)
        # output = torch.clamp(output, min=0, max=1)
        if postpro is True and index is not None:
            output = torch.mul(output,index) + inputImg
        loss   = self.loss_fn(output, realImg)
        acc    = self.metrics(output, realImg)

        if index is not None:
            
            localOutput = torch.mul(output, index)
            localReal  = torch.mul(realImg, index)

            self.globalVal += [loss.item()]
            localMSEloss    = self.loss_fn(localOutput, localReal)
            loss = (self.genGlobalrate * loss) + (self.genLocalrate * localMSEloss)

            self.localVal += [localMSEloss.item()]

        self.lossVal += [loss.item()]
        self.accVal  += [acc.item()]

        return output, attn_weight

    
    def outStep(self, inputImg, index=None):
        if index is not None:
            output, attn = self.model(inputImg)
            output = torch.mul(output, index)
            return output + inputImg, attn
        else:
            return self.model(inputImg)


if __name__ =="__main__":
    test_tensor = torch.randn((8, 3, 256, 256))
    test_model = GEL2("TrainParameterExample.json")

    test_output = test_model.outStep(test_tensor)
    print(f"{test_output.size()}")
    # mp.spawn(GEL2, args= (2,"TrainParameterExample.json"), join=True)
    