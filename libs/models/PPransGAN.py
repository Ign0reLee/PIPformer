
import numpy as np

import torch
import torchvision
import torch.nn as nn

from libs.models.baseGAN import baseGAN
from libs.models.metrics import PSNR
from libs.transformers import GenerateTransformer, DiscriminateTransformer, WGANGPGradientPenalty

from libs.cnns import PatchDiscriminator

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x

# ----------------------------------------
#            Patch-based Painting GAN Network
# ----------------------------------------

class PPransGAN(baseGAN):
    def __init__(self, jsonPath, gpu):
        baseGAN.__init__(self, jsonPath)

        self.gpu = gpu
        self.netG =  GenerateTransformer(self.n_layers, self.emb_size, self.num_heads,
                                        self.dff, self.in_channels, self.randomNoise,
                                        self.patch_size, self.img_size, self.rate, self.ffn_rate).cuda(gpu)

        # self.netD = DiscriminateTransformer(self.n_layers, self.emb_size, self.num_heads,
        #                                     self.dff, self.in_channels, self.patch_size, self.img_size,
        #                                     self.rate, self.ffn_rate, local=self.genLocal).cuda(gpu)
        self.netD = PatchDiscriminator().cuda(gpu)
        self.perceptualNet = PerceptualNet().cuda(gpu)

        self.optimG = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netG.parameters()), lr= self.lr, betas=[0, 0.99])
        self.optimD = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr= self.lr, betas=[0, 0.99])
        self.l1Loss  = nn.L1Loss().cuda(gpu)
        self.mseLoss = nn.MSELoss().cuda(gpu)
        self.advLoss = nn.BCELoss().cuda(gpu)
        self.metrics = PSNR().cuda(gpu)

        # Set Log
        self.loss = {"LOSS":[], "G":[], "D":[]}
        self.acc  = {"PSNR":[], "valPSNR":[]}

        if self.genLocal:
            self.loss["GlobalG"] = []
            self.loss["GlobalD"] = []
            self.loss["LocalG"] = []
            self.loss["LocalD"] = []

    
    def oneTrainStep(self, inputImg, realImg, dropIndex=None):
        # Update the Discrfiminator
        self.optimD.zero_grad()

        # Make Fake Data
        fakeOut, _ = self.netG(inputImg)

        # Pred Real and Pred Fake

        #############################################
        #   WGAN-GP
        #############################################
        # if self.genLocal:
        #     predGlobalReal, predLocalReal, _ = self.netD(realImg, dropIndex)
        #     lossLocalReal                    = -torch.sum(predLocalReal)
        #     predGlobalFake, predLocalFake, _ = self.netD(fakeOut, dropIndex)
        #     lossLocalFake                    = torch.sum(predLocalFake)
        # else:
        #     predGlobalReal, _ = self.netD(realImg)
        #     predGlobalFake, _ = self.netD(fakeOut)
        # lossGlobalReal = -torch.sum(predGlobalReal)
        # lossGlobalFake = torch.sum(predGlobalFake)
        # Calculate WGAN-GP
        # GP = WGANGPGradientPenalty(realImg, fakeOut, self.netD, self.labmdaGP, genLocal=self.genLocal, indexMap=dropIndex)
        # Loss Backward
        # lossGlobalD = lossGlobalReal + lossGlobalFake + GP
        # if self.genLocal:
        #     lossLocalD = lossLocalReal + lossLocalFake + GP
        #     self.loss["LocalD"].append(lossLocalD.item())
        #     self.loss["GlobalD"].append(lossGlobalD.item())
        
        # Make Loss D and backward 
        # lossD = (globalMSE * self.genGlobalrate) + lossGlobalD
        # if self.genLocal:
        #     lossD += (localMSE * self.genLocalrate) + lossLocalD

        #############################################
        #   SN-Patch GAN
        #############################################
        
        predGlobalReal = self.netD(realImg)
        predGlobalFake = self.netD(fakeOut)
        lossGlobalReal = -torch.mean(predGlobalReal)
        lossGlobalFake = torch.mean(predGlobalFake)

        # Calculate Global and Local
        lossD = 0.5 * (lossGlobalReal + lossGlobalFake)
        
        # Backward Loss
        self.loss["D"].append(lossD.item())
        lossD.backward(retain_graph=True)
        
        #Update One Step
        self.optimD.step()

        # Update the Generator
        self.optimG.zero_grad()
        self.optimD.zero_grad()

        # Make Fake Data
        fakeOut, _ = self.netG(inputImg)
        
        # ############################################
        #   MSE
        # ############################################
        # Calculate MSE
        globalMSE = self.mseLoss(fakeOut, realImg)
        if self.genLocal:
            localOutput = torch.mul(fakeOut, dropIndex)
            localReal  = torch.mul(realImg, dropIndex)
            localMSE = self.mseLoss(localOutput, localReal)

        # ############################################
        #   Perceptual Loss
        # ############################################
        # Calculate MSE
        fakeFeatures = self.perceptualNet(fakeOut)
        realFeatures = self.perceptualNet(realImg)
        perceptualLoss = self.l1Loss(realFeatures, fakeFeatures)

        #############################################
        #   WGAN-GP
        #############################################
        # Pred Fake Data
        # if self.genLocal:
        #     predGlobalFake, predLocalFake, _ = self.netD(fakeOut)
        #     lossLocalFake                    = -torch.sum(predLocalFake)
        # else:
        #     predGlobalFake, _ = self.netD(fakeOut)
        
        # lossGlobalFake = -torch.sum(predGlobalFake)
        # lossG = lossGlobalFake
        # if self.genLocal:
        #     lossG += lossLocalFake
        #     self.loss["LocalG"].append(lossGlobalFake.item())
        #     self.loss["GlobalG"].append(lossLocalFake.item())
        
        #############################################
        #   SN-Patch GAN
        #############################################
        predGlobalFake = self.netD(fakeOut)
        lossG = -torch.mean(predGlobalFake)
        loss  = lossG + (globalMSE * 10) + (perceptualLoss *10)
        if self.genLocal:
            loss += (localMSE *10)

        # Loss Backward
        self.loss["G"].append(lossG.item())
        self.loss["LOSS"].append(loss.item())
        # lossG.backward(retain_graph=True)
        loss.backward(retain_graph=True)

        # Update Generator one step
        self.optimG.step()

        acc    = self.metrics(fakeOut, realImg)
        self.acc["PSNR"].append(acc.item())
    
    def oneValStep(self, inputImg, realImg, index=None, postpro=True):
        output, attn_weight = self.netG(inputImg)
        if postpro is True and index is not None:
            output = torch.mul(output,index) + inputImg
        
        acc    = self.metrics(output, realImg)
        self.acc["valPSNR"].append(acc.item())

        return output, attn_weight

    
    def outGeneration(self, inputImg, index=None):
        if index is not None:
            output, attn = self.netG(inputImg)
            output = torch.mul(output, index)
            return output + inputImg, attn
        else:
            return self.netG(inputImg)


if __name__ =="__main__":
    test_tensor = torch.randn((8, 3, 256, 256))
    test_model = PPransGAN("TrainParameterExample.json")

    test_output = test_model.outStep(test_tensor)
    print(f"{test_output.size()}")
    # mp.spawn(GEL2, args= (2,"TrainParameterExample.json"), join=True)
    