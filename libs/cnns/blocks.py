import torch
from torch import nn

from .layers import *
from .utils import *

r"""
    Original Code :
        https://github.com/godisboy/SN-GAN
    
    Reference :
        https://arxiv.org/pdf/1802.05957.pdf (SN Patch GAN Paper)
        https://arxiv.org/abs/2102.07074 (Trans GAN Paper)
        https://arxiv.org/pdf/2105.10189v1.pdf (Combining Transformer Generators with CNN)

    This is SN Patch-GAN Discriminator
    According to https://arxiv.org/abs/2102.07074, using just Transformer is not good performance
    because of Data hungry model
    according to https://arxiv.org/pdf/2105.10189v1.pdf,
    using SN Patch GAN Discriminator is best performance with Transformer Generator
    so our networks testing SN Patch-GAN discriminator
    """

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, nker=64, pad_type="zero", activation="relu", norm="bn"):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(in_channels, nker, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block2 = Conv2dLayer(nker, nker * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block3 = Conv2dLayer(nker * 2, nker * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block4 = Conv2dLayer(nker * 4, nker * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block5 = Conv2dLayer(nker * 4, nker * 4, 1, 1, 0, pad_type = pad_type, activation = activation, norm = norm, sn = True)
        self.block6 = Conv2dLayer(nker * 4, 1, 1, 1, 0, pad_type = pad_type, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = self.block1(img)                                    # out: [B, 64, 128, 128]
        x = self.block2(x)                                      # out: [B, 128, 64, 64]
        x = self.block3(x)                                      # out: [B, 256, 32, 32]
        x = self.block4(x)                                      # out: [B, 256, 16, 16]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 1, 16, 16]

        return x