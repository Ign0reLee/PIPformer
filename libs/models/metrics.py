import torch
import torch.nn as nn

from torch import Tensor

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.name = "PSNR"

    @staticmethod
    def __call__(img1:Tensor, img2:Tensor, dataRange=1.0):
        # img's shape must be [B, C, H, W]
        # img's range must be same as data_range
        mse = torch.mean((img1 - img2) ** 2)
        return 10 * torch.log10((dataRange **2) / mse)


if __name__ == "__main__":
    img1 = torch.randn((1,3,224,224))
    img2 = torch.randn((1,3,224,224))
    psnr = PSNR()
    print(psnr(img1, img2).item())