import copy

import torch
import numpy as np
from torch import nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Linear

from .utils import *
from .layers import *

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class GenerateTransformer(nn.Module):
    def __init__(self,
                n_layers,
                emb_size,
                num_heads,
                dff,
                in_channels,
                randomNoise=False,
                patch_size = 16,
                img_size = 256,
                rate=0.5,
                ffn_rate=0.5,
                eps=1e-6):

        super(GenerateTransformer, self).__init__()

        self.generator = Generator(n_layer=n_layers,
                                 emb_size =emb_size,
                                 num_heads=num_heads,
                                 dff = dff,
                                 patch_size=patch_size,
                                 img_size=img_size,
                                 in_channels=in_channels,
                                 rate=rate,
                                 ffn_rate=ffn_rate, 
                                 eps=eps,
                                 randomNoise=randomNoise)
        # Original Code
        self.toRGB = nn.Sequential(
            # Rearrange('b (h w) e -> b e (h) (w)', h=img_size//patch_size),
            # in ViT, using a conv layer instead of a linear one -> performance gains
            nn.ConvTranspose2d(emb_size, in_channels, kernel_size=patch_size, stride=patch_size),
        )
        # # First Code
        # self.toRGB = nn.Sequential()
        # self.makeToRGB(in_channels, emb_size, patch_size, img_size)

        # # Seconde Code
        # self.toRGB = nn.Sequential(
        #     Rearrange('b (h w) e -> b e (h) (w)', h=img_size//patch_size),
        #     # in ViT, using a conv layer instead of a linear one -> performance gains
        #     nn.ConvTranspose2d(emb_size, 256, kernel_size=2, stride=2),
        #     nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        #     nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        #     nn.ConvTranspose2d(256, in_channels, kernel_size=2, stride=2),
        # )

    def makeToRGB(self, in_channels, emb_size, patch_size, img_size):
        blocks = int(np.log2(patch_size))
        i = 1

        self.toRGB.add_module("reshape", Rearrange('b (h w) e -> b e (h) (w)', h=img_size//patch_size))
        self.toRGB.add_module(f"deconv_{i}", nn.ConvTranspose2d(emb_size, emb_size//(2**i), kernel_size=2, stride=2))

        for index in range(1, blocks-1):
            i += 1
            self.toRGB.add_module(f"deconv_{i}", nn.ConvTranspose2d(emb_size//(2**index), emb_size//(2**(index+1)), kernel_size=2, stride=2))
        
        self.toRGB.add_module(f"deconv_{i+1}", nn.ConvTranspose2d(emb_size//(2**(index+1)), in_channels, kernel_size=2, stride=2))

        
    
    def forward(self, inputs, enc_padding_mask=None):
        output, attentionWegihts = self.generator(inputs, enc_padding_mask)
        final_output =self.toRGB(output)
        return final_output, attentionWegihts


class Generator(nn.Module):
    def __init__(self,
                 n_layer,
                 emb_size,
                 num_heads,
                 dff,
                 patch_size=16,
                 img_size=256,
                 in_channels=3,
                 rate=0.1,
                 ffn_rate=0.1, 
                 eps=1e-6,
                 randomNoise=False):

        super(Generator, self).__init__()
        
        self.d_model = emb_size
        self.n_layers = n_layer
        self.randomNoise = randomNoise
        
        # Original Code(Just One CNN to make Embedding)
        self.projection = nn.Sequential(
            # in ViT, using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # #First Code
        # self.projection = nn.Sequential()
        # self.makeProjection(in_channels, emb_size, patch_size)
        # # Sencond Code
        # self.projection = nn.Sequential(
        #     nn.Conv2d(in_channels, 256, kernel_size=2, stride=2),
        #     nn.Conv2d(256, 256, kernel_size=2, stride=2),
        #     nn.Conv2d(256, 256, kernel_size=2, stride=2),
        #     nn.Conv2d(256, emb_size, kernel_size=2, stride=2),
        #     Rearrange('b e (h) (w) -> b (h w) e'),
        # )
        # Positional Encoding
        self.positions = nn.Parameter(torch.randn(emb_size, (img_size // patch_size), (img_size // patch_size)))
        # self.positions = nn.Parameter(positional_encoding((img_size//patch_size**2), emb_size))
        self.gen_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gen_layers.append(TransformerLayer(d_embed=emb_size, d_model=emb_size, num_heads=num_heads, dff=dff, rate=rate, ffn_rate=ffn_rate, eps=eps, img_size=img_size//patch_size))

        self.dropout = nn.Dropout(p=rate)

    def makeProjection(self, in_channels, emb_size, patch_size):
        blocks = int(np.log2(patch_size)) - 1
        i = 1
        self.projection.add_module(f"conv_{i}",nn.Conv2d(in_channels, emb_size//(2**blocks), kernel_size=2, stride=2))

        for index in range(blocks, 0, -1):
            i += 1
            self.projection.add_module(f"conv_{i}",nn.Conv2d(emb_size//(2**index), emb_size//(2**(index-1)), kernel_size=2, stride=2))

        self.projection.add_module("flatten", Rearrange('b e (h) (w) -> b (h w) e'))

    def forward(self, x, mask=None):
        # Initalize Weight Dictionary
        attentionWegihts  = {}

        # Projection
        x = self.projection(x) 
        # Random Noise
        if self.randomNoise:
            x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # Add Positional Encoding
        x += self.positions
        x = self.dropout(x)

        for i, layer in enumerate(self.gen_layers):
            x, attnBlock = layer(x, mask)
            attentionWegihts[f"Generator_Layer{i}"] = attnBlock
            
        return x, attentionWegihts


    
if __name__ =="__main__":
    n_layer=6
    emb_size=768
    num_heads=8
    dff=1024
    patch_size=16
    img_size=256
    in_channels=3
    rate=0.1
    ffn_rate=0.1
    eps=1e-6

    test_generator = Generator(n_layer=n_layer, emb_size=emb_size, num_heads=num_heads, dff=dff)
    test_tensor = torch.randn((1, 3, 256, 256))
    output, attn = test_generator(test_tensor)
    attn_shape   = attn["Generator_Layer1"].size()
    print(f"Generator Output : {output.size()}")
    print(f"Generator Attn   : {attn_shape}")

    test_generatortrans = GenerateTransformer(n_layers=n_layer, in_channels=3,emb_size=emb_size, num_heads=num_heads, dff=dff)
    output, attn = test_generatortrans(test_tensor, None)
    attn_shape   = attn["Generator_Layer1"].size()
    print(f"Generator Transformers Output : {output.size()}")
    print(f"Generator Transformers Attn   : {attn_shape}")
