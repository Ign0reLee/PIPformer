import copy

import torch
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

        self.toRGB = nn.Sequential(
            Rearrange('b (h w) e -> b e (h) (w)', h=img_size//patch_size),
            # in ViT, using a conv layer instead of a linear one -> performance gains
            nn.ConvTranspose2d(emb_size, in_channels, kernel_size=patch_size, stride=patch_size),
        )
    
    def forward(self, inputs, enc_padding_mask=None):
        output, attentionWegihts = self.generator(inputs, enc_padding_mask)
        final_output =self.toRGB(output)
        return final_output, attentionWegihts


class DiscriminateTransformer(nn.Module):
    def __init__(self,
                n_layers,
                emb_size,
                num_heads,
                dff,
                in_channels=3,
                patch_size = 16,
                img_size = 256,
                rate=0.5,
                ffn_rate=0.5,
                eps=1e-6,
                n_classes=1,
                local=False):

        super(DiscriminateTransformer, self).__init__()

        self.emb_size = emb_size

        self.discriminator = Discriminator(n_layer=n_layers,
                                        emb_size=emb_size,
                                        num_heads=num_heads,
                                        dff=dff,
                                        patch_size=patch_size,
                                        img_size=img_size,
                                        in_channels=in_channels,
                                        rate=rate,
                                        ffn_rate=ffn_rate, 
                                        eps=eps)
        self.classificationHead = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
            nn.Sigmoid()
        )

        if local:
            self.localClassificationHead = nn.Sequential(
                nn.LayerNorm(emb_size),
                nn.Linear(emb_size, n_classes),
                nn.Sigmoid()
            )
        else:
            self.localClassificationHead=False

    def forward(self, inputs, dropIndex=None, enc_padding_mask=None):
        output, attn_wegihts = self.discriminator(inputs, enc_padding_mask)
        global_critic = self.classificationHead(output)

        if self.localClassificationHead:
            dropIndex = torch.LongTensor(dropIndex).expand(-1, -1, self.emb_size)
            localPatch = torch.gather(output, dim=1, index=dropIndex)
            local_critic = self.localClassificationHead(localPatch)
            return global_critic, local_critic, attn_wegihts

        return global_critic, attn_wegihts


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

        self.projection = nn.Sequential(
            # in ViT, using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    
        # Positional Encoding
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2, emb_size))
        # self.positions = nn.Parameter(positional_encoding((img_size//patch_size**2), emb_size))
        self.gen_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gen_layers.append(TransformerLayer(d_embed=emb_size, d_model=emb_size, num_heads=num_heads, dff=dff, rate=rate, ffn_rate=ffn_rate, eps=eps))

        self.dropout = nn.Dropout(p=rate)

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


class Discriminator(nn.Module):
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
                 eps=1e-6):

        super(Discriminator, self).__init__()
        
        self.emb_size = emb_size
        self.n_layers = n_layer
        
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        self.gen_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gen_layers.append(TransformerLayer(d_embed=emb_size, d_model=emb_size, num_heads=num_heads, dff=dff, rate=rate, ffn_rate=ffn_rate, eps=eps))

        self.dropout = nn.Dropout(p=rate)

    def forward(self, x, mask=None):
        batch_size        = x.size()[0]
        attentionWegihts  = {}
        # Projection
        x = self.projection(x)
        # Concat CLS Tokens
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add Positional Encoding
        x += self.positions
        x = self.dropout(x)

        for i, layer in enumerate(self.gen_layers):
            x, attnBlock = layer(x, mask)
            attentionWegihts[f"Discriminator_Layer{i}"] = attnBlock
            
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

    test_discriminator = Discriminator(n_layer=n_layer, in_channels=3,emb_size=emb_size, num_heads=num_heads, dff=dff)
    output, attn = test_discriminator(test_tensor)
    attn_shape   = attn["Discriminator_Layer1"].size()
    print(f"Discriminator Output : {output.size()}")
    print(f"Discriminator Attn   : {attn_shape}")

    test_discriminator = DiscriminateTransformer(n_layers=n_layer, in_channels=3,emb_size=emb_size, num_heads=num_heads, dff=dff)
    output, attn = test_discriminator(test_tensor, None)
    attn_shape   = attn["Discriminator_Layer1"].size()
    print(f"Discriminator Transformers Output : {output.size()}")
    print(f"Discriminator Transformers Attn   : {attn_shape}")

    test_discriminator = DiscriminateTransformer(n_layers=n_layer, in_channels=3,emb_size=emb_size, num_heads=num_heads, dff=dff, local=True)
    output, local_output, attn = test_discriminator(test_tensor, np.array([0,1,2])[np.newaxis, :, np.newaxis],None)
    attn_shape   = attn["Discriminator_Layer1"].size()
    print(f"Discriminator Transformers Output : {output.size()}")
    print(f"Discriminator Transformers Local Output : {local_output.size()}")
    print(f"Discriminator Transformers Attn   : {attn_shape}")