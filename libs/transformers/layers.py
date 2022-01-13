import numpy as np

import torch
import torch.nn as nn



def WGANGPGradientPenalty(input, fake, discriminator, weight, backward=True, genLocal=False, indexMap=None):
    r"""
    Gradient penalty as described in
    "Improved Training of Wasserstein GANs"
    https://arxiv.org/pdf/1704.00028.pdf

    Original Reference Code
    https://github.com/facebookresearch/pytorch_GAN_zoo

    Args:

        - input (Tensor): batch of real data
        - fake (Tensor): batch of generated data. Must have the same size
          as the input
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """

    batchSize = input.size(0)
    alpha = torch.rand(batchSize, 1)
    alpha = alpha.expand(batchSize, int(input.nelement()/batchSize)).contiguous().view(input.size())
    alpha = alpha.to(input.device)
    interpolates = alpha * input + ((1 - alpha) * fake)

    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    
    if genLocal:
        decisionInterpolate, _, _= discriminator(interpolates, dropIndex=indexMap)
    else:    
        decisionInterpolate, _ = discriminator(interpolates)
    decisionInterpolate = decisionInterpolate[:, 0].sum()

    gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                    inputs=interpolates,
                                    create_graph=True, retain_graph=True)

    gradients = gradients[0].view(batchSize, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = (((gradients - 1.0)**2)).sum() * weight

    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, d_model, dff, rate=0.5):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=rate)
        self.fc2 = nn.Linear(dff, d_model)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model,  d_embed, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.prob = nn.Softmax(dim=-1)

        self.wq = nn.Linear(d_embed, d_model)
        self.wk = nn.Linear(d_embed, d_model)
        self.wv = nn.Linear(d_embed, d_model)
        
        self.fc = nn.Linear(d_model, d_model)

    def split_head(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1,2)

    def calculate_attention(self, value, key, query, mask=None):
        # Query Shape: (Batch, d_k, Squence_Length)
        # Key   Shape: (Batch, d_k, Squence_Length)
        # Value Shape: (Batch, d_k, Squence_Length)
        d_k = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = attention_score / np.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, 1e-9)
        attention_prob = self.prob(attention_score)
        out = torch.matmul(attention_prob, value)
        return out, attention_prob

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query = self.wq(query)
        key   = self.wk(key)
        value = self.wv(value)

        query = self.split_head(query, batch_size)
        key   = self.split_head(key,   batch_size)
        value = self.split_head(value, batch_size)

        if mask is not None:
            mask = mask.unsqueeze(1)

        scaled_attetnion, attention_weights = self.calculate_attention(query, key, value, mask)
        out = scaled_attetnion.transpose(1,2)
        out = out.contiguous().view(batch_size, -1,  self.d_model)
        out = self.fc(out)
        return out, attention_weights
    

class TransformerLayer(nn.Module):

    def __init__(self,
                d_embed,
                d_model,
                num_heads,
                dff,
                rate=0.1,
                ffn_rate=0.5,
                eps=1e-6):

        super(TransformerLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_embed=d_embed, d_model=d_model, num_heads=num_heads)
        self.ffn = PositionWiseFeedForwardLayer(d_model=d_model, dff=dff, rate=ffn_rate)

        self.layernorm1 = nn.LayerNorm(d_model, eps=eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=eps)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None):
        x = self.layernorm1(x)
        attn_output, attn_weight = self.mha(x, x, x, mask)
        attn_output    = self.dropout1(attn_output)
        out1           = self.layernorm2(x + attn_output)

        ffn_output     = self.ffn(out1)
        ffn_output     = self.dropout2(ffn_output)
        out2           = out1 + ffn_output
        return out2, attn_weight
