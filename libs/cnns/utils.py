import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter

r"""
Original code :
    https://github.com/csqiangwen/DeepFillv2_Pytorch
    https://github.com/godisboy/SN-GAN
"""

def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

def max_singular_value( W, u=None, Ip=1):
        r"""
        power iteration for weight parameter
        """
        #xp = W.data
        if not Ip >= 1:
            raise ValueError("Power iteration should be a positive integer")
        if u is None:
            u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
        _u = u
        for _ in range(Ip):
            _v = l2normalize(torch.matmul(_u, W.data), eps=1e-12)
            _u = l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
        sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
        return sigma, _u


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)