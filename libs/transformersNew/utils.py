import numpy as np

import torch

import matplotlib.pyplot as plt

def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(1e4, (2 * (i//2))/ np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:,np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.from_numpy(pos_encoding).type(torch.float32)

if __name__=="__main__":
    n, d= 2048, 512
    pos_encoding = positional_encoding(n,d)
    print(pos_encoding.shape)

    pos_encoding = pos_encoding[0]
    pos_encoding = torch.reshape(pos_encoding, (n, d//2, 2))
    pos_encoding = torch.transpose(pos_encoding, 2, 0)
    pos_encoding = torch.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap= 'RdBu')
    plt.ylabel("Depth")
    plt.xlabel("Position")
    plt.colorbar()
    plt.show()