import numpy as np

def Normalization(img, mean=0.5, std=0.5):
    return (img - mean) / std  

def DropPatch(img, indexMap):
    return img * indexMap