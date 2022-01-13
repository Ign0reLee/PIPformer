import cv2
import numpy as np


class NPatchDrop(object):
    # NPatchDrop

    def __init__(self, patchSize=16):
        self.patchSize = patchSize

    def __call__(self, data):
        img = data["inputs"]
        index = data["index"]
        indexMap = np.zeros_like(img)

        # Drop Patch
        for id in index:
            startH, startW = id * self.patchSize
            endH, endW     = (id + 1) * self.patchSize
            img[startH:endH, startW:endW, :]      = 0
            indexMap[startH:endH, startW:endW, :] = 1
            
        
        data["inputs"] = img
        data["indexMap"] = indexMap
        return data

class MaskDrop(object):
    # NPatchDrop

    def __init__(self):
        self.name = "Mask Drop Mode"

    def __call__(self, data):
        img = data["inputs"]
        index = data["mask"]

        img = img * (1-index)
        
        data["inputs"] = img
        data["indexMap"] = index
        return data


class Resize(object):

    # ReSizing Data

    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        inputs = data["inputs"]
        real = data["real"]

        data["inputs"] = cv2.resize(inputs, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)      
        data["real"] = cv2.resize(real, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)
        
        return data

class Normalization(object):
    
    # Normalized Data

    def __init__(self, mean=0.5, std= 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):

        inputs = data["inputs"]
        real = data["real"]

        data["inputs"] = (inputs - self.mean) / self.std  
        data["real"] = (real - self.mean) / self.std

        return data