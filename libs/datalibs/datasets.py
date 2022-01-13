import os
import cv2
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None, patchSize=16, size=(256,256), N=7, task=None, paintingSize=1, maskPath=None):
        self.data_dir = data_dir
        self.maskPath = maskPath
        self.transform = transforms
        self.patchSize = patchSize

        self.lst_data = os.listdir(self.data_dir)
        self.lst_data.sort()
        if maskPath is not None:
            self.mask_lst = os.listdir(self.maskPath)
            self.mask_lst.sort()
        

        self.N = N
        self.task = task

        self.patchHNumber = size[0] // patchSize
        self.patchWNumber = size[1] // patchSize

        if self.task is None:
            self.patchIndex = np.array([np.array([i, j]) for i in range(self.patchHNumber) for j in range(self.patchWNumber)])
            self.patchIndexSize = len(self.patchIndex)

        elif self.task == "inpainting":
            self.paintingSize = paintingSize 
            if paintingSize%2 == 0:
                self.odd = False
            else:
                self.odd = True
            paintingSize += 1 # For remaining at least 1 pixel
            self.patchIndex = np.array([np.array([i, j]) for i in range(paintingSize ,self.patchHNumber-paintingSize) for j in range(paintingSize,self.patchWNumber-paintingSize)])
            self.patchIndexSize = len(self.patchIndex)

        elif self.task == "outpainting":
            paintingIndex = list(range(paintingSize)) + list(range(self.patchHNumber-paintingSize, self.patchHNumber))
            line0 = np.array([np.array([i, j]) for i in paintingIndex for j in range(self.patchWNumber)])
            line1 = np.array([np.array([i, j]) for j in paintingIndex for i in range(paintingSize, self.patchHNumber-paintingSize)])
            self.patchIndex = np.concatenate((line0, line1), axis=0)

        self.to_tensor = ToTensor()
    
    def __len__(self):
        return len(self.lst_data)
    
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_dir, self.lst_data[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        

        # Image Scaling
        if img.dtype == np.uint8:
            img = img / 255.0

        data = {"real"   : img} # Label
        data["inputs"]   = img # Inputs

        if self.maskPath is not None:
            mask = cv2.imread(os.path.join(self.maskPath, self.mask_lst[index]), cv2.IMREAD_COLOR)
            mask = mask / 255.0
            data["mask"] = mask

        # If Task is Random Drop Patch Index
        if self.task is None:
            dropIndex =  np.random.choice(self.patchIndexSize, size=self.N)
            data["index" ] = [self.patchIndex[index] for index in dropIndex]
            data["flat_index"] = dropIndex[:, np.newaxis]
        
        elif self.task == "inpainting":
            CenterIndex = np.random.randint(0, high=self.patchIndexSize, dtype=np.int32)
            CenterIndex = self.patchIndex[CenterIndex]
            if self.odd:
                miny, minx = CenterIndex - (self.paintingSize//2)
                maxy, maxx = CenterIndex + (self.paintingSize//2)
                data["index"] = [np.array([i, j]) for i in range(miny, maxy) for j in range(minx, maxx)]
            
            else:
                randomFactorL = np.random.uniform(0,1)
                randomFactorR = np.around(1 - randomFactorL)
                miny, minx = CenterIndex - (self.paintingSize//2 - np.around(randomFactorL))
                maxy, maxx = CenterIndex + (self.paintingSize//2 - randomFactorR)
                data ["index"] =  [np.array([i, j]) for i in range(miny, maxy) for j in range(minx, maxx)]

        elif self.task == "outpainting":
            data["index"] = self.patchIndex
        
        # If Task is Inpainting
        # Data Transforming
        if self.transform:
            data = self.transform(data)
        
        data = self.to_tensor(data)

        return data

class ToTensor(object):

    # Make Image [B, H, W, C] to [B, C, H, W]
    # Make Image Numpy Array to Tensor

    def __call__(self, data):

        for key, value in data.items():
            if key == "inputs" or key == "real" or key=="indexMap":
                value = value.transpose((2, 0, 1)).astype(np.float32)
                data[key] = torch.from_numpy(value)
            else:
                data[key] = torch.from_numpy(np.array(value))

        return data



if __name__ == "__main__":
    testPath = "../Data/FFHQ"
    testDataset = Dataset(data_dir=testPath, patchSize=16, size=(256,256), N=7, task="outpainting", paintingSize=2)

    print(f"{testDataset.patchWNumber}, {testDataset.patchIndex}, {np.shape(testDataset.patchIndex)}")
  