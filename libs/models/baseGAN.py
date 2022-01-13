import json
import numpy as np
import matplotlib.pyplot as plt

import torch

class baseGAN():
    def __init__(self, jsonPath):
        self.__jsonParser__(jsonPath)
        
        # Set Default Additional Function
        self.fn_toNumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1)
        self.fn_toDenorm = lambda x, mean, std: (x * std) + mean
    
    def makeVisual(self, imageTensor):
        return np.clip(self.fn_toNumpy(self.fn_toDenorm(imageTensor, 0.5, 0.5)).squeeze(), a_min=0, a_max=1)
    
    def makeAttnVisual(self, writer, attn, step, name):
        fig = plt.figure() 
        fig = plt.figure(figsize=(16, 8))
    
        for h, head in enumerate(attn):
            
            ax = fig.add_subplot(2, 4, h+1)
            attnax = ax.matshow(head)
            fig.colorbar(attnax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            ax.grid(False)     
            ax.set_xlabel(f'{name} Head {h+1}')
        
        plt.tight_layout()
        self.plot_to_tensorboard(writer, fig, step, name)
        plt.close()

    def plot_to_tensorboard(self, writer, fig, step, name):
        """
        Takes a matplotlib figure handle and converts it using
        canvas and string-casts to a numpy array that can be
        visualized in TensorBoard using the add_image function

        Parameters:
            writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
            fig (matplotlib.pyplot.fig): Matplotlib figure handle.
            step (int): counter usually specifying steps/epochs/time.
        """

        # Draw figure on canvas
        fig.canvas.draw()

        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
        img = img / 255.0
        # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

        # Add figure in numpy "image" to TensorBoard writer
        writer.add_image(f'attention_{name}', img, step, dataformats="HWC")
        plt.close(fig)
    
    
    def __jsonParser__(self, jsonPath):

        with open(jsonPath, "r") as jsonFile:
            jsonData = json.load(jsonFile)
            
            # Dataset Information
            self.name = jsonData["name"]
            self.pathDB = jsonData["path"]
            self.ckptDir = jsonData["ckpt_dir"]
            self.logDir  = jsonData["log_dir"]

            # Image Information
            self.img_size= jsonData["img_size"]
            self.in_channels= jsonData["in_channels"]

            # Train Information
            self.batchSize = jsonData["batchSize"]
            self.lr        = jsonData["learningRate"]
            self.epochs    = jsonData["trainEpochs"]
            self.saveIter  = jsonData["saveIter"]

            # Patch Information
            self.patch_size= jsonData["patch_size"]
            self.N         = jsonData["dropNumber"]

            # Model Architecture Information
            self.n_layers = jsonData["n_layers"]
            self.emb_size = jsonData["emb_size"]
            self.num_heads= jsonData["num_heads"]
            self.dff= jsonData["dff"]

            # Model Hyper-Parameter Information
            self.rate= jsonData["rate"]
            self.ffn_rate= jsonData["ffn_rate"]
            self.genGlobalrate = jsonData["genGlobalrate"]
            self.genLocalrate  = jsonData["genLocalrate"]
            self.labmdaGP      = jsonData["wganGPrate"]
            
            # Generator Information
            self.randomNoise= bool(jsonData["randomNoise"])
            self.genLocal   = bool(jsonData["genLocal"])
