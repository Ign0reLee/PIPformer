import cv2, os
import copy
import numpy as np
import tkinter as tk

import torch

import tkinter.messagebox as MessageBox
import tkinter.filedialog as FileDialog

import matplotlib.pyplot as plt

from PIL import Image as PILImage
from PIL import ImageTk

from tkinter import *
from .vis_utils import *

class tkWindow():
    def __init__(self, rootWin, patchSize, imgSize, trainer, visualPath, mix=True):
        self.rootWin = rootWin
        self.rootWin.geometry('800x800')
        self.rootWin.update()
        self.rootWin.title("Inference Patch Painting with Transformer")
        self.rootX =self.rootWin.winfo_width()
        self.rootY = self.rootWin.winfo_height()

        self.patchSize = patchSize
        self.visualPath = visualPath
        self.imgSize   = imgSize
        self.mix       = mix
        self.oriImgRelX = 0.1
        self.oriImgRelY = 0.04
        self.patchImgRelX = 0.6
        self.patchImgRelY = 0.04
        self.outImgRelX   = 0.1
        self.outImgRelY   = 0.6
        self.attnImgRelX  = 0.6
        self.attnImgRelY  = 0.6
        self.ButtonRelX   = 0.3
        self.ButtonRelY   = 0.4
        self.ButtonXdiff  = 100
        self.LabelText    = 20

        self.patchHNumber = imgSize // patchSize
        self.patchWNumber = imgSize // patchSize
        self.patchIndex = np.array([np.array([i, j]) for i in range(self.patchHNumber) for j in range(self.patchWNumber)])
        self.indexMap  = np.ones((imgSize, imgSize, 3))
        self.indexLog  = []

        self.trainer = trainer
        self.makeMenubar()
        self.makeButton()
        self.rootWin.config(menu=self.menubar)
        
    def makeMenubar(self):
        self.menubar = Menu(self.rootWin)
        fileMenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="FILE", menu=fileMenu)
        # filemenu.add_command(label="New", command=domenu)
        # filemenu.add_command(label="Open", command=domenu)
        fileMenu.add_command(label="Open Image", command=self.Load)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.rootWin.quit)

        helpMenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="HELP", menu=helpMenu)
        helpMenu.add_command(label="About...", command=self.HelpMenu)
    
    def makeButton(self):
        runButton = Button(self.rootWin, text="RUN", overrelief="solid", width=10, command=self.runButton, repeatdelay=1000, repeatinterval=100)
        runButton.place(relx=self.ButtonRelX, rely=self.ButtonRelY)

        run128Button = Button(self.rootWin, text="RUN128", overrelief="solid", width=10, command=self.run128Button, repeatdelay=1000, repeatinterval=100)
        run128Button.place(relx=self.ButtonRelX, rely=self.ButtonRelY, x=self.ButtonXdiff)

        initButton = Button(self.rootWin, text="INITIALIZE", overrelief="solid", width=10, command=self.runInit, repeatdelay=1000, repeatinterval=100)
        initButton.place(relx=self.ButtonRelX, rely=self.ButtonRelY, x=self.ButtonXdiff * 2)

        
    
    def runInit(self):
        self.convert_to_tkimage(self.filename)
        self.convert_to_patch_tkimage()
        self.labelPatch.configure(image=self.photolineImg)
        self.labelPatch.image = self.photolineImg

        self.__init__(rootWin=self.rootWin, patchSize=self.patchSize, imgSize=self.imgSize, trainer=self.trainer, visualPath=self.visualPath, mix=self.mix)
        

    
    def Load(self):
        self.filename = FileDialog.askopenfilename(initialdir=os.path.abspath(__file__), title="Select Load file",
                                            filetypes=(("Images Files(.jpg, .jpeg, .png)", ["*.png", "*.jpg", "*.jpeg"]),
                                            ("All Files", "*.*")))
        try:
            self.imgtk = self.convert_to_tkimage(self.filename)
            labelOriText = Label(self.rootWin, text="Original Image")
            labelOri = Label(self.rootWin, image=self.imgtk)
            labelOri.place(relx=self.oriImgRelX, rely=self.oriImgRelY)
            labelOriText.place(relx=self.oriImgRelX, rely=self.oriImgRelY, y = -self.LabelText)

            self.convert_to_patch_tkimage()
            labelPatchText = Label(self.rootWin, text="Click to Image Drop Here")
            self.labelPatch = Label(self.rootWin, image=self.photolineImg)
            self.labelPatch.place(relx = self.patchImgRelX, rely = self.patchImgRelY)
            labelPatchText.place(relx=self.patchImgRelX, rely=self.patchImgRelY, y=-self.LabelText)
            self.labelPatch.bind('<Button-1>', self.makePatch)

        except:
            pass

    def makePatch(self, event):
        mouseX= event.x
        mouseY= event.y
        index = np.array([mouseY//self.patchSize, mouseX//self.patchSize])
        self.indexLog.append(tuple(index))

        self.dropIndex(index)
    
    def dropIndex(self, index):
        pos = index * self.patchSize
        self.indexMap[pos[0]:pos[0] + self.patchSize, pos[1]:pos[1] + self.patchSize] = 0
        self.lineImg[pos[0]:pos[0] + self.patchSize, pos[1]:pos[1] + self.patchSize] = 0
        
        lineImg = PILImage.fromarray(self.lineImg)
        photolineImg = ImageTk.PhotoImage(image=lineImg)
        self.labelPatch.configure(image=photolineImg)
        self.labelPatch.image = photolineImg
    
    def HelpMenu(self):
        MessageBox.showinfo("Help", "1. Load Your Image's click FILE Menu and Load Image \n\n2. Make Patch you want. \n\n3. Finally Click Run button For running")

    def convert_to_tkimage(self, path):
        src = cv2.imread(path)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.src = cv2.resize(src, dsize=(self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA)
        src = PILImage.fromarray(self.src)
        imgtk = ImageTk.PhotoImage(image=src)
        return imgtk
    
    def convert_to_patch_tkimage(self):
        h, w, _ = self.src.shape
        patchNumber = h//self.patchSize
        self.patchIndex = np.array([np.array([i, j]) for i in range(patchNumber) for j in range(patchNumber)])

        self.lineImg = self.src.copy()
        for index in range(patchNumber+1):    
            startValue = index * self.patchSize
            #draw Width
            cv2.line(self.lineImg, (0, startValue), (w, startValue), (255,0,0), 3)
            #draw Height
            cv2.line(self.lineImg, (startValue, 0), (startValue, h), (255,0,0), 3)

        lineImg = PILImage.fromarray(self.lineImg)
        self.photolineImg = ImageTk.PhotoImage(image=lineImg)
    

    def runButton(self):
        normImg = Normalization(self.src/255, 0.5, 0.5)
        dropImg = DropPatch(normImg, self.indexMap).transpose((2, 0, 1))[np.newaxis,:,:,:]

        inputs = torch.from_numpy(dropImg.astype(np.float32)).cuda(0)
        if self.mix:
            index = torch.from_numpy(1-self.indexMap.transpose((2, 0, 1))[np.newaxis,:,:,:]).cuda(0)
        else:
            index = None

        self.trainer.gan.model.eval()
        with torch.no_grad():
            outputs, attn = self.trainer.gan.outStep(inputs, index)


        inputs  = self.trainer.gan.makeVisual(inputs)
        outputs = self.trainer.gan.makeVisual(outputs)

        plt.imsave(os.path.join(self.visualPath, "Outputs.jpg"), outputs)
        plt.imsave(os.path.join(self.visualPath, "inputs.jpg"), inputs)
        plt.imsave(os.path.join(self.visualPath, "mask.jpg"), 1 - self.indexMap)

        for h, head in enumerate(attn[list(attn.keys())[-1]].to("cpu").detach().numpy()[0]):
            self.saveAttnetion("Attention", h, head )


        outImg = self.convert_to_tkimage(os.path.join(self.visualPath, "Outputs.jpg"))
        labelOutText = Label(self.rootWin, text=f"Epoch {self.trainer.startEpoch} Drop Patch {len(set(self.indexLog))} Output Image")
        labelOut = Label(self.rootWin, image=outImg)
        labelOut.place(relx=self.outImgRelX, rely=self.outImgRelY)
        labelOut.configure(image=outImg)
        labelOut.image = outImg
        labelOutText .place(relx=self.outImgRelX, rely=self.outImgRelY, y =-self.LabelText)

        attnImg = self.convert_to_tkimage(os.path.join(self.visualPath, f"Attention_Head_{h+1}.jpg"))
        labelAttnText = Label(self.rootWin, text=f"Epoch {self.trainer.startEpoch} Drop Patch {len(set(self.indexLog))} Attention Head 8 Image")
        labelAttn = Label(self.rootWin, image=attnImg)
        labelAttn.place(relx=self.attnImgRelX, rely=self.attnImgRelY)
        labelAttn.configure(image=attnImg)
        labelAttn.image = attnImg
        labelAttnText .place(relx=self.attnImgRelX, rely=self.attnImgRelY, y =-self.LabelText, x=-10)
    
    def run128Button(self):
        nowSelected = len(set(self.indexLog))

        if nowSelected < 128:
            newIndex = np.random.choice(len(self.patchIndex), size= 128-nowSelected)
            for i in newIndex:
                index = self.patchIndex[i]
                self.indexLog.append(tuple(index))
                index *= self.patchSize
                self.indexMap[index[0]:index[0] + self.patchSize, index[1]:index[1] + self.patchSize] = 0
                
            self.run128Button()
        
        else:
            self.runButton()

    
    def saveAttnetion(self, name, h, attention):
        fig = plt.figure()
        ax = plt.gca()
        attnax = ax.matshow(attention)
        fig.colorbar(attnax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.grid(False)     
        ax.set_xlabel(f'{name} Head {h+1}')
        plt.savefig(os.path.join(self.visualPath, f"{name}_Head_{h+1}.jpg"))
        plt.close()
        

    
