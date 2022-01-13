import argparse
import torch, os
import torch.nn as nn

from tkinter import *

import collections.abc as container_abcs

from libs.trainer.GEL2trainer import PatchPainting_GEL2
from libs.modules import *

os.environ["USE_STATIC_NCCL"] = "1"

parser = argparse.ArgumentParser(description="Evaluation of Patch Painting Transformer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-v', "--visualization", help='Print more data', action='store_false', dest="visual")
parser.add_argument('-m', "--mix", help='Mixing Original Data', action='store_true', dest="mix")
parser.add_argument('-e', '--epoch', default= "None", help="Load Epochs", type=str, dest="epoch")
parser.add_argument('-c', '--config', default= "TrainParameterExample.json", help="Model's Config", type=str, dest="config")
parser.add_argument("-s", "--shared_path", default=os.path.dirname(os.path.abspath(__file__)), help="Shared File Path For Distributed Learning", dest="shared")
parser.add_argument("-vPath", "--visualization_path", default=os.path.join(".", "visualization"), dest="visPath")

args = parser.parse_args()

vis            = args.visual
mix            = args.mix
epoch          = args.epoch
configFilePath = args.config
sharedFilePath = args.shared
visualPath     = args.visPath

if not os.path.exists(visualPath):
    os.mkdir(visualPath)

if epoch is "None":
    epoch = None
else:
    epoch = int(epoch)


if __name__ =="__main__":
    Trainer = PatchPainting_GEL2(configFilePath, sharedFilePath, startEpoch=epoch)
    Trainer.visual_init(epoch)

    root = Tk()
    print(f"Mixing... {mix}")
    mainWin = tkWindow(root, Trainer.patch_size, Trainer.img_size, Trainer, visualPath, mix=mix)
    root.mainloop()