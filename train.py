import argparse
import torch, os
import torch.nn as nn

import collections.abc as container_abcs

from libs.trainer.GEL2trainer import PatchPainting_GEL2


os.environ["USE_STATIC_NCCL"] = "1"


parser = argparse.ArgumentParser(description="Evaluation of Patch Painting Transformer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config', default= "TrainParameterExample.json", help="Model's Config", type=str, dest="config")
parser.add_argument("-s", "--shared_path", default=os.path.dirname(os.path.abspath(__file__)), help="Shared File Path For Distributed Learning", dest="shared")

args = parser.parse_args()

configFilePath = args.config
sharedFilePath = args.shared

if __name__ =="__main__":
    Trainer = PatchPainting_GEL2(configFilePath, sharedFilePath)
    Trainer.runTrain()