import argparse
import torch, os
import torch.nn as nn

import collections.abc as container_abcs


os.environ["USE_STATIC_NCCL"] = "1"


parser = argparse.ArgumentParser(description="Patch Painting Train Code",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model_name', type=str, help='Name of model to train, availabels models are [PPformers | PPransGAN]', choices=["PPformers", "PPransGAN"])
parser.add_argument('-c', '--config', default= "TrainParameterExample.json", help="Model's Config", type=str, dest="config")
parser.add_argument("-s", "--shared_path", default=os.path.dirname(os.path.abspath(__file__)), help="Shared File Path For Distributed Learning", dest="shared")
parser.add_argument('-r', "--restart", help='restart', action='store_true', dest="restart")
parser.add_argument('-d', "--DDP", help='run with DDP', action='store_true', dest="ddp")


args = parser.parse_args()

modelName = args.model_name.lower()
configFilePath = args.config
sharedFilePath = args.shared
restart = args.restart
ddp     = args.ddp

def loadTrainer(modelName):
    module = "libs.trainer"
    if modelName == "ppformers":
        exec(f"from {module}.GEL2trainer import PatchPainting_GEL2 as trainer")
    else:
        if not ddp:
            exec(f"from {module}.PPranGANtrainer import PPransGANtrainer as trainer")
        else:
            exec(f"from {module}.PPranGANtrainerDDP import PPransGANtrainer as trainer")
    
    return eval("trainer")

if __name__ =="__main__":
    assert modelName == "ppformers" or modelName == "ppransgan", f"Please Check Model's Name"
    trainer = loadTrainer(modelName)
    Trainer = trainer(configFilePath, sharedFilePath, restart, ddp=ddp)
    Trainer.runTrain()