import os
import socket

import torch
import torch.nn


from contextlib import closing
import torch.distributed as dist


def init_process(rank, size, backend="nccl"):
    print(f'Setting UP Local GPU :{rank} |  World Size : {size} | Backend : {backend}')
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = find_free_port()
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    print(f"127.0.0.1 {find_free_port()}")
    dist.init_process_group(backend,init_method='env://')
    print(f"{rank} init complete")

def find_free_port():

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
        

def cleanup():
    dist.destroy_process_group()

def save_gen(ckpt_dir, netG, optimG, epoch, model_name="PatchPainting"):
    r"""
    Model Saver

    Inputs:
        ckpt_dir   : (string) check point directory
        netG       : (nn.module) Generator Network
        opitmG     : (torch.optim) Generator's Optimizers
        epoch      : (int) Now Epoch
        model_name : (string) Saving model file's name
    """
    if hasattr(netG, "module"):
        netG_dicts = netG.module.state_dict()
        try:
            optimG_dicts = optimG.module.state_dict()
        except:
            optimG_dicts = optimG.state_dict()
    else:
        netG_dicts = netG.state_dict()
        optimG_dicts = optimG.state_dict()

    torch.save({"netG": netG_dicts,
                "optimG" : optimG_dicts},
                os.path.join(ckpt_dir, model_name, f"{model_name}_{epoch}.pth"))

def load_gen(ckpt_dir,  netG,  optimG, name, epoch=None, gpu=None):
    r"""
    Model Lodaer

    Inputs:
        ckpt_dir : (string) check point directory
        netG     : (nn.module) Generator Network
        opitmG   : (torch.optim) Generator's Optimizers
        step     : (int) find step.  if NOne, last scale

    """
    ckpt_lst = os.listdir(ckpt_dir)

    if epoch is not None:
        ckptFile = os.path.join(ckpt_dir, name+f"_{epoch}.pth")
    else:
        ckpt_lst.sort()
        ckptFile = os.path.join(ckpt_dir, ckpt_lst[-1])

    if not os.path.exists(ckptFile):
        raise ValueError(f"Please Check Check Point File Path or Epoch, File is not exists!")

    # Load Epochs Now
    epoch = int(ckpt_lst[-1].split("_")[-1][:-4])

    # Load Model 
    if gpu is not None:
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(ckptFile, map_location=mapLocation)
    else:
        dict_model = torch.load(ckptFile)
    
    try:
        netG.load_state_dict(dict_model['netG'])
    except:
        netG.module.load_state_dict(dict_model['netG'])

    optimG.load_state_dict(dict_model["optimG"])

    return netG,  optimG, epoch



def save(ckpt_dir, netG, netD, optimG, optimD, step, model_name="PatchPainting"):
    r"""
    Model Saver

    Inputs:
        ckpt_dir   : (string) check point directory
        netG       : (nn.module) Generator Network
        netD       : (nn.module) Discriminator Network
        opitmG     : (torch.optim) Generator's Optimizers
        optimD     : (torch.optim) Discriminator's  Optimizers
        step       : (int) Now Step
        model_name : (string) Saving model file's name
    """

    if hasattr(netG, "module"):
        netG_dicts = netG.module.state_dict()
        netD_dicts = netD.module.state_dict()
        try:
            optimG_dicts = optimG.module.state_dict()
            optimD_dicts = optimD.module.state_dict()
        except:
            optimG_dicts = optimG.state_dict()
            optimD_dicts = optimD.state_dict()
    else:
        netG_dicts = netG.module.state_dict()
        netD_dicts = netD.module.state_dict()
        optimG_dicts = optimG.state_dict()
        optimD_dicts = optimD.state_dict()


    torch.save({"netG": netG_dicts,
                "netD": netD_dicts,
                "optimG" : optimG_dicts,
                "optimD" : optimD_dicts},
                os.path.join(ckpt_dir, model_name, f"{model_name}_{step}.pth"))

def load(ckpt_dir, netG, netD, optimG, optimD,  step=None):
    r"""
    Model Lodaer

    Inputs:
        ckpt_dir : (string) check point directory
        netG     : (nn.module) Generator Network
        netD     : (nn.module) Discriminator Network
        opitmG   : (torch.optim) Generator's Optimizers
        optimD   : (torch.optim) Discriminator's  Optimizers
        step     : (int) find step.  if NOne, last scale
    """
    
    ckpt_lst = None

    if step is not None:
        if ckpt_lst is not None:
            ckpt_lst = [i for i in ckpt_lst if i.split("_")[-1][0] == step]
        else:
            ckpt_lst = [i for i in os.listdir(ckpt_dir) if int(i.split("_")[-1][:-4]) == step]
    
    if ckpt_lst is None:
        ckpt_lst = os.listdir(ckpt_dir)
    print(ckpt_lst)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


    # Load Step
    step = int(ckpt_lst[-1].split("_")[-1][:-4])

    # Load Model
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))
    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model["optimG"])
    optimD.load_state_dict(dict_model["optimD"])
    
    return netG, netD, optimG, optimD,  step