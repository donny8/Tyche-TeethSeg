__author__ = "Dohyun Kim <donny8.kim@gmail.com>"


import os
import psutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchio as tio 
import torch.nn as tnn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from einops import repeat

def strip(caseid:str):
    return caseid.strip()

def file_checker(path):
    if not os.path.exists(path):
        raise IOError(f'"{path}" file do not exits')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_memory_usage(tag, order=3, pprint=False):
    assert order<4

    pid = os.getpid()
    current_process = psutil.Process(pid)
    usage = current_process.memory_info()[0]/(1024)**order
    if order==0:
        degree = 'Byte'
    elif order==1:
        degree = 'KByte'
    elif order==2:
        degree = 'MByte'  
    elif order==3:
        degree = 'GByte'  

    if pprint:
        print('>>>>>>>> {} : {} {}'.format(tag, usage, degree))

    return usage

def save_loss_plot(target, mode, exp_name, fig_path):
    mkdir(f'{fig_path}/{mode}')

    plt.title(f'{mode}\n{exp_name}')
    plt.plot(target)
    plt.scatter(np.arange(len(target)), target)
    plt.savefig(f'{fig_path}/{mode}/{mode}_{exp_name}.png')
    plt.close()