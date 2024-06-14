import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.init as init
import io, os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import wordnet, brown
from collections import Counter


def init_optimizer(params, optim, lr, weight_decay):
    """initialize optimizer.

    Args:
        params (list): list of parameters to optimaze
        optim (str): type of optimizer
        lr (float): learning rate
        weight_decay (float): weight_decay
    """
    if optim =="adam":
        optimizer=torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim =="sgd":
        optimizer=torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    else:
        raise Exception('found unexpected optimizer!')
    return optimizer

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        pass
        # init.normal_(m.weight.data, mean=1, std=0.02)
        # init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        try:
            init.normal_(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        print('skip embedding layer')
    else:
        pass


def loss_plot(folder, file, steps=0):
    
    fig = plt.figure(figsize=(12,6))

    address = os.path.join(folder, file, 'loss_log.json')
    batchs = []
    with open(address, 'r') as f:
        for line in f:
            b = json.loads(line)
            batchs.append(b)
    batchs_loss_pct = []

    for b in batchs:
        size = len(b['0'])
        if len(b['0']) != size:
            continue
        non_zero = np.array(b['0']) > 0.0
        loss_pct = []
        for step in range(steps):
            if step == 0:
                loss_pct.append(np.ones_like(np.array(b[str(step)])[non_zero]))
            else:
                loss_pct.append(np.array(b[str(step)])[non_zero]/np.array(b['0'])[non_zero])
        batchs_loss_pct.append(loss_pct)

    batch_mean_loss = np.zeros((len(batchs_loss_pct), steps))
    for i, b in enumerate(batchs_loss_pct):
        for step in range(steps):
            batch_mean_loss[i, step] = np.mean(b[step])
    plt.plot(batch_mean_loss.mean(0),)
        
    plt.savefig(os.path.join(folder, file, 'loss_plot.jpg'))
          
def custom_norm(tensor, item_num=None):
    """customized norm"""
    norm = torch.norm(tensor, p=1)
    if not item_num:
        shape = tensor.view(1,-1).shape[1]
        return norm/shape
    else:
        return norm/item_num