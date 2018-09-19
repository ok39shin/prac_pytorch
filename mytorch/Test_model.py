# model test by MNIST
# coding : utf-8

import os
import sys

import unittest
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from model import NN_classifier, CNN2d_classifier

### 1. dataload
def make_MNIST_dataset(bs=100, validset=None):
    transform = transforms.Compose(
            [transforms.ToTensor(),
             tranforms.Normalize((0.5, ), (0.5, ))]
            )
    testset = datasets.MNIST(root='./data',
                             train=False,
                             download=True,
                             transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=bs,
                                             shuffle=False,
                                             )

    trainset = datasets.MNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transform)
    if validset:
        trainset, validset = split_validset(trainset, crsval=False) # already shuffled
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=bs,
                                                  # shuffle=False,
                                                  )
        validloader = torch.utils.data.DataLoader(validset,
                                                  batch_size=bs,
                                                  # shuffle=False,
                                                  )
        return trainloader, validloader, testloader
    else:
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=bs,
                                                  shuffle=True,
                                                  )
        return trainloader, testloader

def split_validset(trainset, seed=1, val_siz=0.1, shuf=True, plotsmp=False, crsval=False):
    # it is for dataset does not have validation set
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((val_siz >= 0) and (val_siz <= 1)), error_msg

    num_train = len(trainset)
    indexes = list(range(num_train))
    split = int(np.floor(val_siz * num_train))
    split_ind = list(range(0, num_train, split)) + [None]

    if shuf:
        np.random.seed(seed)
        np.random.shuffle(indexes)

    valid_ind = indexes[split_ind[0]:split_ind[1]]
    train_ind = list(set(indexes) ^ set(valid_ind))
    
    validset_dv = [trainset[ind] for ind in valid_ind]
    trainset_dv = [trainset[ind] for ind in train_ind]
    return validset_dv, trainset_dv

### 2. define
def define(mtype, mp):
    # 2.1 def model
    if mtype == 'NN':
        model = NN_classifier(mp['indim'], mp['nhid'], mp['hiddim'], mp['nclass'])
    elif mtype == 'CNN2d':
        model = CNN2d_classifier(mp['indims'], mp['f_num'], mp['f_siz'], mp['strides'], mp['c_nlay'], mp['c_laydim'], mp['c_out'])
    # 2.2 def loss
    lossf = model.lossf
    # 2.3 def optim
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    return model, lossf, optimizer

### 3. train loop
def train_loop():
    pass

### 4. evaluation
def evaluation(detail=False):
    pass

def main(mtype, mp, bs=100, validset=None):
    # 1. dataload
    if validset:
        train_loader, valid_loader, test_loader = make_MNIST_dataset(bs=bs, validset=validset)
    else:
        train_loader, test_loader = make_MNIST_dataset(bs=bs, validset=validset)
    # 2. define 
    model, lossf, optimizer = define(mtype, mp)
    # 3. train
    train_loop()
    # 4. last evaluation
    evalation(detail=True)

if __name__ == '__main__':
    mtype = 'NN'
    bs = 100
    validset = True
    mp = {}
    # for NN_classifier
    mp['indim'] = 28*28
    mp['nhid'] = 3
    mp['hiddim'] = 2048
    mp['nclass'] = 10
    # for CNN2d_classifier
    mp['indims'] = 28
    mp['f_num'] = [10, 20]
    mp['f_siz'] = [5, 5]
    mp['strides'] = [1, 1]
    mp['c_nlay'] = 1
    mp['c_laydim'] = 256
    mp['c_out'] = 10
    main(mtype, mp, bs=bs, validset=validset)
