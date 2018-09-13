# NN models
# coding : utf-8

import os
import sys

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, indim, hiddims, outdim, mtype='NN', sigmoid=False):
        # sigmoid = True -> loss = BCELoss
        super(NN, self).__init__()
        self.mtype = mtype
        self.indim = indim
        self.hiddims = hiddims
        self.outdim = outdim
        self.sigmoid = sigmoid
        
        # set layers
        self.layers = nn.Sequential()
        dims = [indim] + hiddims + [outdim]
        for i in range(len(dims)-1):
            self.layers.add_module('Linear {}'.format(i+1), nn.Linear(dims[i], dims[i+1]))
            self.layers.add_module('activate {}'.format(i+1), nn.ReLU())
        if sigmoid:
            self.layers.add_module('Sigmoid', nn.Sigmoid())
    
    def forward(self, x):
        x = self.layers(x)
        return x

class NN_classifier(NN):
    def __init__(self, indim, hiddims, outdim):
        super().__init__(indim, hiddims, outdim, mtype='NN_classifier', sigmoid=False)
        self.layers.add_module('Softmax', nn.Softmax())

class CNN2d(nn.Module):
    def __init__(self, indims, f_num, f_siz, strides, pads, mtype='CNN2d', first_fnum=1, pool=None, sigmoid=False):
        '''
            indims  : input dim 2d, if int dim is square (int | tuple)
            f_num   : flame num ([int, int, ... int])
            f_siz   : flame size ([int, int, ... int])
            strides : stride ([int, int, ... int] | [tuple, tuple, ... tuple])
            pads    : padding list ([int, int, ... int] | [tuple, tuple, ... tuple])

        option:
            mtype   : model type. default is "CNN2d"
            first_fnum  : first flame num. default is 1
            sigmoid : flag of sigmoid at last layer
        '''
        super(CNN2d, self).__init__()
        self.mtype = mtype
        if type(indims) == int:
            self.indims = (indims, indims)
        elif type(indims) == tuple:
            self.indims = indims
        f_num = [first_fnum] + f_num
        self.f_num = f_num # list [int, int, ... int]
        self.f_siz = f_siz # list [int, int, ... int]
        self.strides = strides # list [int, int, ... int] or list [tuple, tuple, ... tuple]
        self.pads = pads
        self.activate = nn.LeakyReLU()
        self.pool = pool

        # set layers
        self.layers = nn.Sequential()
        ## conv 2d
        for i in range(len(f_siz)):
            self.layers.add_module('Conv2d {}'.format(i), nn.Conv2d(f_num[i], f_num[i+1], f_siz[i], stride=strides[i], padding=pads[i]))
            if self.pool:
                self.layers.add_module('pooling {}'.format(i), self.pool(2))
            self.layers.add_module('activate {}'.format(i), self.activate)
        if sigmoid:
            self.layers.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        return x

if __name__ == '__main__':
    indim, hiddims, outdim = 32, [256, 256], 10
    sigmoid = True
    net = NN(indim, hiddims, outdim, sigmoid)
    net = NN_classifier(indim, hiddims, outdim)
    indims, f_num, f_siz, strides, pads = 28, [32, 64, 128], [9, 7, 5], [2, 2, 2], [4, 3, 2]
    net = CNN2d(indims, f_num, f_siz, strides, pads, sigmoid=sigmoid)
    print(net)
