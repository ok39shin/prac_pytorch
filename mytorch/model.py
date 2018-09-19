# NN models
# coding : utf-8

import os
import sys
import copy

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, indim, hiddims, outdim, sigmoid=False):
        # sigmoid = True -> loss = BCELoss
        nn.Module.__init__(self)
        self.mtype = 'NN'
        self.indim = indim
        self.hiddims = hiddims
        self.outdim = outdim
        self.sigmoid = sigmoid

        # set layers
        self.layers = nn.Sequential()
        dims = [indim] + hiddims + [outdim]
        for i in range(len(dims)-1):
            self.layers.add_module( \
                    'Linear {}'.format(i+1), \
                    nn.Linear(dims[i], dims[i+1])
                    )
            self.layers.add_module( \
                    'activate {}'.format(i+1), \
                    nn.ReLU()
                    )
        if sigmoid:
            self.layers.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.layers(x)
        return x

class NN_classifier(NN):
    def __init__(self, indim, nhid, hiddim, nclass):
        hiddims = [hiddim] * nhid
        NN.__init__(self, indim, hiddims, nclass, sigmoid=False)
        self.mtype = 'NN_classifier'
        self.nhid = nhid
        self.hiddim = hiddim
        self.nclass = nclass
        self.layers.add_module('LogSoftmax', nn.LogSoftmax(dim=0))

        self.lossf = nn.NLLLoss()

class CNN2d(nn.Module):
    def __init__(self, indims, f_nums, f_sizs, strides, first_fnum=1, pool=None, sigmoid=False):
        '''
            indims  : input dim 2d, if int dim is square (int | tuple)
            f_nums  : flame num ([int, int, ... int])
            f_sizs  : flame size ([int, int, ... int])
            strides : stride ([int, int, ... int] | [tuple, tuple, ... tuple])

        option:
            first_fnum  : first flame num. default is 1
            pool    : pooling layer
            sigmoid : flag of sigmoid at last layer
        '''
        nn.Module.__init__(self)
        # super(CNN2d, self).__init__()
        self.mtype = 'CNN2d'
        self.indims = self._int2tuple(indims)
        f_nums = [first_fnum] + f_nums
        self.f_nums = f_nums # list [int, int, ... int]
        self.f_sizs = [self._int2tuple(f_siz) for f_siz in f_sizs] # list [tuple, tuple, ... tuple]
        self.strides = [self._int2tuple(stride) for stride in strides] # list [tuple, tuple, ... tuple]
        self._nlay = len(f_sizs)
        self.pads = self._get_pads() # list [tuple, tuple, ... tuple]
        self.activate = nn.LeakyReLU()
        self.pool = pool
        self.ndims = self._get_ndims()

        # set layers
        self.layers = nn.Sequential()
        ## conv 2d
        for i in range(self._nlay):
            self.layers.add_module( \
                    'Conv2d {}'.format(i), \
                    nn.Conv2d(self.f_nums[i], \
                              self.f_nums[i+1], \
                              self.f_sizs[i], \
                              stride=self.strides[i], \
                              padding=self.pads[i]
                              )
                    )
            if self.pool:
                self.layers.add_module( \
                        'pooling {}'.format(i), \
                        self.pool(2)
                        )
            self.layers.add_module( \
                    'activate {}'.format(i), \
                    self.activate
                    )
        if sigmoid:
            self.layers.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        return x

    def _int2tuple(self, x):
        return (x, x) if type(x)==int else x

    def _get_pads(self):
        pads = []
        for i in range(self._nlay):
            pads.append(tuple([self.f_sizs[i][j]//2 for j in range(len(self.f_sizs[i]))]))
        return pads

    def _get_ndims(self):
        # https://deepage.net/deep_learning/2016/11/07/convolutional_neural_network.html#convolutional-neural-networkの特徴
        ndims = [self.indims]
        for i in range(self._nlay):
            ndims.append(tuple([(ndims[i][j]+2*self.pads[i][j]-self.f_sizs[i][j])//self.strides[i][j]+1 for j in range(len(ndims[i]))]))
        return ndims

class CNN2d_classifier(CNN2d, NN_classifier):
    def __init__(self, indims, f_num, f_siz, strides, c_nlay, c_laydim, c_out, first_fnum=1, pool=None):
        CNN2d.__init__(self, indims, f_num, f_siz, strides, first_fnum=first_fnum, pool=pool, sigmoid=False)
        CNNlayers = copy.copy(self.layers)
        c_indim = self.f_nums[-1]
        for dim in self.ndims[-1]:
            c_indim *= dim
        if c_nlay > 0:
            NN_classifier.__init__(self, c_indim, c_nlay, c_laydim, c_out)
            CLSlayers = copy.copy(self.layers)
        else:
            CLSlayers = self.passf()
        self.mtype = 'CNN2d_classifier'
        self.CNNlayers = CNNlayers
        self.CLSlayers = CLSlayers

    def forward(self, x):
        x = self.CLSlayers(self.CNNlayers(x).view(-1, self.indim))
        return x

    def passf(self):
        pass

if __name__ == '__main__':
    indim, hiddims, outdim = 32, [256, 256], 10
    sigmoid = True
    net = NN(indim, hiddims, outdim, sigmoid)
    indim, nlay, laydim = 32, 3, 256
    net = NN_classifier(indim, nlay, laydim, outdim)
    indims, f_num, f_siz, strides, pads = 28, [32, 64, 128], [9, 7, 5], [2, 2, 2], [4, 3, 2]
    net = CNN2d(indims, f_num, f_siz, strides, pads, sigmoid=sigmoid)
    print(net)
