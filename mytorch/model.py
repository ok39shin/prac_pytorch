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
    def __init__(self, indim, hiddims, outdim, sigmoid=False):
        # sigmoid = True -> loss = BCELoss
        super(NN, self).__init__()
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
        super(NN_classifier, self).__init__(indim, hiddims, nclass, sigmoid=False)
        self.mtype = 'NN_classifier'
        self.nhid = nhid
        self.hiddim = hiddim
        self.nclass = nclass
        self.layers.add_module('LogSoftmax', nn.LogSoftmax(dim=0))

        self.lossf = nn.NLLLoss()

class CNN2d(nn.Module):
    def __init__(self, indims, f_nums, f_sizs, strides, first_fnum=1, activ=nn.LeakyReLU(), pool=None, sigmoid=False):
        '''
            indims  : input dim 2d, if int dim is square (int | tuple)
            f_nums  : flame num ([int, int, ... int])
            f_sizs  : flame size ([int, int, ... int])
            strides : stride ([int, int, ... int] | [tuple, tuple, ... tuple])

        option:
            first_fnum  : first flame num. default is 1
            activ   : activation layer
            pool    : pooling layer
            sigmoid : flag of sigmoid at last layer
        '''
        super(CNN2d, self).__init__()
        self.mtype = 'CNN2d'
        self.indims = self._int2tuple(indims)
        f_nums = [first_fnum] + f_nums
        self.f_nums = f_nums # list [int, int, ... int]
        self.f_sizs = [self._int2tuple(f_siz) for f_siz in f_sizs] # list [tuple, tuple, ... tuple]
        self.strides = [self._int2tuple(stride) for stride in strides] # list [tuple, tuple, ... tuple]
        self._nlay = len(f_sizs)
        self.pads = self._get_pads() # list [tuple, tuple, ... tuple]
        self.ndims = self._get_ndims()
        self.layers = self._set_layers()
        self.activ = activ
        self.pool = pool
        self.sigmoid = sigmoid

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

    def _set_layers(self):
        layers = nn.Sequential()
        ## conv 2d
        for i in range(self._nlay):
            layers.add_module('Conv2d {}'.format(i),
                              nn.Conv2d(self.f_nums[i],
                                        self.f_nums[i+1],
                                        self.f_sizs[i],
                                        stride=self.strides[i],
                                        padding=self.pads[i]
                                        )
                             )
        return layers

    def forward(self, x, view=True):
        for i in range(self._nlay):
            x = self.layers[i](x)
            x = self.activ(x)
            if self.pool:
                x = self.pool(x)
        if view:
            x = x.view(x.shape[0], -1)
        if self.sigmoid:
            x = F.sigmoid(x)
        return x

class CNNtrans2d(CNN2d):
    def __init__(self, ndims, f_nums, f_sizs, strides, last_fnum=1, activ=nn.LeakyReLU(), sigmoid=False):
        super(CNN2d, self).__init__()
        self.ndims = self._int2tuple(ndims)
        self.f_nums = f_nums + [last_fnum]
        self.f_sizs = self._int2tuple(f_sizs)
        self.strides = self._int2tuple(strides)
        self._nlay = len(f_sizs)
        self.slice_ids = self._get_slice_ids()
        self.layers = self._set_layers()
        self.activ = activ
        self.sigmoid = sigmoid

    def _get_slice_ids(self):
        slice_ids = [tuple([self.__cal_sliceid(i,j) for j in range(2)]) for i in range(self._nlay)]
        return slice_ids

    def __cal_sliceid(self, i, j):
        # (start ind, size)
        return (self.f_sizs[i]//2 , self.ndims[i+1][j])

    def _set_layers(self):
        layers = nn.Sequential()
        ## conv 2d
        for i in range(self._nlay):
            layers.add_module('ConvTranspose2d {}'.format(i),
                              nn.ConvTranspose2d(self.f_nums[i],
                                                 self.f_nums[i+1],
                                                 self.f_sizs[i],
                                                 stride=self.strides[i],
                                                 )
                             )
        return layers

    def _slice_tensor(self, x, i):
        return x.narrow(2, self.slice_ids[i][0][0], self.slice_ids[i][0][1]).narrow(3, self.slice_ids[i][1][0], self.slice_ids[i][1][1])

    def forward(self, x, view=True):
        if view:
            x = x.view(x.shape[0], self.f_nums[0], self.ndims[0][0], self.ndims[0][1])
        for i in range(self._nlay):
            x = self.layers[i](x)
            x = self._slice_tensor(x, i)
            x = self.activ(x)
        if self.sigmoid:
            x = F.sigmoid(x)
        return x

class CNN2d_classifier(nn.Module):
    def __init__(self, indims, f_nums, f_sizs, strides, c_nlay, c_laydim, c_out, first_fnum=1, pool=None):
        super(CNN2d_classifier, self).__init__()
        self.CNN = CNN2d(indims, f_nums, f_sizs, strides, first_fnum=first_fnum, pool=pool, sigmoid=False)

        c_indim = self.CNN.f_nums[-1]
        for dim in self.CNN.ndims[-1]:
            c_indim *= dim
        if c_nlay > 0:
            self.CLS = NN_classifier(c_indim, c_nlay, c_laydim, c_out)
        else:
            self.CLS = self._passf()
        self.mtype = 'CNN2d_classifier'

    def forward(self, x):
        x = self.CLS(self.CNN(x, view=True))
        return x

    def _passf(self):
        pass
