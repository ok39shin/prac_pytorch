#/usr/bin/env python
import torch
from model import NN, NN_classifier, CNN2d, CNN2d_classifier

def t_NN():
    indim = 28*28
    hiddims = [2048, 1024, 512]
    outdim = 10
    bs = 10
    mdl = NN(indim, hiddims, outdim)
    x = torch.randn(bs, indim)
    y = mdl(x)
    assert y.shape == (bs, outdim), 'shape is not equal'

def t_CNN2d():
    indims = 28
    f_num = [10, 20, 30]
    f_siz = [9, 7, 5]
    strides = [2, 2, 2]
    first_fnum = 1
    bs = 10
    mdl = CNN2d(indims, f_num, f_siz, strides, first_fnum=first_fnum)
    x = torch.randn(bs, first_fnum, mdl.indims[0], mdl.indims[1])
    y = mdl(x)
    assert y.shape == (bs, mdl.ndims[-1][0]*mdl.ndims[-1][1]*f_num[-1]), \
           'shape is not equal: y.shape={0},estm_out={1}'.format(y.shape, (bs, mdl.ndims[-1][0]*mdl.ndims[-1][1]*f_num[-1]))

def t_NN_classifier():
    indim = 28*28
    nhid = 3
    hiddim = 256
    nclass = 10
    bs = 10
    mdl = NN_classifier(indim, nhid, hiddim, nclass)
    x = torch.randn(bs, indim)
    y = mdl(x)
    assert y.shape == (bs, nclass)

def t_CNN2d_classifier():
    indims = 28
    f_nums = [10, 20, 30]
    f_sizs = [9, 7, 5]
    strides = [2, 2, 2]
    c_nlay = 3
    c_laydim = 256
    c_out = 10
    bs = 10
    mdl = CNN2d_classifier(indims, f_nums, f_sizs, strides, c_nlay, c_laydim, c_out)
    x = torch.randn(bs, 1, mdl.indims[0], mdl.indims[1])
    y_CNN = mdl.CNNlayers(x)
    assert y_CNN.shape == (bs, f_nums[-1], mdl.ndims[-1][0], mdl.ndims[-1][1])
    y_CLS = mdl.CLSlayers(y_CNN.view(bs, -1))
    assert y_CLS.shape == (bs, c_out)
    y = mdl(x)
    assert y.shape == (bs, c_out)
