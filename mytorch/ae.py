# AE models
# coding : utf-8

import os
import sys
import copy

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CNN2d, CNNtrans2d

### Auto Encoder ###
class AE_encoder(nn.Module):
    def __init__(self, indim, hiddims, zdim, activ=nn.LeakyReLU()):
        super(AE_encoder, self).__init__()
        self.indim = indim
        self.hiddims = hiddims
        self.zdim = zdim
        self.activ = activ
        self.layers = self._set_layers()

    def _set_layers(self):
        layers = nn.Sequential()
        dims = [self.indim] + self.hiddims + [self.zdim]

        for i in range(len(dims[:-1])):
            layers.add_module('encoder Linear {}'.format(i),
                                   nn.Linear(dims[i], dims[i+1]))
            if i != range(len(dims[:-1]))[-1]:
                layers.add_module('activate {}'.format(i),
                                   self.activ)
        return layers

    def forward(self, x):
        z = self.layers(x)
        return z

class AE_decoder(nn.Module):
    def __init__(self, zdim, hiddims_rv, indim, activ=nn.LeakyReLU(), sigmoid=None):
        super(AE_decoder, self).__init__()
        self.zdim = zdim
        self.hiddims_rv = hiddims_rv
        self.indim = indim
        self.activ = activ
        if sigmoid:
            self.sigmoid = True
        else:
            self.sigmoid = None
        self.layers = self._set_layers()

    def _set_layers(self):
        layers = nn.Sequential()
        dims = [self.zdim] + self.hiddims_rv + [self.indim]

        for i in range(len(dims[:-1])):
            layers.add_module('decoder Linear {}'.format(i),
                              nn.Linear(dims[i], dims[i+1]))
            if i != range(len(dims[:-1]))[-1]:
                layers.add_module('activate {}'.format(i),
                                  self.activ)
            else:
                if self.sigmoid:
                    layers.add_module('sigmoid', nn.Sigmoid())
        return layers

    def forward(self, z):
        x = self.layers(z)
        return x

class AE(nn.Module):
    def __init__(self, indim, hiddims, zdim, activ=nn.LeakyReLU(), sigmoid=None):
        super(AE, self).__init__()
        self.mtype = 'AE'
        self.encoder = AE_encoder(indim, hiddims, zdim, activ=activ)
        hiddims_rv = [hiddims[-i-1] for i in range(len(hiddims))]
        self.decoder = AE_decoder(zdim, hiddims_rv, indim, activ=activ, sigmoid=sigmoid)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

### Variational Auto Encoder ###
class VAE_encoder(AE_encoder):
    def __init__(self, indim, hiddims, zdim, activ=nn.LeakyReLU()):
        assert len(hiddims) > 0
        hiddim_last = hiddims[-1]
        hiddims_mid = hiddims[:-1]
        super(VAE_encoder, self).__init__(indim, hiddims_mid, hiddim_last, activ=activ)
        self.lay_mu = nn.Linear(hiddim_last, zdim)
        self.lay_var = nn.Linear(hiddim_last, zdim)
        self.hiddims = hiddims
        self.zdim = zdim

    def forward(self, x):
        h_enc = self.activ(self.layers(x))
        mu, logvar = self.lay_mu(h_enc), self.lay_var(h_enc)
        return mu, logvar

class VAE_decoder(AE_decoder):
    def __init__(self, zdim, hiddims_rv, indim, activ=nn.LeakyReLU(), sigmoid=None):
        super(VAE_decoder, self).__init__(zdim, hiddims_rv, indim, activ=activ, sigmoid=sigmoid)

class VAE(nn.Module):
    def __init__(self, indim, hiddims, zdim, activ=nn.LeakyReLU(), sigmoid=None):
        super(VAE, self).__init__()
        self.mtype = 'VAE'
        self.encoder = VAE_encoder(indim, hiddims, zdim, activ=activ)
        hiddims_rv = [hiddims[-i-1] for i in range(len(hiddims))]
        self.decoder = VAE_decoder(zdim, hiddims_rv, indim, activ=activ, sigmoid=sigmoid)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z_smp = self.sample_latent(mu, logvar)
        y = self.decoder(z_smp)
        return y

    def sample_latent(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(std.size()).normal_()
            return eps.mul(std).add(mu)
        else:
            return mu

### Convolutional Auto Encoder
class CAE_encoder(CNN2d):
    def __init__(self, indims, f_nums, f_sizs, strides, first_fnum=1, activ=nn.LeakyReLU(), pool=None, sigmoid=False):
        super(CAE_encoder, self).__init__(indims, f_nums, f_sizs, strides, first_fnum=first_fnum, activ=activ, pool=pool, sigmoid=sigmoid)

class CAE_decoder(CNNtrans2d):
    def __init__(self, ndims_rv, f_nums_rv, f_sizs_rv, strides_rv, last_fnum=1, activ=nn.LeakyReLU()):
        super(CAE_decoder, self).__init__(ndims_rv, f_nums_rv, f_sizs_rv, strides_rv, last_fnum=last_fnum, activ=activ)

class CAE(nn.Module):
    def __init__(self, indims, f_nums, f_sizs, strides, first_fnum=1, last_fnum=1, activ=nn.LeakyReLU()):
        super(CAE, self).__init__()
        self.mtype = 'CAE'
        self.encoder = CAE_encoder(indims, f_nums, f_sizs, strides, first_fnum=first_fnum)
        ndims_rv = self._rv_list(self.encoder.ndims)
        f_nums_rv = self._rv_list(f_nums)
        f_sizs_rv = self._rv_list(f_sizs)
        strides_rv = self._rv_list(strides)
        self.decoder = CAE_decoder(ndims_rv, f_nums_rv, f_sizs_rv, strides_rv, last_fnum=last_fnum)

    def _rv_list(self, li):
        return [li[-i-1] for i in range(len(li))]

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
