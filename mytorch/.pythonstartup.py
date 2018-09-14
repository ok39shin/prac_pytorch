import numpy as np
import torch
import torch.nn as nn
from model import CNN2d, CNN2d_classifier
indims = 28
f_num = [32, 64, 128]
f_siz = [(9, 5), 7, 5]
strides = [3, 2, 2]
c_nlay = 2
c_dimlay = 256
c_outdim = 10

x = torch.randn(5,1,28,28)
model = CNN2d_classifier(indims, f_num, f_siz, strides, c_nlay, c_dimlay, c_outdim)
x1 = model.layers[:2](x)
x2 = model.layers[:4](x)
x3 = model.layers[:6](x)
