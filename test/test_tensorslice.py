# test slice tensor
# coding : utf-8

import torch
import pytest

class slice_index(object):
    def calc(x, slices):
        return x[:, :, slices[0][0]:slices[0][1], slices[1][0]:slices[1][1]]
class slice_narrow(object):
    def calc(x, slices):
        return x.narrow(2, slices[0][0], slices[0][1]).narrow(3, slices[1][0], slices[1][1])

@pytest.mark.parametrize(
        ("method, slices"),
        [
            (slice_index, [(1, 27), (2, 26)]),
            (slice_narrow, [(1, 26), (2, 24)]),
            ]
)
def t_timeofslice(method, slices):
    x = torch.randn(5, 1, 28, 28)
    # x = x.cuda()
    for i in range(1000000):
        a = method.calc(x, slices)
    print('shape: ', a.shape)
