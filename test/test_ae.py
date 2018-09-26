#/usr/bin/env python
import torch
from mytorch.ae import AE, VAE, CAE
import copy
import pytest

@pytest.mark.parametrize(
        ("indim, hiddims, zdim, activ, bs, sigmoid"),
        [
            (28, [14], 10, torch.nn.LeakyReLU(), 10, True),
            (28, [14], 10, None, 10, True),
            (28, [14], 10, torch.nn.LeakyReLU(), 10, False),
            (28, [14, 7], 10, torch.nn.LeakyReLU(), 10, True),
            (28, [], 10, torch.nn.LeakyReLU(), 10, True),
            ]
)
def t_AE(indim, hiddims, zdim, activ, bs, sigmoid):
    model = AE(indim, hiddims, zdim, activ=activ, sigmoid=sigmoid)
    x = torch.randn(bs, indim)
    y = model(x)
    assert x.shape == y.shape
    print(model)

@pytest.mark.parametrize(
        ("indim, hiddims, zdim, activ, bs, sigmoid"),
        [
            (28, [14], 10, torch.nn.LeakyReLU(), 10, True),
            # (28, [14], 10, None, 10, True),
            (28, [14], 10, torch.nn.LeakyReLU(), 10, False),
            (28, [14, 7], 10, torch.nn.LeakyReLU(), 10, True),
            # (28, [], 10, torch.nn.LeakyReLU(), 10, True),
            ]
)
def t_VAE(indim, hiddims, zdim, activ, bs, sigmoid):
    model = VAE(indim, hiddims, zdim, activ=activ, sigmoid=sigmoid)
    x = torch.randn(bs, indim)
    mu, var = model.encoder(x)
    assert mu.shape == var.shape == (bs, zdim)
    z_smp = model.sample_latent(mu, var)
    assert z_smp.shape == (bs, zdim)
    y = model.decoder(z_smp)
    assert x.shape == y.shape
    print(model)

@pytest.mark.parametrize(
        ("indims, f_nums, f_sizs, strides, bs, first_fnum, last_fnum"),
        [
            (28, [16, 32, 64], [9, 7, 5], [2, 2, 2], 10, 1, 1),
            (28, [16, 32, 64], [9, 7, 5], [3, 3, 3], 10, 1, 1),
            (28, [16, 32, 64], [8, 6, 4], [2, 2, 2], 10, 1, 1),
            (28, [], [], [], 10, 1, 1),
            (28, [16, 32, 64], [9, 7, 5], [2, 2, 2], 10, 1, 10),
            ]
)
def t_CAE(indims, f_nums, f_sizs, strides, bs, first_fnum, last_fnum):
    model = CAE(indims, f_nums, f_sizs, strides, first_fnum=first_fnum)
    print(model.encoder.ndims)
    print(model.decoder.ndims)
    if type(indims) == int:
        x = torch.randn(bs, first_fnum, indims, indims)
    elif type(indims) == tuple:
        x = torch.randn(bs, first_fnum, indims[0], indims[1])
    # encoder
    tmp = copy.copy(x)
    for i in range(model.encoder._nlay):
        assert x.shape == (bs, model.encoder.f_nums[i], model.encoder.ndims[i][0], model.encoder.ndims[i][1])
        x = model.encoder.layers[i](x)
    assert x.shape == (bs, model.encoder.f_nums[-1], model.encoder.ndims[-1][0], model.encoder.ndims[-1][1])
    x = tmp
    # decoder
    z = model.encoder(x, view=False)
    for i in range(model.decoder._nlay):
        assert z.shape == (bs, model.decoder.f_nums[i], model.decoder.ndims[i][0], model.decoder.ndims[i][1])
        z = model.decoder.layers[i](z)
        z = model.decoder._slice_tensor(z, i)
    assert z.shape == (bs, model.decoder.f_nums[-1], model.decoder.ndims[-1][0], model.decoder.ndims[-1][1])
    # all process
    y = model(x)
    assert y.shape == x.shape
