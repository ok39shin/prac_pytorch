#/usr/bin/env python
import torch
from mytorch.ae import AE, VAE
# import pytest

def test_AE():
    indim = 28*28
    hiddims = [512, 256, 128]
    zdim = 10
    activ = torch.nn.LeakyReLU()
    sigmoid = True
    model = AE(indim, hiddims, zdim, activ=activ, sigmoid=sigmoid)
    bs = 10
    x = torch.randn(bs, indim)
    y = model(x)
    assert x.shape == y.shape

# @pytest.mark.parametrize(
#         ("indim","hiddims","zdim","activ","bs","sigmoid"),
#         [
#             (28, [14], 10, torch.nn.LeakyReLU(), 10, True),
#             (256, [128, 64], 32, None, 1, True),
#             (256, [], 32, None, 1, True),
#             (256, [128, 64], 32, None, 1, False)
#             ]
# )
def test_VAE():
    indim = 28*28
    hiddims = [512, 256, 128]
    zdim = 10
    activ = torch.nn.LeakyReLU()
    sigmoid = True
    model = VAE(indim, hiddims, zdim, activ=activ, sigmoid=sigmoid)
    bs = 10
    x = torch.randn(bs, indim)
    mu, var = model.encoder(x)
    assert mu.shape == var.shape == (bs, zdim)
    z_smp = model.sample_latent(mu, var)
    assert z_smp.shape == (bs, zdim)
    y = model.decoder(z_smp)
    assert x.shape == y.shape
