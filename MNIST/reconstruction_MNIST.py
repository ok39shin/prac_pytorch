# classify number of MNIST dataset
# coding : utf-8

import os
import sys
import argparse

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import datasets, transforms

from mytorch.ae import AE, CAE, VAE

import pdb

def train(args, model, device, train_loader, optimizer, epoch, lossf=nn.BCELoss(reduction='sum'), view=False):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        if view:
            data = data.view(data.shape[0], -1)
        target = data
        optimizer.zero_grad()
        output = torch.sigmoid(model(data)) if lossf.__class__ == nn.BCELoss else model(data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, lossf=nn.BCELoss(reduction='sum'), view=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            if view:
                data = data.view(data.shape[0], -1)
            target = data
            output = torch.sigmoid(model(data)) if lossf.__class__ == nn.BCELoss else model(data)
            test_loss += lossf(output, target).item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

def parser():
    parse = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # training params
    parse.add_argument('--batch-size', type=int, default=64, metavar='N',
                       help='input batch size for training (default: 64)')
    parse.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                       help='input batch size for testing (default: 1000)')
    parse.add_argument('--epochs', type=int, default=10, metavar='N',
                       help='number of epochs to train (default: 10)')
    parse.add_argument('--lr', type=float, default=0.01, metavar='LR',
                       help='learning rate (default: 0.01)')
    parse.add_argument('--momentum', type=float, default=0.5, metavar='M',
                       help='SGD momentum (default: 0.5)')
    parse.add_argument('--no-cuda', action='store_true', default=False,
                       help='disables CUDA training')
    parse.add_argument('--seed', type=int, default=1, metavar='S',
                       help='random seed (default: 1)')
    parse.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # model params
    parse.add_argument('--mtype', type=str, default='AE',
                       help='kind of model: AE, CAE, VAE')
    # Conv AE
    parse.add_argument('--f_nums', type=list, default=[10, 20],
                       help='frame num list')
    parse.add_argument('--f_sizs', type=list, default=[7, 5],
                       help='frame size list')
    parse.add_argument('--strides', type=list, default=[2, 2],
                       help='stride size list')
    # Linear AE
    parse.add_argument('--hiddims', type=list, default=[512, 256],
                       help='hidden layer dim')
    parse.add_argument('--zdim', type=int, default=128,
                       help='latent variable dim')
    return parse

def main():
    args = parser().parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # MNIST Normalize
    # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    view = False
    sigmoid = True
    if args.mtype == 'CAE':
        indims = 28
        model = CAE(indims, args.f_nums, args.f_sizs, args.strides).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        lossf = nn.BCELoss()
        lossf.reduction = 'sum'
    else:
        indim = 28 * 28
        if args.mtype == 'AE':
            model = AE(indim, args.hiddims, args.zdim).to(device)
        elif args.mtype == 'VAE':
            model = VAE(indim, args.hiddims, args.zdim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        lossf = nn.BCELoss()
        lossf.reduction = 'sum'
        view = True

    # print(model)
    # print(args.seed)
    # sys.exit(1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, lossf=lossf, view=view)
        test(args, model, device, test_loader, lossf=lossf, view=view)

if __name__ == '__main__':
    main()
