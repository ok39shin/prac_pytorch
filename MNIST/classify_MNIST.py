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

from mytorch.model import NN_classifier, CNN2d_classifier

def train(args, model, device, train_loader, optimizer, epoch, lossf=nn.NLLLoss(reduction='sum'), view=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if view:
            data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, lossf=nn.NLLLoss(reduction='sum'), view=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if view:
                data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += lossf(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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
    parse.add_argument('--mtype', type=str, default='conv',
                       help='kind of model: conv=CNN(default), linear=NN')
    parse.add_argument('--f_nums', type=list, default=[10, 20],
                       help='frame num list')
    parse.add_argument('--f_sizs', type=list, default=[7, 5],
                       help='frame size list')
    parse.add_argument('--strides', type=list, default=[2, 2],
                       help='stride size list')
    parse.add_argument('--c_nlay', type=int, default=2,
                       help='classifier layer num')
    parse.add_argument('--c_laydim', type=int, default=256,
                       help='classifier layer dim')
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
    if args.mtype == 'conv':
        indims = 28
        c_out = 10
        model = CNN2d_classifier(indims, args.f_nums, args.f_sizs, args.strides, args.c_nlay, args.c_laydim, c_out).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        lossf = model.CLS.lossf
        lossf.reduction = 'sum'
    elif args.mtype == 'Linear':
        indim = 28 * 28
        c_out = 10
        model = NN_classifier(indim, args.c_nlay, args.c_laydim, c_out).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        lossf = nn.NLLLoss(reduction='sum')
        view = True

    # print(model)
    # print(args.seed)
    # sys.exit(1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, lossf=lossf, view=view)
        test(args, model, device, test_loader, lossf=lossf, view=view)

if __name__ == '__main__':
    main()
