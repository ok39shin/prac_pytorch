# for drawing figure
# coding : utf-8

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

class _Plot2dfig_Conf(object):
    def __init__(self, title, xlbl, ylbl):
        self.legend = False
        self._fig_conf(title, xlbl, ylbl)

    def _fig_conf(self, title, xlbl, ylbl):
        # graph title
        if title:
            plt.title(title)
        # graph axis label
        if xlbl:
            plt.xlabel(xlbl)
        if ylbl:
            plt.ylabel(ylbl)

    def set_lim(self, xlim=None, ylim=None):
        pass

    def save_fig(self, fname=None):
        if self.legend:
            plt.legend()
        if not fname:
            fname = './test.png'
        plt.savefig(fname)
        plt.close()

class _Plot1d_Signal(_Plot2dfig_Conf):
    def __init__(self, title, xlbl, ylbl):
        super(_Plot1d_Signal, self).__init__(title, xlbl, ylbl)

    def plot_x(self, x, label=None):
        t = range(len(x))
        if label:
            self.legend = True
            plt.plot(t, x, label=label)
        else:
            plt.plot(t, x)


class Plot1d_Signals(_Plot1d_Signal):
    def __init__(self, xs, title, xlbl, ylbl, xlim=None, ylim=None, labels=None, fname=None):
        super(Plot1d_Signals, self).__init__(title, xlbl, ylbl)
        xs = self._2ndarray(xs)
        labels = self._checklabels(xs.shape[0], labels)
        for dim in range(xs.shape[0]):
            self.plot_x(xs[dim], labels[dim])
        self.set_lim(xlim, ylim)
        self.save_fig(fname=fname)

    def _2ndarray(self, xs):
        if not type(xs) == np.ndarray:
            xs = np.array(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        return xs

    def _checklabels(self, num, labels):
        if not labels:
            labels = [None] * num
        assert num == len(labels)
        return labels

class Plot_LossCurve(Plot1d_Signals):
    def __init__(self, xs, title='Loss curve', xlbl='epoch', ylbl='Loss', labels=['loss'], fname='./loss.png'):
        super(Plot_LossCurve, self).__init__(xs, title, labels=labels, fname=fname)

class Plot_AccCurve(Plot1d_Signals):
    def __init__(self, xs, title='Acc curve', xlbl='epoch', ylbl='Acc', labels=['acc'], fname='./acc.png'):
        super(Plot_AccCurve, self).__init__(xs, title, labels=labels, fname=fname)
