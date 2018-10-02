#/usr/bin/env python
import pytest

import numpy as np

import matplotlib.pyplot as plt
from mytorch.drawFig import Plot1d_Signals

@pytest.mark.parametrize(
        ("xs, title, xlbl, ylbl, xlim, ylim, fname, labels"),
        [
            # default
            (np.array(range(100)), None, None, None, None, None, './TEST1.png', None),
            # for _2ndarray
            # (np.zeros((2, 3)), None, None, None, None, None, './TEST.png', None),
            # (range(100), None, None, None, None, None, './TEST.png', None),
            # ([1, 2, 3], None, None, None, None, None, './TEST.png', None),
            # ([[1, 2, 3], [4, 5, 6]], None, None, None, None, None, './TEST.png', None),
            # for _checklabels
            # (np.array(range(100)), None, None, None, None, None, './TEST.png', ['aaa']),
            # (np.array(range(100)), None, None, None, None, None, './TEST.png', ['aaa', 'bbb']),
            # (np.zeros((2, 3)), None, None, None, None, None, './TEST.png', ['aaa']),
            # (np.zeros((2, 3)), None, None, None, None, None, './TEST.png', ['aaa', 'bbb']),
            # for _fig_conf
            # (np.array(range(100)), 'TITLE', 'X', 'Y', None, None, './TEST2.png', None),
            # for save_fig
            (np.array(range(100)), None, None, None, None, None, None, None),
            (np.sin(np.linspace(0, 2*np.pi, 100)), None, None, None, None, None, './TEST.png', ['y=sin(x)']),
            ]
)
def t_Plot1d_Signal(xs, title, xlbl, ylbl, xlim, ylim, labels, fname):
    Plot1d_Signals(xs, title, xlbl, ylbl, xlim, ylim, labels=labels, fname=fname)
