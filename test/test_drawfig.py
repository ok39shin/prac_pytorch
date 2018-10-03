#/usr/bin/env python
import pytest

import numpy as np

from mytorch.draw_fig import Plot1d_Signals

@pytest.mark.parametrize(
        ("xs, title, xlbl, ylbl, xlim, ylim, fname, labels"),
        [
            (np.array(range(100)), None, None, None, None, None, './TEST.png', None),
            # (np.array(range(100)), 'TITLE', 'X', 'Y', None, None, './TEST.png', None),
            # (np.array(range(100)), None, None, None, None, None, './TEST.png', None),
            # (np.array(range(100)), None, None, None, None, None, './TEST.png', 'y=x'),
            # (np.sin(np.linspace(0, 2*np.pi, 100)), None, None, None, None, None, './TEST.png', 'y=sin(x)'),
            ]
)
def t_Plot1d_Signal(xs, title, xlbl, ylbl, xlim, ylim, labels, fname):
    Plot1d_Signals(xs, title, xlbl, ylbl, xlim, ylim, labels=labels, fname=fname)
