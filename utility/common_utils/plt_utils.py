#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : plt_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# printing / visualization


import numpy as np
import matplotlib


matplotlib.use('Agg')


def import_plt():
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    return plt


def get_axes(num, plt):
    nrows = int(np.sqrt(num))
    ncols = num // nrows + 1

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols*4, nrows*4))
    ax_array = ax.flatten()
    return fig, ax_array


plt = import_plt()
