#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualize_debiasingResults.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 27.10.2019
# Last Modified Date: 28.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
# from IPython import embed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        default='debiasingResults.json')
    parser.add_argument('--save_image', type=str,
                        default='debiasingResults.jpg')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--ipython', action='store_true')
    parser.add_argument('--pause', type=float, default=10)
    parser.add_argument('--shrink', type=float, default=1)
    args = parser.parse_args()
    return args


def visualize(ax, xticks, series, n, kw_configs, kw_data, alpha, shrink):
    x = np.arange(n)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    for series, config in kw_configs.items():
        y = kw_data[series]['y']
        dy = kw_data[series]['dy'] * shrink
        color = config['color']
        ax.plot(x, y,
                color=color,
                linestyle=config['linestyle'],
                label=series)
        ax.fill_between(x, y - dy, y + dy, color=color, alpha=alpha)
    ax.legend()


def get_data(filename):
    with open(filename, 'r') as f:
        loaded = json.load(f)
    kw_configs = loaded['kw_configs']
    xticks = loaded['xticks']
    series = loaded['series']
    n = len(xticks)
    data = np.array(loaded['data'])[::-1].transpose((1, 0, 2))
    kw_data = {
        name: {
            'y': data[i, :, 0],
            'dy': data[i, :, 1],
        }
        for i, name in enumerate(series)
    }
    return xticks, series, n, kw_configs, kw_data


def main(args):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, args.filename)
    imagename = os.path.join(dirname, args.save_image)
    xticks, series, n, kw_configs, kw_data = get_data(filename)
    fig, ax = plt.subplots()
    visualize(ax, xticks, series, n, kw_configs, kw_data,
              alpha=args.alpha, shrink=args.shrink)
    plt.pause(args.pause)
    plt.savefig(imagename)


if __name__ == '__main__':
    args = get_args()
    main(args)
