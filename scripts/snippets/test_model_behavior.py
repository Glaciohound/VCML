#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : test_model_behavior.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 02.10.2019
# Last Modified Date: 03.12.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.init as init
import sys
import argparse
import matplotlib.pyplot as plt
from IPython.core import ultratb
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from IPython import embed

sys.path.append('.')
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=True)

color_series = ['black', 'blue', 'red', 'yellow', 'blue']

from utility.common import valid_part
from models.nn.framework import functional


def l2norm(tensor):
    return tensor.pow(2).sum(-1).sqrt()


def go_far(functional, half_gaussion, args):
    fig, ax = plt.subplots()
    x = torch.zeros(args.dim, requires_grad=True)
    init.normal_(x, 0, args.init_variance)
    optimizer = optim.SGD([x], lr=args.lr)

    length_history = []
    for i in tqdm(range(args.steps)):
        length = l2norm(x)
        length_history.append(length.detach().numpy())

        loss = -functional.ln_cdf(length)
        step(loss, optimizer)
        ax.cla()
        ax.plot(length_history)
        ax.set_xlabel('step')
        ax.set_ylabel('length')
        ax.label_outer()
        plt.pause(0.00001)

    savefig(args)


def A_X_trace(functional, half_gaussion, args):
    fig, ax = plt.subplots()

    for color in color_series[:args.series]:
        x = torch.zeros(args.dim)
        unit = torch.Tensor([1., 0.])
        # init.normal_(x, 0, args.init_variance)
        x[0], x[1] = tuple(map(float,
                               input('input the initial pos: ').split(', ')))

        x = x.detach().requires_grad_()
        optimizer = optim.SGD([x], lr=args.lr)

        history = np.zeros((args.steps, 2))
        for i in tqdm(range(args.steps)):
            x_detach = x.detach().numpy()
            if not args.silent:
                ax.scatter(*x_detach, c=color)
            else:
                history[i] = x_detach

            ln_pr = half_gaussion.ln_conditional(x[None], unit[None])[0]
            step(-ln_pr, optimizer)
            plt.pause(0.00001)

        if args.silent:
            ax.scatter(history[:, 0], history[:, 1], c=color)

    plt.axis('square')
    embed()
    savefig(args)


def A_X_surface(functional, half_gaussion, args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    axis_values = torch.arange(
        args.min, args.max, (args.max-args.min) / args.num_grid
    ) + 0.1
    x, y = torch.meshgrid(axis_values, axis_values)
    unit = torch.Tensor([1., 0.])
    points = torch.stack([x.flatten(), y.flatten()], 1)
    output = half_gaussion.ln_conditional(points, unit[None])[0][:, 0]
    z = output.reshape(x.shape)
    z[~valid_part(z)] = 0
    z = torch.clamp(z, -args.clamp, args.clamp)

    ax.plot_surface(x, y, z)
    plt.pause(args.pause)


def logit_A_X_surface(functional, half_gaussion, args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    axis_values = torch.arange(
        args.min, args.max, (args.max-args.min) / args.num_grid
    ) + 0.1
    x, y = torch.meshgrid(axis_values, axis_values)
    unit = torch.Tensor([1., 0.])
    points = torch.stack([x.flatten(), y.flatten()], 1)
    output = functional.logit_ln(
        half_gaussion.ln_conditional(points, unit[None])[0][:, 0]
    )
    z = output.reshape(x.shape)
    z[~valid_part(z)] = 0
    z = torch.clamp(z, -args.clamp, args.clamp)

    ax.plot_surface(x, y, z)
    plt.pause(args.pause)


def X_A_surface(functional, half_gaussion, args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    axis_values = torch.arange(
        args.min, args.max, (args.max-args.min) / args.num_grid
    ) + 0.1
    x, y = torch.meshgrid(axis_values, axis_values)
    unit = torch.Tensor([1., 0.])
    points = torch.stack([x.flatten(), y.flatten()], 1)
    output = half_gaussion.ln_conditional(unit[None], points)[0][0]
    z = output.reshape(x.shape)
    z[~valid_part(z)] = 0
    z = torch.clamp(z, -args.clamp, args.clamp)

    ax.plot_surface(x, y, z)
    plt.pause(args.pause)


def lambda_surface(functional, half_gaussion, args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    axis_values = torch.arange(
        args.min, args.max, (args.max-args.min) / args.num_grid
    ) + 0.1
    x, y = torch.meshgrid(axis_values, axis_values)
    unit = torch.Tensor([1., 0.])
    points = torch.stack([x.flatten(), y.flatten()], 1)
    output = half_gaussion.ln_lambda(unit[None], points)[0]
    z = output.reshape(x.shape)
    z[~valid_part(z)] = 0
    z = torch.clamp(z, -args.clamp, args.clamp)

    ax.plot_surface(x, y, z)
    plt.pause(args.pause)


def abs_lambda_surface(functional, half_gaussion, args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    axis_values = torch.arange(
        args.min, args.max, (args.max-args.min) / args.num_grid
    ) + 0.1
    x, y = torch.meshgrid(axis_values, axis_values)
    unit = torch.Tensor([1., 0.])
    points = torch.stack([x.flatten(), y.flatten()], 1)
    output = half_gaussion.ln_lambda(unit[None], points)[0]
    z = output.reshape(x.shape)
    z[~valid_part(z)] = 0
    z = torch.abs(z)
    z = torch.clamp(z, -args.clamp, args.clamp)

    ax.plot_surface(x, y, z)
    plt.pause(args.pause)


def X_A_trace(functional, half_gaussion, args):
    fig, ax = plt.subplots()

    for color in color_series[:args.series]:
        x = torch.zeros(args.dim)
        unit = torch.Tensor([1., 0.])
        # init.normal_(x, 0, args.init_variance)
        x[0], x[1] = tuple(map(float,
                               input('input the initial pos: ').split(', ')))

        x = x.detach().requires_grad_()
        optimizer = optim.SGD([x], lr=args.lr)

        history = np.zeros((args.steps, 2))
        for i in tqdm(range(args.steps)):
            x_detach = x.detach().numpy()
            if not args.silent:
                ax.scatter(*x_detach, c=color)
            else:
                history[i] = x_detach

            ln_pr = half_gaussion.ln_conditional(unit[None], x[None])[0]
            step(-ln_pr, optimizer)
            plt.pause(0.00001)

        if args.silent:
            ax.scatter(history[:, 0], history[:, 1], c=color)

    embed()
    savefig(args)


def step(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def Arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='')

    parser.add_argument('--sample_size', type=int, default=10000)
    parser.add_argument('--max_length', type=float, default=200)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--init_variance', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--series', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--slack', action='store_true')

    parser.add_argument('--image_dir', type=str,
                        default='../nothing/visualize/behavior/')
    parser.add_argument('--silent', action='store_true')

    parser.add_argument('--min', type=float, default=-2)
    parser.add_argument('--max', type=float, default=2)
    parser.add_argument('--clamp', type=float, default=10)
    parser.add_argument('--num_grid', type=int, default=20)
    parser.add_argument('--pause', type=int, default=20)
    return parser


def main():
    args = Arg().parse_args()
    half_gaussion = functional.HalfGaussianConditionalLogit(
        args.sample_size, args.max_length, args.slack
    )

    function_map = {
        'go_far': go_far,
        'A_X_trace': A_X_trace,
        'X_A_trace': X_A_trace,
        'A_X_surface': A_X_surface,
        'logit_A_X_surface': logit_A_X_surface,
        'X_A_surface': X_A_surface,
        'lambda_surface': lambda_surface,
        'abs_lambda_surface': abs_lambda_surface,
    }
    function_map[args.experiment](
        functional, half_gaussion, args)


def savefig(args):
    filename = input('assign a name for the image (`none` if abort): ')
    if filename != 'none':
        plt.savefig(os.path.join(args.image_dir, f'{filename}.jpg'))


if __name__ == '__main__':
    main()
