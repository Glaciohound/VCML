#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : math_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 09.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# math tools

import numpy as np
import torch
import torch.nn.functional as F

from .tensor_utils import to_numpy


def ceiling_division(x, y):
    return (x-1) // y + 1


def min_fn(*xs):
    output = xs[0]
    for x in xs[1:]:
        output = torch.min(output, x)
    return output


def max_fn(x, y):
    return torch.max(x, y)


def equal_ratio(x, y):
    match = equal_items(x, y)
    return match.mean()


def equal_items(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        match = (np.array(x) == np.array(y)).astype(float)
    else:
        match = (x == y).float()
    return match


def recall(x, y):
    match = to_numpy(x) * to_numpy(y)
    return match.sum() / y.sum()


def arange(n):
    return list(range(n))


def logit_and(x, y):
    max_ = max_fn(x, y)
    min_ = min_fn(x, y)
    residue_true = -min_fn(max_, max_ - min_)

    return min_ - F.softplus(residue_true)


def logit_exist(x, y):
    return x + F.softplus(-y)


def log_and(x, y):
    return log(x) + log(y)


def log_or(x, y):
    return -logit_and(-x, -y) + log_and(-x, -y)
    # return log(max_fn(x, y))


def log(x):
    return -F.softplus(-x)


def log_xor(x, y):
    return F.softplus(y - x) - F.softplus(-x) - F.softplus(y)


def log_xand(x, y):
    return log_xor(-x, y)


def logit_xor(x, y):
    return x + F.softplus(y - x) - F.softplus(y + x)


def logit_xand(x, y):
    return -logit_xor(x, y)
