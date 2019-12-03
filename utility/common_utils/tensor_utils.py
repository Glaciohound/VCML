#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tensor_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 02.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# pytorch/numpy utilities


import numpy as np
import torch
import torch.nn.functional as F
from pprint import pprint


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        if isinstance(x[0], float):
            return torch.Tensor(x)
        elif isinstance(x[0], int):
            return torch.LongTensor(x)
        else:
            return x
    elif isinstance(x, np.ndarray):
        if x.dtype.char in ['d', 'f']:
            return torch.Tensor(x)
        elif x.dtype.char in ['l', 'b']:
            return torch.LongTensor(x)
        else:
            raise Exception('not convertable')
    elif isinstance(x, int) or isinstance(x, float) \
            or np.isscalar(x):
        return torch.tensor(x)
    else:
        raise Exception('not convertable')


def detach(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor = tensor.numpy()

    return tensor


def matmul(*mats):
    output = mats[0]
    for x in mats[1:]:
        if isinstance(output, torch.Tensor):
            output = torch.matmul(output, x)
        else:
            output = np.matmul(output, x)
    return output


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, torch.autograd.Variable):
        return x.data.cpu().numpy()


def to_normalized(x):
    if isinstance(x, torch.Tensor):
        return F.normalize(x, dim=-1)
    elif isinstance(x, np.ndarray):
        return to_normalized(torch.Tensor(x)).numpy()
    else:
        raise Exception('unsupported type: %s' % str(type(x)))


def init_seed(n=-1, index=-1):
    if n != -1:
        if index != -1:
            seed = n + index
        else:
            seed = n
        torch.manual_seed(seed)
        np.random.seed(seed)


def is_cuda(x):
    return x.device.type == 'cuda'


def valid_part(x, assert_finite=False):
    output = torch.isnan(x).bitwise_not() * (x.abs() != float('inf'))
    if assert_finite:
        output = output * (x.abs() < 1e10)
    return output


def is_valid_value(x, assert_finite):
    if not valid_part(x, assert_finite).all():
        return False
    else:
        return True


def assert_valid_value(*values, assert_finite=False):
    for i, x in enumerate(values):
        if not is_valid_value(x, assert_finite):
            pprint(values)
            print('Invalid tensor is', i)
            raise Exception('invalid value')


def index_by_list(tensor, indexes):
    if isinstance(tensor, torch.Tensor) or \
            isinstance(tensor, np.ndarray):
        return tensor[indexes]
    elif isinstance(tensor, list) or \
            isinstance(tensor, tuple):
        return [tensor[ind] for ind in indexes]
    else:
        raise Exception()
