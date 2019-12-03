#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : choice_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 25.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# selecting functions


import numpy as np

from .basic_utils import sum_list


def pick_one(x, requirement=None):
    if requirement:
        x = [y for y in x if requirement(y)]
    return list(x)[0]


def random_one(x, *arg, **kwarg):
    output = random_choice(x, *arg, num=1, **kwarg)
    if output is None:
        return None
    else:
        return output[0]


def group_by_value(x, fn, on_value=None):
    if on_value is None:
        def by_fn(y):
            return fn(y)
    else:
        def by_fn(y):
            return fn(on_value[y])

    all_values = set([by_fn(y) for y in x])
    output = {v: set() for v in all_values}

    for y in x:
        output[by_fn(y)].add(y)
    output = {
        k: list(v) for k, v in output.items()
    }
    return output


def random_choice(x, num, requirement=None, balance_fn=None,
                  on_value=None, replace=False, **kwarg):

    if requirement is not None:
        if on_value is None:
            x = [y for y in x if requirement(y)]
        else:
            x = [y for y in x if requirement(on_value[y])]

    if balance_fn is not None:
        group_indexes = group_by_value(x, balance_fn, on_value)
        if num > 0:
            each_num = num // len(group_indexes)
        else:
            each_num = max([len(group) for group in group_indexes.values()])
        outputs = [
            random_choice(
                group,
                num=each_num,
                replace=replace,
            )
            for group in group_indexes.values()
        ]
        return sum_list(*outputs)

    array = np.empty(len(x), 'object')
    array[:] = x
    x = array

    if len(x) == 0:
        return []
    elif num == -1:
        return x.tolist()
    elif len(x) >= num:
        return np.random.choice(x, num, replace=replace, **kwarg).tolist()
    else:
        n_round = num // x.shape[0]
        return np.concatenate(
            [x] * n_round +
            [np.random.choice(x, num-len(x)*n_round,
                              replace=replace, **kwarg)]
        ).tolist()


def random_choice_ratio(x, ratio):
    return random_choice(x, int(len(x) * ratio))
