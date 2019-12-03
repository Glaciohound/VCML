#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : basic_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 15.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# python basic utilities

import sys
import numpy as np


def dict_gather(dicts):
    summon = {}
    for d in dicts:
        for key, value in d.items():
            if key not in summon:
                summon[key] = []
            summon[key].append(value)
    output = {
        key: np.array(values)
        for key, values in summon.items()
    }
    return output


def contains(x, elements):
    for e in elements:
        if e in x:
            return True
    return False


def only_contains(x, elements):
    for y in x:
        if y not in elements:
            return False
    return True


def belongs_to(stats, query):
    for cat, items in stats.items():
        if query in items:
            return cat
    return None


def intersection(*arg, as_set=False):
    """
    Taking the intersection of multiple iterables.
    """
    output = arg[0]

    if as_set:
        output = set(output)
    else:
        # as list
        output = list(output)

    for y in arg[1:]:
        if as_set:
            output = output.intersection(set(y))
        else:
            set_y = set(y)
            output = [i for i in output if i in set_y]

    return output


def union(*arg, as_set=False):
    """
    Taking the union of multiple iterables
    If the first input is set, or `as_set` is True, the output will be cast
    to a set variable. Otherwise, the output will be a list instance
    """

    output = arg[0]

    if as_set:
        output = set(output)
    else:
        # as list
        output = list(output)

    for y in arg[1:]:
        if as_set:
            output = output.union(set(y))
        else:
            set_output = set(output)
            output = output + [i for i in y if i not in set_output]

    return output


def sum_list(*arg):
    output = arg[0]
    for y in arg[1:]:
        output = output + y
    return output


def difference(x, y):
    # only set or list supported
    if isinstance(x, set):
        return x.difference(set(y))
    else:
        set_y = set(y)
        return [i for i in x if i not in set_y]


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and\
            not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class matrix_dict:
    """
    A matrix-like dict class
    Each query takes two keys
    """
    def __init__(self, keys_x, keys_y, values):
        self.keys_x = keys_x
        self.keys_y = keys_y
        self.values = values
        self.build_dict()
        self.assert_values()

    def build_dict(self):
        self.dict_x = dict(zip(self.keys_x, range(self.dim_x)))
        self.dict_y = dict(zip(self.keys_y, range(self.dim_y)))

    def assert_values(self):
        values = self.values
        assert isinstance(values, list) and len(values) == self.dim_x
        for y_values in values:
            assert isinstance(y_values, list)
            assert len(y_values) == self.dim_y

    @property
    def dim_x(self):
        return len(self.keys_x)

    @property
    def dim_y(self):
        return len(self.keys_y)

    def __getitem__(self, query):
        query_x, query_y = query
        output = self.values[
            self.dict_x[query_x]
        ][
            self.dict_y[query_y]
        ]
        return output
