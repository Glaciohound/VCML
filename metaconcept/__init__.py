#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Chi Han, Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/16/2019
#
# Distributed under terms of the MIT license.

_info = None
_args = None


class LazyBinder(object):
    def __init__(self, getter_func):
        object.__setattr__(self, '__getter_func', getter_func)

    def __getter(self):
        return object.__getattribute__(self, '__getter_func')()

    def __getattr__(self, attr):
        return getattr(self.__getter(), attr)

    def __setattr__(self, attr, value):
        return setattr(self.__getter(), attr, value)


info = LazyBinder(lambda: _info)
args = LazyBinder(lambda: _args)


def set_global_info(new_info):
    global _info
    assert _info is None
    _info = new_info


def set_global_args(new_args):
    global _args
    assert _args is None
    _args = new_args

