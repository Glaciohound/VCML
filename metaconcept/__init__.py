#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Chi Han, Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/16/2019
#
# Distributed under terms of the MIT license.

info = None
args = None


def set_global_info(new_info):
    global info
    assert info is None
    info = new_info


def set_global_args(new_args):
    global args
    assert args is None
    args = new_args
