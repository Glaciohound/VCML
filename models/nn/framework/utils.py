#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 13.08.2019
# Last Modified Date: 21.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license

import torch


infinitesimal = 1e-30
infinite = 1e30
finite = 1e3


def clamp_infinite(value):
    return torch.clamp(value, -infinite, infinite)


def clamp_finite(value):
    return torch.clamp(value, -finite, finite)
