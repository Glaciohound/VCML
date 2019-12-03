#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : dataloader.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 23.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

from torch.utils.data import DataLoader


def get_dataloader(dataset, collate_fn, args):

    kwargs = {'num_workers': args.num_workers,
              'drop_last': False,
              'pin_memory': False,
              'batch_size': args.batch_size,
              'shuffle': True,
              'collate_fn': collate_fn,
              }

    return DataLoader(
        dataset,
        **kwargs)
