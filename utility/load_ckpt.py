#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : load_ckpt.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 27.07.2019
# Last Modified Date: 03.12.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file contains utility functions for loading embedding weights

import os
import torch
import numpy as np
import pandas as pd
import csv

from dataset.question_dataset.utils.misc import singular
from utility.common import make_dir


def load_embedding(filename, logger):
    '''
    with open(filename, 'r') as f:
        read_lines = f.readlines()

    data = [
        line.rstrip('\n').split(' ')
        for line in read_lines
    ]
    data = np.array(data)
    words = data[:, 0]
    weights = data[:, 1:].astype(float)
    return words, weights
    '''
    logger(f'Loading GloVe embedding from {filename}')
    table = pd.read_csv(
        filename, sep=' ', index_col=0,
        header=None, quoting=csv.QUOTE_NONE)
    return table


def align_weights(table, target, default, logger):
    logger('Aligning embedding weights from pretrained glove to model')

    def get_value(word):
        try:
            word = word.lower()
            word = singular(word)
            output = table.loc[word].values
        except Exception:
            output = default
        return output
    weight = np.stack([
        get_value(word)
        for word in target
    ])
    return weight


def download_ckpt(args, task, name, index, is_main=True):
    if task in ['CLEVR', 'GQA']:
        ckpt_name = task + '.pth'
    elif 'meronym' in name:
        ckpt_name = 'CUB_meronym.pth'
    else:
        ckpt_name = 'CUB_hypernym.pth'
    temp_dir = os.path.join(args.temp_dir, str(index))
    make_dir(temp_dir)

    ckpt_link = os.path.join(args.webpage, 'ckpt', ckpt_name)
    ckpt_file = os.path.join(temp_dir, ckpt_name)
    os.system(f'rm -f {ckpt_file}')
    if is_main:
        os.system(f'wget {ckpt_link} -P {temp_dir}')
    else:
        os.system(f'wget -q {ckpt_link} -P {temp_dir}')
    ckpt = torch.load(ckpt_file)
    os.system(f'rm {ckpt_file}')
    return ckpt
