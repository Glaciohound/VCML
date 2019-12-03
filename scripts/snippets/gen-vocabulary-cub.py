#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen-vocabulary-cub.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 22.07.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This code is for generating the vocabulary file of CUB (vocabulary.json). The
# whole vocabulary set should contain all bird classes, body-part attributes
# and higher-level biological groups.

import os
import sys
from IPython.core import ultratb
from IPython import embed

from utility.common import dump, load

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)

"""
Component functions for Main
"""


def main(hypernym_json, isinstanceof_json, vocabulary_pkl):
    hypernym = load(hypernym_json)
    isinstanceof = load(isinstanceof_json)
    vocabulary = isinstanceof['species'] + \
        isinstanceof['body_attribute'] + \
        list(hypernym.keys())
    dump(vocabulary, vocabulary_pkl)
    embed()


"""
Running main program
"""
if __name__ == '__main__':
    print('Program Starts')

    root_dir = '.'
    # relative path
    hypernym_json = 'knowledge/cub_hypernym.json'
    isinstanceof_json = 'knowledge/cub_isinstanceof.json'
    data_dir = '../data/cub'
    vocabulary_pkl = 'processed/word2index/vocabulary.pkl'
    # get absolute path
    data_dir = os.path.join(root_dir, data_dir)
    hypernym_json = os.path.join(root_dir, hypernym_json)
    isinstanceof_json = os.path.join(root_dir, isinstanceof_json)
    vocabulary_pkl = os.path.join(data_dir, vocabulary_pkl)

    # run main function
    main(hypernym_json, isinstanceof_json, vocabulary_pkl)
