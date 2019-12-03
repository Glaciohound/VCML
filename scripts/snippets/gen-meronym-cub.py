#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen-meronym-cub.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 16.07.2019
# Last Modified Date: 30.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license

"""
This file describes code used for generating meronym data for CUB-birds
dataset.
"""

import os
import sys
import numpy as np
from IPython.core import ultratb
from IPython import embed

from utility.common import dump, load

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)


"""
Backbone functions
"""


def generate_meronym(classes, attributes, data, true_th, false_th):
    meronym = {}
    true_matrix = data >= true_th
    false_matrix = data <= false_th
    for i, cls in enumerate(classes):
        meronym[cls] = {
            'true': [], 'false': []
        }
        for j, attr in enumerate(attributes):
            if true_matrix[i, j]:
                meronym[cls]['true'].append(attr)
            elif false_matrix[i, j]:
                meronym[cls]['false'].append(attr)

    return meronym


def main(
    bird_class_txt, bird_attribute_txt, labels_txt, meronym_json,
    true_th, false_th,
):
    classes = read_names(bird_class_txt)
    attributes = read_names(bird_attribute_txt)
    data = read_numbers(labels_txt) / 100

    cub_meronym = generate_meronym(
        classes, attributes, data,
        true_th, false_th
    )
    cub_meronym = {'cub': cub_meronym}

    meronym = load(meronym_json)
    meronym.update(cub_meronym)
    dump(meronym, meronym_json)

    embed()


"""
Some utility functions
"""


def read_names(txt_file):
    with open(txt_file, 'r') as f:
        read_lines = f.readlines()

    names = [s.lstrip('0123456789. ').rstrip('\n')
             for s in read_lines]
    return names


def read_numbers(txt_file):
    with open(txt_file, 'r') as f:
        read_lines = f.readlines()

    data = [
        [
            float(num) for num in line.rstrip('\n').split(' ')
        ]
        for line in read_lines
    ]
    data = np.array(data)
    return data


if __name__ == '__main__':
    print('Program Starts')

    root_dir = '.'
    # relative path
    meronym_json = 'knowledge/meronym.json'
    data_dir = '../data/cub'
    bird_class_txt = 'raw/classes.txt'
    bird_attribute_txt = 'raw/attributes/attributes.txt'
    labels_txt = 'raw/attributes/class_attribute_labels_continuous.txt'
    # get absolute path
    data_dir = os.path.join(root_dir, data_dir)
    meronym_json = os.path.join(root_dir, meronym_json)
    bird_class_txt = os.path.join(data_dir, bird_class_txt)
    bird_attribute_txt = os.path.join(data_dir, bird_attribute_txt)
    labels_txt = os.path.join(data_dir, labels_txt)

    true_th = 0.75
    false_th = 0.

    # run main function
    main(
        bird_class_txt, bird_attribute_txt, labels_txt, meronym_json,
        true_th, false_th
    )
