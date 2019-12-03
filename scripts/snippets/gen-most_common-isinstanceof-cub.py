#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen-most_common-isinstanceof-cub.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 21.07.2019
# Last Modified Date: 30.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license

"""
This file describes code used for generating most_common counter for CUB-birds
dataset.
"""

import os
import sys
from IPython.core import ultratb
from IPython import embed
from collections import defaultdict

from utility.common import dump

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)


"""
Backbone functions
"""


def get_body_types(attributes):
    output = defaultdict(lambda: [])
    for name in attributes:
        haswhat, content = name.split('::')
        what = haswhat.split('has_')[-1]
        output[what].append(name)
    return output


def main(bird_class_txt, bird_attribute_txt,
         most_common_json, isinstanceof_json):
    classes = read_names(bird_class_txt)
    attributes = read_names(bird_attribute_txt)
    body_instanceof = get_body_types(attributes)
    cub_most_common = {
        name: 1
        for name in classes + attributes
    }
    cub_most_common = {'cub': cub_most_common}
    cub_isinstanceof = {
        'species': classes,
        **body_instanceof,
    }

    print(f'Outputting most common knowledge to {most_common_json}')
    dump(cub_most_common, most_common_json)
    print(f'Outputting isinstanceof knowledge to {isinstanceof_json}')
    dump(cub_isinstanceof, isinstanceof_json)

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


if __name__ == '__main__':
    print('Program Starts')

    root_dir = '.'
    # relative path
    most_common_json = 'knowledge/cub_most_common.json'
    isinstanceof_json = 'knowledge/cub_isinstanceof.json'
    data_dir = '../data/cub'
    bird_class_txt = 'raw/classes.txt'
    bird_attribute_txt = 'raw/attributes/attributes.txt'

    # get absolute path
    data_dir = os.path.join(root_dir, data_dir)
    most_common_json = os.path.join(root_dir, most_common_json)
    isinstanceof_json = os.path.join(root_dir, isinstanceof_json)
    bird_class_txt = os.path.join(data_dir, bird_class_txt)
    bird_attribute_txt = os.path.join(data_dir, bird_attribute_txt)

    # run main function
    main(bird_class_txt, bird_attribute_txt,
         most_common_json, isinstanceof_json)
