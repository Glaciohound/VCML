#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : clean-up-knowledge.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 16.07.2019
# Last Modified Date: 30.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This code splits knowledge files according to the the dataset,
# from {type}.json to {dataset1}_{type}.json, {dataset2}_{type}.json, ... etc
#
# This code is one-off, and not meant to be used for multiple times.

import os
import sys
from IPython.core import ultratb

from utility.common import dump, load

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)


def tidy(knowledge_dir, knowledge_types):
    """ main function """
    for this_type in knowledge_types:
        knowledge_file = os.path.join(knowledge_dir, f'{this_type}.json')
        this_knowledge = load(knowledge_file)
        for dataset, knowledge in this_knowledge.items():
            new_filename = os.path.join(
                knowledge_dir, f'{dataset}_{this_type}.json')
            knowledge_len = len(knowledge)
            if knowledge_len != 0:
                print(f'dumping {this_type} knowledge for dataset {dataset}: '
                      f'\'{str(knowledge)[:50]} ... \' '
                      f'which has knowledge length {knowledge_len}: ')
                dump(knowledge, new_filename)


"""
Running main program
"""
if __name__ == '__main__':
    print('Program Starts')

    root_dir = '.'
    knowledge_folder = 'knowledge'
    knowledge_dir = os.path.join(root_dir, knowledge_folder)
    knowledge_types = [
        'hypernym',
        'synonym',
        'isinstanceof',
        'meronym',
        'most_common',
    ]

    # run main function
    tidy(
        knowledge_dir, knowledge_types,
    )
