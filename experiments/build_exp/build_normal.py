#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : build_normal.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 26.07.2019
# Last Modified Date: 30.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This dataset splits dataset normally, and contains three types of conceptual
# questions: synonym, isinstanceof, hypernym


from dataset.visual_dataset.utils.split_visual import original_split


config = {
    'name': 'normal',
    'register_synonyms': True,
    'register_hypernyms': False,
    'register_meronyms': False,
    'visual_question_types': [
        'exist',
        'classification'
    ],
    'conceptual_question_types': [
        'synonym',
        'isinstanceof',
        'samekind',
    ],
    'visual_split_fn': original_split,
    'split_kwarg': {},
}
