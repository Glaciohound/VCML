#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : build_cub_meronym.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 30.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from dataset.visual_dataset.utils.split_visual import original_split


config = {
    'name': 'cub_hypernym',
    'register_synonyms': False,
    'register_hypernyms': False,
    'register_meronyms': True,
    'visual_question_types': [
        'cub_exist_meronym',
        'cub_classification_meronym',
    ],
    'conceptual_question_types': [
        'meronym_cub'
    ],
    'visual_split_fn': original_split,
    'split_kwarg': {},
}
