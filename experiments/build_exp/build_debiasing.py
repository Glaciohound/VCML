#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : build_debiasing.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 28.07.2019
# Last Modified Date: 23.09.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# Building a debiasing dataset


from dataset.visual_dataset.utils.split_visual\
    import split_by_visual_bias


default_bias = {
    'gqa': {
        'glass': [
            'beige',
            'black',
            'blue',
            'brown',
            'blond',
            'dark',
            'gold',
        ],
        'wood': [
            'gray',
            'green',
            'orange',
            'pink',
            'purple',
            'red',
            'silver',
        ],
        'rock': [
            'beige',
            'brown',
            'blond',
            'gold',
            'green',
            'pink',
            'yellow',
            'white',
            'tan',
        ],
    },
    'clevr': {
        'cube': [
            'gray',
            'blue',
            'brown',
            'yellow',
        ]
    }
}

config = {
    'name': 'visual_bias',
    'register_synonyms': True,
    'register_hypernyms': False,
    'register_meronyms': False,
    'visual_question_types': [
        'biased_exist',
        'exist',
        'classification',
    ],
    'conceptual_question_types': [
        'samekind',
    ],
    'visual_split_fn': split_by_visual_bias,
    'split_kwarg': {'visual_bias': default_bias},
    'default_bias': default_bias,
}
