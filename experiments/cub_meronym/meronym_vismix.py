#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : meronym_vismix.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_cub_meronym import \
    raw_classify_meronym, balanced_exist, meronym_balanced_full
from ..utils import load_questions


def mixed_hypernym(dataset, args, logger):
    logger('Mixing visual and conceptual suites for hypernym experment')
    with logger.levelup():
        clsf_part = raw_classify_meronym(dataset, args, logger)
        exist_part = balanced_exist(dataset, args, logger)
        mero_part = meronym_balanced_full(dataset, args, logger)
        mixed = load_questions.combine_with_ratio(
            (clsf_part, exist_part, mero_part),
            load_questions.vg_vismix_ratios(args),
            args, logger,
        )
    return mixed


def get_training_schedule(concepts, args, dataset, logger):
    suite = mixed_hypernym(dataset, args, logger)
    training_schedule = [
        {
            'length': 100,
            'question_splits': suite,
            'test_concepts': (),
        }
    ]
    return training_schedule


config = {
    'dataset': 'cub_meronym',
    'training_schedule': get_training_schedule,
}
