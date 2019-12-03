#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : meronym_exist.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 26.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_cub_meronym import \
    balanced_exist, meronym_balanced_full
from ..utils import load_questions


def mixed_exist_meronym(dataset, args, logger):
    logger('Mixing visual and conceptual suites for meronym experment')
    with logger.levelup():
        exist_part = balanced_exist(dataset, args, logger)
        mero_part = meronym_balanced_full(
            dataset, args, logger)
        mixed = load_questions.combine_with_ratio(
            (exist_part, mero_part),
            load_questions.cg_ratios(args),
            args, logger,
        )
    return mixed


def get_training_schedule(concepts, args, dataset, logger):
    suite = mixed_exist_meronym(dataset, args, logger)
    training_schedule = [
        {
            'length': 100,
            'question_splits': suite,
            'test_concepts': [],
        }
    ]
    return training_schedule


config = {
    'dataset': 'cub_meronym',
    'training_schedule': get_training_schedule,
}
