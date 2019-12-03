#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : cub_exist_hypernym.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 31.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_cub_hypernym import \
    balanced_exist, hypernym_balanced_split
from ..utils import assemble, load_questions


def mixed_exist_hypernym(test_concepts, dataset, args, logger):
    logger('Mixing visual and conceptual suites for hypernym experment')
    with logger.levelup():
        exist_part = balanced_exist(dataset, args, logger)
        hyp_part = hypernym_balanced_split(
            test_concepts, dataset, args, logger)
        mixed = load_questions.combine_with_ratio(
            (exist_part, hyp_part),
            load_questions.cg_ratios(args),
            args, logger,
        )
    return mixed


def get_training_schedule(concepts, args, dataset, logger):
    test_concepts = assemble.get_testConcepts(
        concepts, args, logger)
    suite = mixed_exist_hypernym(test_concepts, dataset, args, logger)
    training_schedule = [
        {
            'length': 100,
            'question_splits': suite,
            'test_concepts': test_concepts,
        }
    ]
    return training_schedule


config = {
    'dataset': 'cub_hypernym',
    'training_schedule': get_training_schedule,
}
