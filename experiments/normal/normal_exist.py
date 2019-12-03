#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : normal_exist.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 13.08.2019
# Last Modified Date: 05.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_normal import \
    balanced_exist
from ..utils import assemble, load_questions


def exist_suite(test_concepts, dataset, args, logger):
    logger('Mixing visual and conceptual suites for synonym experment')
    with logger.levelup():
        exist_part = balanced_exist(dataset, args, logger)
        mixed = load_questions.apply_ratio(
            exist_part, args.split_ratio,
            args, logger,
        )
    return mixed


def get_training_schedule(concepts, args, dataset, logger):
    test_concepts = assemble.get_testConcepts(
        concepts, args, logger)
    suite = exist_suite(test_concepts, dataset, args, logger)
    training_schedule = [
        {
            'length': 100,
            'question_splits': suite,
            'test_concepts': test_concepts,
        }
    ]
    return training_schedule


config = {
    'dataset': 'normal',
    'training_schedule': get_training_schedule,
}
