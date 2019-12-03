#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : normal_clsf.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 13.08.2019
# Last Modified Date: 18.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_normal import raw_classify
from ..utils import assemble, load_questions


def classify_suite(test_concepts, dataset, args, logger):
    logger('Mixing visual and conceptual suites for synonym experment')
    with logger.levelup():
        clsf_part = raw_classify(dataset, args, logger)
        suite = load_questions.apply_ratio(
            clsf_part, args.split_ratio, args, logger)
    return suite


def get_training_schedule(concepts, args, dataset, logger):
    test_concepts = assemble.get_testConcepts(
        concepts, args, logger)
    suite = classify_suite(test_concepts, dataset, args, logger)
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
