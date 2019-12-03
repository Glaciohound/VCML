#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : normal_synonym.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 13.08.2019
# Last Modified Date: 18.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_normal import \
    raw_classify, synonym_balanced_split
from ..utils import assemble, load_questions


def mixed_synonym(test_concepts, dataset, args, logger):
    logger('Mixing visual and conceptual suites for synonym experment')
    with logger.levelup():
        clsf_part = raw_classify(dataset, args, logger)
        syn_part = synonym_balanced_split(
            test_concepts, dataset, args, logger)
        mixed = load_questions.combine_with_ratio(
            (clsf_part, syn_part),
            load_questions.cg_ratios(args),
            args, logger,
        )
    return mixed


def get_training_schedule(concepts, args, dataset, logger):
    test_concepts = assemble.get_testConcepts(
        concepts, args, logger)
    suite = mixed_synonym(test_concepts, dataset, args, logger)
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
