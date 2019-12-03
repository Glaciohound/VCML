#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : normal_vismix_both.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 13.08.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_normal import \
    raw_classify, balanced_exist,\
    synonym_balanced_full, samekind_balanced_full
from ..utils import load_questions


def mixed_both(dataset, args, logger):
    logger('Mixing visual and two conceptual suites')
    with logger.levelup():
        clsf_part = raw_classify(dataset, args, logger)
        exist_part = balanced_exist(dataset, args, logger)
        syn_part = synonym_balanced_full(
            dataset, args, logger)
        sk_part = samekind_balanced_full(
            dataset, args, logger)
        mixed = load_questions.combine_with_ratio(
            (clsf_part, exist_part, syn_part, sk_part),
            load_questions.vg_vismix_ratios(args, 2),
            args, logger,
        )
    return mixed


def get_training_schedule(concepts, args, dataset, logger):
    suite = mixed_both(dataset, args, logger)
    training_schedule = [
        {
            'length': 100,
            'question_splits': suite,
            'test_concepts': (),
        }
    ]
    return training_schedule


config = {
    'dataset': 'normal',
    'training_schedule': get_training_schedule,
}
