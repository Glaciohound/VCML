#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : zeroshot.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 27.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This experiment is for testing the hypernym-relationship on CUB dataset
# It contains only the hypernym-extended classification task and the
# hypernym-type relation questions


from .load_normal import \
    exist_zeroshot, synonym_balanced_full
from ..utils import assemble, load_questions


def zero_shot(dataset, test_concepts, args, logger):
    logger(f'Building a zero-shot dataset for {args.task}')
    with logger.levelup():
        exist_part = exist_zeroshot(
            dataset, test_concepts, args, logger)
        syn_part = synonym_balanced_full(dataset, args, logger)
        mixed = load_questions.combine_with_ratio(
            (exist_part, syn_part),
            load_questions.vg_ratios(args),
            args, logger
        )
    return mixed


def get_training_schedule(concepts, args, dataset, logger):
    test_concepts = assemble.get_testConcepts_zeroshot(
        concepts, args, logger)
    suite = zero_shot(dataset, test_concepts, args, logger)
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
