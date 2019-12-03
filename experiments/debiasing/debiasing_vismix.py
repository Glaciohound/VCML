#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : debiasing_vismix.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 07.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This experiment is for testing the hypernym-relationship on CUB dataset
# It contains only the hypernym-extended classification task and the
# hypernym-type relation questions


from .load_debiasing import \
    classify_debiasing, exist_debiasing, samekind_balanced_full
from ..utils import load_questions


def curriculum_debiasing(dataset, args, logger):
    logger(f'Building a visually-mixed debiasing dataset on {args.task}')
    with logger.levelup():
        clsf_part = classify_debiasing(dataset, args, logger)
        exist_part = exist_debiasing(dataset, args, logger)
        sk_part = samekind_balanced_full(dataset, args, logger)
        mixed = load_questions.combine_with_ratio(
            (clsf_part, exist_part, sk_part),
            load_questions.vg_vismix_ratios(args),
            args, logger
        )
        clsf_only = load_questions.apply_ratio(
            clsf_part, args.split_ratio, args, logger)
    return clsf_only, mixed


def get_training_schedule(concepts, args, dataset, logger):
    clsf_only, mixed = curriculum_debiasing(dataset, args, logger)
    training_schedule = [
        {
            'length': 100,
            'question_splits': mixed,
            'test_concepts': [],
        },
    ]
    logger(training_schedule, pretty=True)
    return training_schedule


config = {
    'dataset': 'debiasing',
    'training_schedule': get_training_schedule,
}
