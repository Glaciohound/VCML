#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : debiasing_clsf_ablation.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 25.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This experiment is for testing the hypernym-relationship on CUB dataset
# It contains only the hypernym-extended classification task and the
# hypernym-type relation questions


from .load_debiasing import classify_debiasing
from ..utils import load_questions


def debiasing_ablation(dataset, args, logger):
    logger(f'Building a classify-debiasing dataset on {args.task} dataset')
    with logger.levelup():
        clsf_part = classify_debiasing(
            dataset, args, logger)
        suite = load_questions.apply_ratio(
            clsf_part, args.split_ratio, args, logger
        )
    return suite


def get_training_schedule(concepts, args, dataset, logger):
    suite = debiasing_ablation(dataset, args, logger)
    training_schedule = [
        {
            'length': 100,
            'question_splits': suite,
            'test_concepts': [],
        }
    ]
    return training_schedule


config = {
    'dataset': 'debiasing',
    'training_schedule': get_training_schedule,
}
