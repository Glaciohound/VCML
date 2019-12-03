#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : cub_few_shot_ablation_byclsf.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 20.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from .load_cub_hypernym import biased_exist, biased_classify
from ..utils import assemble, load_questions


def few_shot_ablation(dataset, test_concepts, args, logger):
    logger('Building an ablated few-shot dataset for CUB')
    with logger.levelup():
        exist_part = biased_exist(
            dataset, test_concepts, args, logger)
        clsf_part = biased_classify(
            dataset, test_concepts, args, logger)
        hybrid_config = load_questions.train_A_for_B()
        hybrid_visual = load_questions.hybrid_splits(
            (clsf_part, exist_part),
            hybrid_config,
            logger,
        )
        suite = load_questions.apply_ratio(
            hybrid_visual, args.split_ratio, args, logger)
    return suite


def get_training_schedule(concepts, args, dataset, logger):
    test_concepts = assemble.get_testConcepts(
        concepts, args, logger)
    suite = few_shot_ablation(dataset, test_concepts, args, logger)
    suite = load_questions.apply_ratio(suite, args.split_ratio, args, logger)
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
