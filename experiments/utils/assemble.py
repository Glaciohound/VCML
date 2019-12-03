#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : assemble.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 31.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# Codes for assembling datasets

from utility.common import \
    random_choice_ratio, load_knowledge, intersection, random_one


def get_testConcepts(source, args, logger):
    test_concepts = random_choice_ratio(
        source, args.generalization_ratio)
    logger(f'Selecting test concepts: \n{test_concepts}')
    logger(f'num = {len(test_concepts)}', resume=True)
    return test_concepts


def get_testConcepts_zeroshot(source, args, logger):
    synonym_stats = load_knowledge(args.task, 'synonym')
    syn_groups = [
        intersection(source, synset)
        for exampler, synset in synonym_stats.items()
    ]
    test_concepts = [
        random_one(synset)
        for synset in random_choice_ratio(syn_groups, 0.5)
        if len(synset) > 1
    ]
    logger(f'Selecting test concepts: \n{test_concepts}')
    logger(f'num = {len(test_concepts)}', resume=True)
    return test_concepts
