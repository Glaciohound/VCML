#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : load_questions.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# utility for dynmaic loading questions

import os
import numpy as np
from collections import Counter

from utility.common import\
    load, contains, only_contains, \
    union, difference, random_choice_ratio
from dataset.question_dataset \
    import question_dataset


# for balancing dataset suites

def balance_Ans_dataset(dataset, logger):
    output = dataset.random_choice(
        requirement=None,
        num=-1,
        balance_fn=lambda q: q['answer'],
    )
    logger(Counter([q['answer'] for q in output.question_list]), pretty=True)
    logger(f'After balancing: {len(output)}', resume=True)
    return output


def balance_KwAns_dataset(dataset, logger, balance_fn=None):
    if balance_fn is None:
        def balance_fn(q):
            return (q['keywords'][0], q['answer'])
    output = dataset.random_choice(
        requirement=None,
        num=-1,
        balance_fn=balance_fn,
    )
    logger(Counter([q['answer'] for q in output.question_list]), pretty=True)
    logger(f'After balancing: {len(output)}', resume=True)
    return output


def filter_kw_dataset(dataset, kw_concepts, logger, reverse=False):
    middle = 'outside' if reverse else 'in'
    logger(f'Filtering a dataset with keywords {middle} {kw_concepts}')
    logger(f'before filtering: size={len(dataset)}', resume=True)
    output = dataset.filter(
        lambda q: contains(q['keywords'], kw_concepts) != reverse
    )
    logger(f'after filtering: size={len(output)}', resume=True)
    return output


def balance_Ans_suite(suite, logger):
    logger('Balancing questions according to answers')
    with logger.levelup():
        output = {
            split: balance_Ans_dataset(
                dataset, logger
            )
            for split, dataset in suite.items()
        }
    return output


def balance_KwAns_suite(suite, logger, balance_fn=None):
    logger('Balancing questions according to keywords and answers')
    with logger.levelup():
        output = {
            split: balance_KwAns_dataset(
                dataset, logger, balance_fn
            )
            for split, dataset in suite.items()
        }
    return output


def fewer_bias(questions, test_concepts, ratio, logger):
    logger('Biasing a set of questions')
    with logger.levelup():
        logger(f'Original size: {len(questions)}')
        involved_indexes = {
            concept: [ind for ind, qst in questions.questions.items()
                      if concept in qst['keywords']]
            for concept in test_concepts
        }
        remove = union(
            *tuple(random_choice_ratio(indexes, 1 - ratio)
                   for indexes in involved_indexes.values()),
            as_set=True,
        )
        full = union(
            *tuple(involved_indexes.values()),
            as_set=True,
        )
        logger(f'Removed: {len(remove)} out of {len(full)}', resume=True)
        output = questions.copy()
        output.set_indexes(difference(output.indexes, remove))
    return output


def fewer_bias_clsf(questions, test_concepts, ratio, logger,
                    reverse=False):
    logger('Biasing a set of classify questions')
    with logger.levelup():
        logger('Copying the original dataset')
        output = questions.copy(deep=True)
        logger('Biasing')
        for q in logger.tqdm(output.question_list):
            classify_concepts = [
                op['argument'] for op in q['semantic']
            ]
            test_indexes = [
                i for i, concept in enumerate(classify_concepts)
                if (concept in test_concepts) != reverse
            ]
            shape = (q['answer'].shape[0], len(test_indexes))
            mask = np.random.rand(*shape) < ratio
            q['confidence'][:, test_indexes] = \
                q['confidence'][:, test_indexes] * mask
    return output


def get_size(suite, logger, print_it=False):
    size = {split: len(dataset) for split, dataset in suite.items()}
    if print_it:
        logger(f'Size of the dataset: {size}')
    return size


# for manipulating dataset suites
def apply_ratio(suite, ratio, args, logger):
    logger('applying ratio: train={0:0.3f}, val={1:0.3f}, test={2:0.3f}'
           .format(ratio['train'], ratio['val'], ratio['test']))
    return {
        split: dataset.random_choice(
            None,
            args.size * ratio[split],
            None
        )
        for split, dataset in suite.items()
    }


def split_testConcepts(suite, test_concepts, logger):
    logger('Splitting questions according to test_concepts')
    output = {
        'train': suite['train'].filter(
            lambda q: not contains(q.get('keywords', ()), test_concepts)
        ),
        'val': suite['val'].filter(
            lambda q: not contains(q.get('keywords', ()), test_concepts)
        ),
        'test': suite['test'].filter(
            lambda q: only_contains(q.get('keywords', ()), test_concepts)
        )
    }
    logger(suite_length_string(output), resume=True)
    return output


def identical_suite(dataset, logger):
    logger('Getting an identical suite')
    suite = {
        'train': dataset,
        'val': dataset,
        'test': dataset,
    }
    return suite


def filter_suite(suite, requirement, logger):
    logger('Filtering suite according to requirement')
    output = {
        split: dataset.filter(requirement)
        for split, dataset in suite.items()
    }
    return output


def union_suites(suites):
    output = {}
    for split, dataset in suites[0].items():
        output[split] = dataset
        for another in suites[1:]:
            output[split] = output[split].union(another[split])
    return output


def suite_length_string(suite):
    length_dict = {
        split: len(suite[split])
        for split in suite.keys()
    }
    output = str(length_dict)
    return output


# basic loading functions

def load_question_file(group, name, args, logger):
    logger(f'Loading questions from {group}.{name}')
    filename = os.path.join(args.dataset_dir, 'my_dataset', group, name+'.pkl')
    questions = load(filename)
    questions = dict_questions(questions, name)
    dataset = question_dataset.Dataset(questions, args).get_agent()
    logger(f'size = {len(dataset)}', resume=True)
    return dataset


def dict_questions(questions, category):
    questions = {
        (category, i): q
        for i, q in enumerate(questions)
    }
    return questions


# Below are some ratios for combining two suites
# conceptual generalization ratios
def cg_ratios(args):
    r_con = args.conceptual_question_ratio
    r_vis = 1 - r_con
    visual_ratio = {
        'train': args.split_ratio['train'] * r_vis,
        'val': args.split_ratio['val'] * r_vis,
        'test': 0,
    }
    conceptual_ratio = {
        'train': args.split_ratio['train'] * r_con,
        'val': args.split_ratio['val'] * r_con,
        'test': args.split_ratio['test'],
    }
    return visual_ratio, conceptual_ratio


def two_cg_ratios(args):
    r_con = args.conceptual_question_ratio
    r_vis = 1 - r_con
    visual_ratio = {
        'train': args.split_ratio['train'] * r_vis,
        'val': args.split_ratio['val'] * r_vis,
        'test': 0,
    }
    conceptual_ratio = {
        'train': args.split_ratio['train'] * r_con / 2,
        'val': args.split_ratio['val'] * r_con / 2,
        'test': args.split_ratio['test'] / 2,
    }
    return visual_ratio, conceptual_ratio, conceptual_ratio


# visual generalization ratios
def vg_ratios(args):
    r_con = args.conceptual_question_ratio
    r_vis = 1 - r_con
    visual_ratio = {
        'train': args.split_ratio['train'] * r_vis,
        'val': args.split_ratio['val'] * r_vis,
        'test': args.split_ratio['test'],
    }
    conceptual_ratio = {
        'train': args.split_ratio['train'] * r_con,
        'val': args.split_ratio['val'] * r_con,
        'test': 0,
    }
    return visual_ratio, conceptual_ratio


def vg_vismix_ratios(args, n_conceptual=1):
    r_con = args.conceptual_question_ratio
    r_vis = 1 - r_con
    visual1_ratio = {
        'train': args.split_ratio['train'] * r_vis * 0.99,
        'val': 0,
        'test': 0,
    }
    visual2_ratio = {
        'train': args.split_ratio['train'] * r_vis * 0.01,
        'val': args.split_ratio['val'] * r_vis,
        'test': args.split_ratio['test'],
    }
    conceptual_ratio = {
        'train': args.split_ratio['train'] * r_con / n_conceptual,
        'val': args.split_ratio['val'] * r_con / n_conceptual,
        'test': 0,
    }
    return (visual1_ratio, visual2_ratio) + \
        (conceptual_ratio,) * n_conceptual


def equal_ratios(args):
    ratio = {split: args.split_ratio[split] * 0.5
             for split in args.split_ratio}
    return ratio, ratio


def first_only_ratios(args):
    ratio_first = {split: args.split_ratio[split]
                   for split in args.split_ratio}
    ratio_second = {split: 0
                    for split in args.split_ratio}
    return ratio_first, ratio_second


def combine_with_ratio(suites, ratio, args, logger):
    logger('Combining questions')
    with logger.levelup():
        ratioed = []
        for i in range(len(suites)):
            logger(i)
            with logger.levelup():
                ratioed.append(apply_ratio(suites[i], ratio[i], args, logger))
        united = union_suites(ratioed)
    return united
