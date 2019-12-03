#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : load_cub_hypernym.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 07.08.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from ..utils import load_questions
from utility.common import contains


def hypernym_questions(dataset, args, logger):
    hyp_questions = load_questions.load_question_file(
        dataset, 'hypernym_cub', args, logger
    )
    return hyp_questions


def raw_hypernym(dataset, args, logger):
    logger('Identical hypernym suite')
    with logger.levelup():
        hyp_questions = hypernym_questions(dataset, args, logger)
        hyp_full = load_questions.identical_suite(hyp_questions, logger)
    return hyp_full


def hypernym_balanced_full(dataset, args, logger):
    logger('Loading a full hypernym suite')
    with logger.levelup():
        hyp_suite = raw_hypernym(dataset, args, logger)
        balanced = load_questions.balance_KwAns_suite(hyp_suite, logger)
    return balanced


def hypernym_balanced_split(test_concepts, dataset, args, logger):
    logger('Loading balaced hypernym suite splitted by test_concepts')
    with logger.levelup():
        hyp_suite = raw_hypernym(dataset, args, logger)
        hyp_suite = load_questions.split_testConcepts(
            hyp_suite, test_concepts, logger)
        hyp_suite = load_questions.balance_KwAns_suite(hyp_suite, logger)
    return hyp_suite


# Visual part


def raw_classify_hypernym(dataset, args, logger):
    logger('Loading classify-hypernym suite')
    with logger.levelup():
        raw_suite = {
            'train': load_questions.load_question_file(
                dataset, 'train_cub_classification_hypernym', args, logger
            ),
            'val': load_questions.load_question_file(
                dataset, 'val_cub_classification_hypernym', args, logger
            ),
            'test': load_questions.load_question_file(
                dataset, 'test_cub_classification_hypernym', args, logger
            ),
        }
    return raw_suite


def raw_exist_hypernym(dataset, args, logger):
    logger('Loading exist-hypernym suite')
    with logger.levelup():
        raw_suite = {
            'train': load_questions.load_question_file(
                dataset, 'train_cub_exist_hypernym', args, logger
            ),
            'val': load_questions.load_question_file(
                dataset, 'val_cub_exist_hypernym', args, logger
            ),
            'test': load_questions.load_question_file(
                dataset, 'test_cub_exist_hypernym', args, logger
            ),
        }
    return raw_suite


def balanced_exist(dataset, args, logger):
    logger('Loading a balanced exist suite')
    raw_suite = raw_exist_hypernym(dataset, args, logger)
    balanced_suite = load_questions.balance_KwAns_suite(raw_suite, logger)
    return balanced_suite


def biased_exist(dataset, test_concepts, args, logger):
    logger('Biasing dataset ratio according to test concepts')
    with logger.levelup():
        raw_suite = raw_exist_hypernym(dataset, args, logger)
        biased = {
            'train': load_questions.fewer_bias(
                raw_suite['train'], test_concepts, args.fewshot_ratio, logger
            ),
            'val': raw_suite['val'].filter(
                lambda q: not contains(q['keywords'], test_concepts)
            ),
            'test': raw_suite['test'].filter(
                lambda q: contains(q['keywords'], test_concepts)
            )
        }
        balanced = load_questions.balance_KwAns_suite(biased, logger)
    return balanced


def biased_classify(dataset, test_concepts, args, logger):
    logger('Biasing classify dataset according to test concepts')
    with logger.levelup():
        raw_suite = raw_classify_hypernym(dataset, args, logger)
        biased = {
            'train': load_questions.fewer_bias_clsf(
                raw_suite['train'], test_concepts, args.fewshot_ratio, logger
            ),
            'val': load_questions.fewer_bias_clsf(
                raw_suite['val'], test_concepts, 0, logger
            ),
            'test': load_questions.fewer_bias_clsf(
                raw_suite['test'], test_concepts, 0, logger,
                reverse=True
            )
        }
    return biased
