#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : load_cub_meronym.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 07.08.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from ..utils import load_questions


def meronym_questions(dataset, args, logger):
    mero_questions = load_questions.load_question_file(
        dataset, 'meronym_cub', args, logger
    )
    return mero_questions


def raw_meronym(dataset, args, logger):
    logger('Identical meronym suite')
    with logger.levelup():
        mero_questions = meronym_questions(dataset, args, logger)
        mero_full = load_questions.identical_suite(mero_questions, logger)
    return mero_full


def meronym_balanced_split(test_concepts, dataset, args, logger):
    logger('Loading balaced meronym suite splitted by test_concepts')
    with logger.levelup():
        mero_suite = raw_meronym(dataset, args, logger)
        mero_suite = load_questions.split_testConcepts(
            mero_suite, test_concepts, logger)
        mero_suite = load_questions.balance_KwAns_suite(
            mero_suite, logger,
            balance_fn=lambda q: (q['keywords'][1], q['answer'])
        )
    return mero_suite


def meronym_balanced_full(dataset, args, logger):
    logger('Loading a full meronym suite')
    with logger.levelup():
        mero_suite = raw_meronym(dataset, args, logger)
        balanced = load_questions.balance_KwAns_suite(mero_suite, logger)
    return balanced


# Visual part


def raw_classify_meronym(dataset, args, logger):
    logger('Loading classify-meronym suite')
    with logger.levelup():
        raw_suite = {
            'train': load_questions.load_question_file(
                dataset, 'train_cub_classification_meronym', args, logger
            ),
            'val': load_questions.load_question_file(
                dataset, 'val_cub_classification_meronym', args, logger
            ),
            'test': load_questions.load_question_file(
                dataset, 'test_cub_classification_meronym', args, logger
            ),
        }
    return raw_suite


def raw_exist_meronym(dataset, args, logger):
    logger('Loading exist-meronym suite')
    with logger.levelup():
        raw_suite = {
            'train': load_questions.load_question_file(
                dataset, 'train_cub_exist_meronym', args, logger
            ),
            'val': load_questions.load_question_file(
                dataset, 'val_cub_exist_meronym', args, logger
            ),
            'test': load_questions.load_question_file(
                dataset, 'test_cub_exist_meronym', args, logger
            ),
        }
    return raw_suite


def balanced_exist(dataset, args, logger):
    logger('Loading a balanced exist suite')
    raw_suite = raw_exist_meronym(dataset, args, logger)
    balanced_suite = load_questions.balance_KwAns_suite(raw_suite, logger)
    return balanced_suite
