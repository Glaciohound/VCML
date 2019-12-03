#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : load_debiasing.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 13.08.2019
# Last Modified Date: 06.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from ..utils import load_questions

# Conceptual questions: Synonym and Same-kind


def samekind_questions(dataset, args, logger):
    sk_questions = load_questions.load_question_file(
        dataset, 'samekind', args, logger
    )
    return sk_questions


def raw_samekind(dataset, args, logger):
    logger('Identical same-kind suite')
    with logger.levelup():
        sk_questions = samekind_questions(dataset, args, logger)
        sk_full = load_questions.identical_suite(sk_questions, logger)
    return sk_full


def samekind_balanced_split(test_concepts, dataset, args, logger):
    logger('Loading balaced same-kind suite splitted by test_concepts')
    with logger.levelup():
        sk_suite = raw_samekind(dataset, args, logger)
        sk_suite = load_questions.split_testConcepts(
            sk_suite, test_concepts, logger)
        sk_suite = load_questions.balance_KwAns_suite(sk_suite, logger)
    return sk_suite


def samekind_balanced_full(dataset, args, logger):
    logger('Loading a balanced samekind suite')
    with logger.levelup():
        sk_suite = raw_samekind(dataset, args, logger)
        balanced = load_questions.balance_KwAns_suite(sk_suite, logger)
    return balanced


# Visual quesitons: Classify, Exist and Biased_Exist


def classify_debiasing(dataset, args, logger):
    logger('Loading classify-debiasing suite')
    with logger.levelup():
        debiasing_suite = {
            'train': load_questions.load_question_file(
                dataset, 'train_classification', args, logger
            ),
            'val': load_questions.load_question_file(
                dataset, 'val_biased_exist', args, logger
            ),
            'test': load_questions.balance_KwAns_dataset(
                load_questions.load_question_file(
                    dataset, 'test_biased_exist', args, logger
                ), logger
            ),
        }
    return debiasing_suite


def exist_debiasing(dataset, args, logger):
    logger('Loading exist-debiasing suite')
    with logger.levelup():
        debiasing_suite = {
            'train': load_questions.balance_KwAns_dataset(
                load_questions.load_question_file(
                    dataset, 'train_exist', args, logger
                ), logger
            ),
            'val': load_questions.balance_KwAns_dataset(
                load_questions.load_question_file(
                    dataset, 'val_biased_exist', args, logger
                ), logger
            ),
            'test': load_questions.balance_KwAns_dataset(
                load_questions.load_question_file(
                    dataset, 'test_biased_exist', args, logger
                ), logger
            )
        }
    return debiasing_suite
