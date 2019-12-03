#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : load_normal.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 13.08.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from ..utils import load_questions

# Conceptual questions: Synonym and Same-kind


def synonym_questions(dataset, args, logger):
    syn_questions = load_questions.load_question_file(
        dataset, 'synonym', args, logger
    )
    return syn_questions


def raw_synonym(dataset, args, logger):
    logger('Identical synonym suite')
    with logger.levelup():
        syn_questions = synonym_questions(dataset, args, logger)
        syn_full = load_questions.identical_suite(syn_questions, logger)
    return syn_full


def synonym_balanced_split(test_concepts, dataset, args, logger):
    logger('Loading balaced synonym suite splitted by test_concepts')
    with logger.levelup():
        syn_suite = raw_synonym(dataset, args, logger)
        syn_suite = load_questions.split_testConcepts(
            syn_suite, test_concepts, logger)
        syn_suite = load_questions.balance_KwAns_suite(syn_suite, logger)
    return syn_suite


def synonym_balanced_full(dataset, args, logger):
    logger('Loading a balanced full synonym dataset')
    with logger.levelup():
        syn_suite = raw_synonym(dataset, args, logger)
        syn_suite = load_questions.balance_KwAns_suite(syn_suite, logger)
    return syn_suite


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
    logger('Loading a balanced full samekind dataset')
    with logger.levelup():
        sk_suite = raw_samekind(dataset, args, logger)
        sk_suite = load_questions.balance_KwAns_suite(sk_suite, logger)
    return sk_suite


# Visual quesitons: Classify and Exist


def raw_classify(dataset, args, logger):
    logger('Loading CLEVR/GQA dataset classify suite')
    with logger.levelup():
        raw_suite = {
            'train': load_questions.load_question_file(
                dataset, 'train_classification', args, logger
            ),
            'val': load_questions.load_question_file(
                dataset, 'val_classification', args, logger
            ),
            'test': load_questions.load_question_file(
                dataset, 'test_classification', args, logger
            ),
        }
    return raw_suite


def raw_exist(dataset, args, logger):
    logger('Loading exist suite')
    with logger.levelup():
        raw_suite = {
            'train': load_questions.load_question_file(
                dataset, 'train_exist', args, logger
            ),
            'val': load_questions.load_question_file(
                dataset, 'val_exist', args, logger
            ),
            'test': load_questions.load_question_file(
                dataset, 'test_exist', args, logger
            ),
        }
    return raw_suite


def balanced_exist(dataset, args, logger):
    logger('Building a balanced exist suite')
    raw_suite = raw_exist(dataset, args, logger)
    balanced_suite = load_questions.balance_KwAns_suite(raw_suite, logger)
    return balanced_suite


def classify_zeroshot(dataset, test_concepts, args, logger):
    logger('Zero-outing a classify dataset according to test concepts')
    with logger.levelup():
        zeroshot_suite = {
            'train': load_questions.fewer_bias_clsf(
                load_questions.load_question_file(
                    dataset, 'train_classification', args, logger
                ),
                test_concepts, 0, logger
            ),
            'val': load_questions.balance_KwAns_dataset(
                load_questions.filter_kw_dataset(
                    load_questions.load_question_file(
                        dataset, 'val_exist', args, logger
                    ),
                    test_concepts, logger, reverse=True
                ), logger
            ),
            'test': load_questions.balance_KwAns_dataset(
                load_questions.filter_kw_dataset(
                    load_questions.load_question_file(
                        dataset, 'test_exist', args, logger
                    ),
                    test_concepts, logger, reverse=False
                ), logger
            )
        }
    return zeroshot_suite


def exist_zeroshot(dataset, test_concepts, args, logger):
    logger('Zero-outing an exist dataset according to test concepts')
    with logger.levelup():
        zeroshot_suite = {
            'train': load_questions.balance_KwAns_dataset(
                load_questions.filter_kw_dataset(
                    load_questions.load_question_file(
                        dataset, 'train_exist', args, logger
                    ),
                    test_concepts, logger, reverse=True,
                ), logger
            ),
            'val': load_questions.balance_KwAns_dataset(
                load_questions.filter_kw_dataset(
                    load_questions.load_question_file(
                        dataset, 'val_exist', args, logger
                    ),
                    test_concepts, logger, reverse=True
                ), logger
            ),
            'test': load_questions.balance_KwAns_dataset(
                load_questions.filter_kw_dataset(
                    load_questions.load_question_file(
                        dataset, 'test_exist', args, logger
                    ),
                    test_concepts, logger, reverse=False
                ), logger
            )
        }
    return zeroshot_suite
