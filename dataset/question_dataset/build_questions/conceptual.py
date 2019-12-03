#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : conceptual.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 24.07.2019
# Last Modified Date: 31.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# Functions for building conceptual questions.


from utility.common import \
    is_synonym, is_instance_of, is_hypernym, \
    is_samekind, is_composed_of

from ..utils.misc import \
    cub_proper_group_name, cub_proper_body_name, \
    is_singular, is_greek_or_latin


def hypernym_question(concept1, concept2,
                      hypernym_stats, synonym_stats):
    if is_hypernym(concept2, concept1,
                   hypernym_stats, synonym_stats):
        answer = 'yes'
    else:
        answer = 'no'

    question = {
        'question': 'Is {} a hypernym of {}?'.format(concept2, concept1),
        'semantic': [
            {'operation': 'select_concept', 'argument': format(concept1),
             'dependencies': []},
            {'operation': 'hypernym', 'argument': format(concept2),
             'dependencies': [1]}
        ],

        'keywords': (concept1, concept2),
        'answer': answer,
        'type': 'hypernym',
        'category': 'conceptual',
        'confidence': 1,
    }
    return question


def hypernym_question_cub(concept1, concept2,
                          hypernym_stats, synonym_stats):
    if is_hypernym(concept2, concept1,
                   hypernym_stats, synonym_stats):
        answer = 'yes'
    else:
        answer = 'no'

    rename1 = cub_proper_group_name(concept1)
    rename2 = cub_proper_group_name(concept2)
    copula = 'Is' if is_singular(rename2) else 'Are'
    article1 = 'the ' if not is_greek_or_latin(concept1) else ''
    article2 = 'the ' if not is_greek_or_latin(concept2) else ''

    question = {
        'question': '{} {}{} a hypernym of {}{}?'.
        format(copula, article2, rename2, article1, rename1),
        'semantic': [
            {'operation': 'select_concept', 'argument': format(concept1),
             'dependencies': []},
            {'operation': 'hypernym', 'argument': format(concept2),
             'dependencies': [1]}
        ],

        'keywords': (concept1, concept2),
        'answer': answer,
        'type': 'hypernym',
        'category': 'conceptual',
        'confidence': 1,
    }
    return question


def synonym_question(concept1, concept2, synonym_stats):
    if is_synonym(concept1, concept2, synonym_stats):
        answer = 'yes'
    else:
        answer = 'no'

    question = {
        'question': 'Is {} a synonym of {}?'.format(concept2, concept1),
        'semantic': [
            {'operation': 'select_concept', 'argument': format(concept1),
             'dependencies': []},
            {'operation': 'synonym', 'argument': format(concept2),
             'dependencies': [1]}
        ],

        'keywords': (concept1, concept2),
        'answer': answer,
        'type': 'synonym',
        'category': 'conceptual',
        'confidence': 1,
    }
    return question


def isinstanceof_question(concept1, concept2,
                          isinstanceof_stats, synonym_stats):
    # remove concept "types" with only 1 instance
    if not is_instance_of(concept1, concept2,
                          isinstanceof_stats, synonym_stats) or \
            len(isinstanceof_stats.get(concept2, [])) <= 1:
        return None

    question = {
        'question': '{} is an instance of what?'.format(concept1),
        'semantic': [
            {'operation': 'select_concept', 'argument': format(concept1),
             'dependencies': []},
            {'operation': 'isinstanceof', 'argument': '<NULL>',
             'dependencies': [1]}
        ],

        'keywords': (concept1,),
        'answer': concept2,
        'type': 'isinstanceof',
        'category': 'conceptual',
        'confidence': 1,
    }
    return question


def samekind_question(concept1, concept2,
                      isinstanceof_stats, synonym_stats):
    answer_value = is_samekind(concept1, concept2,
                               isinstanceof_stats, synonym_stats)
    if answer_value is True:
        answer = 'yes'
    elif answer_value is False:
        answer = 'no'
    else:
        return None

    question = {
        'question': 'Are {0} and {1} of the same kind?'.format(
            concept1, concept2),
        'semantic': [
            {'operation': 'select_concept', 'argument': format(concept1),
             'dependencies': []},
            {'operation': 'samekind', 'argument': format(concept2),
             'dependencies': [1]}
        ],

        'keywords': (concept1, concept2),
        'answer': answer,
        'type': 'samekind',
        'category': 'conceptual',
        'confidence': 1,
    }
    return question


def meronym_question_cub(concept1, concept2, meronym_stats):
    """
    This function produces 'Do sparrows usually have red wings?' kind
    questions. In some cases, if the body-part concepts are too complex, the
    questions are re-phrased in a 'which are' manner, such as:
    'Do sparrow usually have bills which are longer than heads?'
    """
    answer = {True: 'yes', False: 'no', None: None}[
        is_composed_of(concept2, concept1, meronym_stats)
    ]
    if answer is None:
        return

    rename2 = cub_proper_group_name(concept2, in_plural=True)
    rename1 = cub_proper_body_name(concept1, in_plural=True)

    question = {
        'question': f'Do {rename2} have {rename1}?',
        'semantic': [
            {'operation': 'select_concept', 'argument': format(concept1),
             'dependencies': []},
            {'operation': 'meronym', 'argument': format(concept2),
             'dependencies': [1]}
        ],

        'keywords': (concept1, concept2),
        'answer': answer,
        'type': 'meronym',
        'category': 'conceptual',
        'confidence': 1,
    }
    return question
