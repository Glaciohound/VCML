#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : question_configs.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 05.08.2019
# Last Modified Date: 31.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from utility.common import \
    load_knowledge, all_elements, all_concepts,\
    all_secondary_elements

from .conceptual import \
    hypernym_question, hypernym_question_cub, \
    meronym_question_cub, \
    synonym_question, isinstanceof_question, samekind_question

from .visual import \
    classification_task, exist, cub_exist


def register_builders(args, concepts, config):
    def task_knowledge(name):
        return load_knowledge(args.task, name)

    subtasks = config['visual_question_types'] + \
        config['conceptual_question_types']
    builders = {}

    for sub in subtasks:
        # ------- Visual Tasks ---------

        # conventional visual tasks
        if sub == 'exist':
            builders[sub] = {
                'type': 'visual', 'builder_fn': exist,
                'concepts': concepts,
                'questionsPimage': args.questionsPimage,
                'kwarg': {'balance': True},
             }

        elif sub == 'biased_exist':
            builders[sub] = {
                'type': 'visual', 'builder_fn': exist,
                'concepts': list(config['default_bias'][args.task].keys()),
                'questionsPimage': args.questionsPimage,
                'kwarg': {'balance': False},
             }

        elif sub == 'classification':
            builders[sub] = {
                'type': 'visual', 'builder_fn': classification_task,
                'concepts': concepts,
                'questionsPimage': 1,
                'kwarg': {},
            }

        # visual tasks for cub-attribute type datasets
        elif sub == 'cub_classification_hypernym':
            builders[sub] = {
                'type': 'visual', 'builder_fn': classification_task,
                'concepts': all_concepts(task_knowledge('hypernym')),
                'questionsPimage': 1,
                'kwarg': {}
            }

        elif sub == 'cub_classification_meronym':
            builders[sub] = {
                'type': 'visual', 'builder_fn': classification_task,
                'concepts': all_secondary_elements(
                    task_knowledge('meronym'), with_keys=True),
                'questionsPimage': 1,
                'kwarg': {}
            }

        elif sub == 'cub_exist_hypernym':
            builders[sub] = {
                'type': 'visual', 'builder_fn': cub_exist,
                'concepts': all_concepts(task_knowledge('hypernym')),
                'questionsPimage': args.questionsPimage,
                'kwarg': {'balance': True}
            }

        elif sub == 'cub_exist_meronym':
            builders[sub] = {
                'type': 'visual', 'builder_fn': cub_exist,
                'concepts': all_secondary_elements(
                    task_knowledge('meronym'), with_keys=True),
                'questionsPimage': args.questionsPimage,
                'kwarg': {'balance': True}
            }

        # ------- Conceptual Tasks ---------
        # conceptual tasks

        elif sub == 'synonym':
            builders[sub] = {
                'type': 'conceptual', 'builder_fn': synonym_question,
                'subject': all_concepts(task_knowledge('synonym'),
                                        lambda k: k in concepts),
                'object': all_concepts(task_knowledge('synonym'),
                                       lambda k: k in concepts),
                'knowledges': [task_knowledge('synonym')],
             }

        elif sub == 'isinstanceof':
            builders[sub] = {
                'type': 'conceptual', 'builder_fn': isinstanceof_question,
                'subject': concepts,
                'object': task_knowledge('isinstanceof').keys(),
                'knowledges': [task_knowledge('isinstanceof'),
                               task_knowledge('synonym')],
            }

        elif sub == 'samekind':
            builders[sub] = {
                'type': 'conceptual', 'builder_fn': samekind_question,
                'subject': concepts,
                'object': concepts,
                'knowledges': [task_knowledge('isinstanceof'),
                               task_knowledge('synonym')],
            }

        elif sub == 'hypernym':
            builders[sub] = {
                'type': 'conceptual', 'builder_fn': hypernym_question,
                'subject': concepts,
                'object': all_elements(task_knowledge('hypernym'),
                                       lambda k: k in concepts),
                'knowledges': [task_knowledge('hypernym'),
                               task_knowledge('synonym')],
            }

        elif sub == 'hypernym_cub':
            builders[sub] = {
                'type': 'conceptual', 'builder_fn': hypernym_question_cub,
                'subject': all_concepts(task_knowledge('hypernym')),
                'object': all_concepts(task_knowledge('hypernym')),
                'knowledges': [task_knowledge('hypernym'),
                               task_knowledge('synonym')],
            }

        elif sub == 'meronym_cub':
            builders[sub] = {
                'type': 'conceptual', 'builder_fn': meronym_question_cub,
                'subject': all_secondary_elements(
                    task_knowledge('meronym')),
                'object': list(task_knowledge('meronym').keys()),
                'knowledges': [task_knowledge('meronym')],
            }

        else:
            raise Exception(f'No such sub-task defined: {sub}')

    return builders
