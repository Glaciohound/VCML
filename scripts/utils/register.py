#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : register.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from utility.common import load_knowledge, all_secondary_elements
from utility import word_index
from dataset.question_dataset.utils import \
    question_utils, program_utils


Word_Index = word_index.Word_Index


def register_question_token(
        question_iter, tools, logger
):
    logger('Registering question datasets')
    for q in logger.tqdm(question_iter):
        tools.metaconcepts.register(['synonym', 'hypernym',
                                     'samekind', 'meronym',
                                     'isinstanceof'],
                                    multiple=True)
        for w in question_utils.question2seq(q['question']):
            tools.words.register(w)

        program = program_utils.semantic2program(q['semantic'])
        for op in program:
            tools.operations.register(op['operation'])
            tools.arguments.register(op['argument'])
            if op['operation'] == 'query':
                tools.attributes.register(op['argument'])
            if op['operation'] in ['filter', 'select_concept',
                                   'synonym', 'hypernym', 'samekind',
                                   'meronym', 'classify']:
                assert op['argument'] in tools.concepts
        if q['type'] == 'isinstanceof':
            tools.attributes.register(q['answer'])

        answer = q['answer']
        if isinstance(answer, str):
            tools.answers.register(answer)


def register_visual_concepts(
        visual_dataset, concepts, args,
        register_synonyms, register_hypernyms, register_meronyms,
        forbidden_concepts,
        logger, experiment_name,
):

    # registering from visual and conceptual datasets
    if forbidden_concepts is not None:
        logger('Filtering out forbidden concepts:')
        logger(forbidden_concepts, resume=True, pretty=True)
        concepts.filter_out(forbidden_concepts)

    logger('Registering visual dataset')
    visual_dataset.register_concepts(concepts)
    if args.most_frequent != -1:
        word_count = load_knowledge(args.task, 'most_common', logger)
        concepts.filter_most_frequent(word_count, args.most_frequent)
        # visual_dataset.filter_concepts(concepts)

    # registering from knowledge
    if register_synonyms:
        logger('Registering synonyms')
        with logger.levelup():
            concepts.register_related(
                load_knowledge(args.task, 'synonym')
            )
    if register_hypernyms:
        logger('Registering hypernyms')
        with logger.levelup():
            concepts.register_related(
                load_knowledge(args.task, 'hypernym')
            )
    if register_meronyms:
        logger('Registering meronyms')
        with logger.levelup():
            concepts.register(
                all_secondary_elements(load_knowledge(args.task, 'meronym')),
                multiple=True,
            )

    return concepts


def init_word2index(logger):
    kit_config = {
        'words': {'use_special_tokens': True},
        'operations': {'use_special_tokens': True},
        'arguments': {'use_special_tokens': True},
        'answers': {'use_special_tokens': False},

        'concepts': {'use_special_tokens': False},
        'attributes': {'use_special_tokens': False},
        'metaconcepts': {'use_special_tokens': False},
    }

    tools = word_index.Tools(kit_config, logger)
    return tools
