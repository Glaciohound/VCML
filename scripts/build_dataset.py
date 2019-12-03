#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : build_dataset.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 11.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file contains codes for building base datasets

import os
import sys
from IPython import embed
from IPython.core import ultratb

from utility.common import make_dir, dump, load_knowledge
from utility.logging import Logger
from dataset.question_dataset.utils import misc
from dataset.question_dataset.build_questions.question_configs import \
    register_builders
from experiments import lazy_import
from dataset.question_dataset.build_questions.build_questions import \
    build_all_visual_questions, \
    build_all_conceptual_questions

from scripts.utils import prepare
from scripts.utils import register

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)


# Main function

def build_dataset(args):
    """
    Main function for building question dataset
    """

    local_dir = os.path.join(args.dataset_dir, args.name)
    make_dir(local_dir)
    logger = Logger(local_dir, is_main=True)
    prepare.print_args(args, logger)
    dataset_config = args.dataset_config[args.task]
    config = lazy_import()[args.mode, args.task, args.experiment].config

    # loading visual dataset
    logger('Loading dataset')
    with logger.levelup():
        visual_dataset = prepare.load_visual_dataset(
            args, logger,
            dataset_config['scene_process']
        )

    logger('Splittinng dataset')
    with logger.levelup():
        # task-specific visual split
        visual_splits = dataset_config['visual_split_fn'](
            visual_dataset, logger, args)
        visual_dataset.mark_splits(
            get_split_indexes(visual_splits))
        # experiment-specific visual split
        visual_splits = config['visual_split_fn'](
            visual_dataset, logger, args, **config['split_kwarg'])
        split_indexes = get_split_indexes(visual_splits)

    # Registering concepts, building exist-checktable
    tools = register.init_word2index(logger)
    logger('Registering visual concepts')
    with logger.levelup():
        register.register_visual_concepts(
            visual_dataset, tools.concepts, args,
            config['register_synonyms'], config['register_hypernyms'],
            config['register_meronyms'],
            load_knowledge(args.task, 'forbidden'),
            logger, args.experiment
        )
        logger(f'Num of concepts: {len(tools.concepts)}')
    logger('Building exist-checktable')
    with logger.levelup():
        exist_checktable = misc.exist_checktable(
            tools.concepts, args, logger
        )

    # building conceptual and visual questions
    builders = register_builders(args, tools.concepts, config)

    logger('Building conceptual questions')
    with logger.levelup():
        conceptual_questions = build_all_conceptual_questions(
            args, builders, tools.concepts,
            config['conceptual_question_types'],
            logger
        )

    logger('Building visual questions')
    with logger.levelup():
        visual_questions = build_all_visual_questions(
            args, config, builders, tools.concepts,
            visual_splits, config['visual_question_types'],
            exist_checktable, logger
        )

    # registering question tokens
    iter_conceptual = list(
        q
        for questions in conceptual_questions.values()
        for q in questions
    )
    iter_visual = list(
        q
        for one_split in visual_questions.values()
        for questions in one_split.values()
        for q in questions
    )
    register.register_question_token(iter_conceptual, tools, logger)
    register.register_question_token(iter_visual, tools, logger)

    # save
    logger('Saving')
    with logger.levelup():
        save(local_dir, logger,
             conceptual_questions, visual_questions,
             visual_dataset.sceneGraphs,
             tools, split_indexes)

    embed()


def save(local_dir, logger,
         conceptual_questions, visual_questions, sceneGraphs,
         tools, split_indexes):

    split_filename = os.path.join(local_dir, 'visual_split.pkl')
    logger(f'saving split indexes to : {split_filename}')
    dump(split_indexes, split_filename)

    sg_filename = os.path.join(local_dir, 'sceneGraphs.pkl')
    logger(f'saving scene-graphs to : {sg_filename}')
    dump(sceneGraphs, sg_filename)

    for split, split_dataset in visual_questions.items():
        for q_type, type_questions in split_dataset.items():
            filename = os.path.join(local_dir, f'{split}_{q_type}.pkl')
            logger(f'saving visual questions: {filename}')
            dump(type_questions, filename)

    for q_type, type_questions in conceptual_questions.items():
        filename = os.path.join(local_dir, f'{q_type}.pkl')
        logger(f'saving conceptual questions: {filename}')
        dump(type_questions, filename)

    tools.save(local_dir)


def get_split_indexes(visual_dataset_dict):
    output = {
        split: dataset.indexes
        for split, dataset in visual_dataset_dict.items()
    }
    return output
