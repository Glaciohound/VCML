#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : build_questions.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 06.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# Building VQA question dataset, by calling functions from
# `.question_builders/`

from utility.common import random_choice

from dataset.visual_dataset.utils.sceneGraph_utils import \
    visual_analyze


# Building visual & conceptual question group

def build_all_visual_questions(
        args, config, builders, concepts,
        visual_splits, visual_question_types,
        exist_checktable, logger):

    all_questions = {}
    for split in ['train', 'val', 'test']:
        logger(f'Building Datasets for split {split}')
        all_questions[split] = {}
        visual_dataset = visual_splits[split]
        with logger.levelup():
            for subtask in visual_question_types:
                task_config = builders[subtask]
                logger(f'Building visual task: {subtask}')
                this_visual = build_one_visual(
                    args, config, task_config,
                    concepts, visual_dataset,
                    exist_checktable,
                    logger,
                )
                logger(f'Dataset size: {len(this_visual)}', resume=True)
                all_questions[split][subtask] = this_visual

    return all_questions


def build_all_conceptual_questions(
        args, builders, concepts,
        conceptual_question_types, logger):

    all_questions = {}
    for subtask, config in builders.items():
        if subtask in conceptual_question_types:
            logger(f'Building conceptual task: {subtask}')
            this_conceptual = build_one_conceptual(
                config['builder_fn'],
                config['subject'],
                config['object'],
                config['knowledges'],
                concepts, logger,
            )
            logger(f'Dataset size: {len(this_conceptual)}', resume=True)
            all_questions[subtask] = this_conceptual

    return all_questions


def build_one_visual(
        args, config, task_config, concepts,
        visual_dataset, exist_checktable, logger):
    output = []

    if args.size < len(visual_dataset):
        selected_ids = random_choice(
            visual_dataset.keys(),
            args.size,
        )
    else:
        selected_ids = list(visual_dataset.keys())

    builder_fn = task_config['builder_fn']

    for scene_id in logger.tqdm(selected_ids):
        scene = visual_dataset.sceneGraphs[scene_id]
        split = scene['split']
        analysis = visual_analyze(
            scene, exist_checktable)

        for j in range(task_config['questionsPimage']):
            question = builder_fn(
                scene,
                analysis,
                task_config['concepts'],
                **task_config['kwarg'],
            )
            if question is not None:
                question['image_id'] = scene_id
                question['split'] = split
                output.append(question)

    for scene in visual_dataset.sceneGraphs.values():
        scene['info'] = {}

    return output


def build_one_conceptual(
    builder_fn,
    subject_set,
    object_set,
    knowledges,
    concepts,
    logger,
):
    output = []

    for subj in logger.tqdm(subject_set):
        for obj in object_set:
            if subj in concepts and obj in concepts:
                question = builder_fn(
                    subj, obj, *knowledges)
                if question is not None:
                    output.append(question)
    return output
