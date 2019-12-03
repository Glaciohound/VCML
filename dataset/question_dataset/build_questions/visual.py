#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visual.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 24.07.2019
# Last Modified Date: 24.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# Visual Questions & Answering questions.


import numpy as np
from collections import defaultdict

from utility.common import \
    union, random_one, intersection
from ..utils.misc import \
    filter_objects, cub_proper_group_name
from dataset.visual_dataset.utils.sceneGraph_utils import \
    get_classification


def classification_task(
        scene, analysis, concepts, **kwarg):
    """
    Produce classification QAs given a scene. This function is for those with
    multiple objects, so that attributes drawn from all objects in the scene
    is enough to balance the True/False ratio of classification items. If
    there happens to be only one object in the image, only one concept from
    outside will be added to slightly balance the True/False ratio.
    """

    concepts_to_classify = list(union(
        *tuple(item.keys() for item in analysis.values()),
        as_set=True,
    ))
    concepts_to_classify = intersection(concepts_to_classify,
                                        concepts)
    if len(concepts_to_classify) == 0:
        return None

    result, confidence = get_classification(analysis, concepts_to_classify)

    question = {
        'question':
        'Please classify objects in the image according to : '
        '{}.'.format(', '.join(concepts_to_classify)),
        'semantic': [
            {'operation': 'classify',
             'argument': name,
             'dependencies': []}
            for name in concepts_to_classify
        ],
        'answer': result,
        'confidence': confidence,
        'type': 'classification',
        'category': 'visual',
    }
    return question


def exist_proto(scene, analysis, concepts, balance, question_fn, **kwarg):
    """
    Builds simple 'Are there any *** objects in the image?' question
    """
    if 'merged' not in scene['info']:
        merged = merge_analysis(analysis, concepts)
        scene['info']['merged'] = merged
    else:
        merged = scene['info']['merged']

    if balance and (len(merged[True]) == 0 or len(merged[False]) == 0):
        return None

    if (random_one((True, False)) is True and len(merged[True]) > 0) or\
            len(merged[False]) == 0:
        queried = random_one(list(merged[True].keys()))
        answer = 'yes'
    else:
        count = list(merged[False].values())
        p = (np.array(count) / sum(count)).tolist()
        queried = random_one(list(merged[False].keys()), p=p)
        answer = 'no'

    which, _ = filter_objects(scene, queried)

    question = {
        'question': question_fn(queried),
        'semantic': [
            {'operation': 'select',
             'argument': '{0} ({1})'.format(queried, ', '.join(which)),
             'dependencies': []},

            {'operation': 'exist', 'argument': '?',
             'dependencies': [0]}
        ],

        'keywords': (queried,),
        'answer': answer,
        'type': 'exist',
        'category': 'visual',
        'confidence': 1,
    }

    return question


def exist(scene, analysis, concepts, balance, **kwarg):
    question = exist_proto(scene, analysis, concepts, balance,
                           normal_exist_question)
    return question


def cub_exist(scene, analysis, concepts, balance, **kwarg):
    question = exist_proto(scene, analysis, concepts, balance,
                           cub_exist_question)
    return question


def normal_exist_question(queried):
    question = f'Are there any {queried} objects in the image?'
    return question


def cub_exist_question(queried):
    rename = cub_proper_group_name(queried, in_plural=True)
    question = f'Are there any {rename} in the image?'
    return question


def merge_analysis(analysis, concepts):
    counter = defaultdict(lambda: {True: 0, False: 0})
    for obj in analysis.values():
        for name, value in obj.items():
            if name in concepts:
                counter[name][value] += 1
    for name in concepts:
        counter[name][False] += 1
    output = {True: {}, False: {}}
    for name, count in counter.items():
        if count[True] > 0:
            output[True][name] = count[True]
        else:
            output[False][name] = count[False]
    return output
