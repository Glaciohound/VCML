#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : sceneGraph_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 05.08.2019
# Last Modified Date: 31.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


from copy import deepcopy
import numpy as np

from .image_utils import match_objects


# task-specific init and post-processing function


def gqa_post_process(sceneGraphs, filename, logger, args):
    file_split = split_from_filename(filename)
    for image_id, scene in sceneGraphs.items():
        # adding scene-object
        scene_obj = {
            'x': 0, 'y': 0,
            'w': scene['width'], 'h': scene['height'],
        }
        scene['objects']['scene_%s' % image_id] = scene_obj

        # getting concepts_contained
        for obj_id, obj in scene['objects'].items():
            concepts = []
            if 'name' in obj:
                concepts.append(obj['name'])
            if 'attributes' in obj:
                concepts += obj['attributes']
            obj['concepts_contained'] = {c: True for c in concepts}

        # id and split
        scene['image_id'] = str(image_id)
        scene['split'] = file_split

        # match objects
        match_objects(scene, inplace=True)
    return sceneGraphs


def clevr_post_process(sceneGraphs, filename, logger, args):
    sceneGraphs = sceneGraphs['scenes']
    sceneGraphs = {
        id_from_fileName(scene['image_filename']): scene
        for scene in sceneGraphs
    }
    for image_id, scene in sceneGraphs.items():
        # id
        scene['image_id'] = str(image_id)
        # objects
        scene['objects'] = {
            str(j): obj
            for j, obj in enumerate(scene['objects'])
        }
        # concepts_contained
        for obj_id, obj in scene['objects'].items():
            concepts = [obj['color'], obj['size'],
                        obj['shape'], obj['material']]
            obj['concepts_contained'] = {c: True for c in concepts}

        # match objects
        match_objects(scene, inplace=True)
    return sceneGraphs


def cub_post_process(sceneGraphs, filename, logger, args):
    attributes = sceneGraphs['attributes']
    sceneGraphs = sceneGraphs['scenes']
    sceneGraphs = {
        scene['split']+'_'+str(i): scene
        for i, scene in enumerate(sceneGraphs)
    }
    for image_id, scene in sceneGraphs.items():
        # id
        scene['image_id'] = str(image_id)
        # objects
        scene['objects'] = {
            str(j): obj
            for j, obj in enumerate(scene['objects'])
        }

        # concepts_contained
        for obj_id, obj in scene['objects'].items():
            obj['concepts_contained'] = {obj['name']: True}
            for k, concept in enumerate(attributes):
                annotation = obj['attributes_uncertain']
                value = annotation['attribute_vector'][k]
                confidence = annotation['confidence_vector'][k]
                if confidence >= args.confidence_th:
                    obj['concepts_contained'][concept] = bool(value)
        # match objects
        match_objects(scene, inplace=True)
    return sceneGraphs


#  setting 'concepts_contained' keyword


def id_from_fileName(filename):
    return filename.rstrip('.jpg').rstrip('.png').split('/')[-1]


def visual_analyze(scene, exist_checktable):
    output = dict()
    results = exist_checktable['results']

    for obj_id, obj in scene['objects'].items():
        output[obj_id] = deepcopy(obj['concepts_contained'])

        for concept, value in obj['concepts_contained'].items():
            if value is True and concept in results:
                for positive in results[concept][True]:
                    output[obj_id][positive] = True
                for negative in results[concept][False]:
                    output[obj_id][negative] = False
    return output


def split_from_filename(filename):
    for split in ['train', 'test', 'val']:
        if split in filename:
            file_split = split
    return file_split


def get_classification(analysis, concepts):
    """
    Get the classification results of the scene on given
    concepts.
    The analysis and alternatively be the obj['concepts_contained']
    """
    result = np.zeros((len(analysis), len(concepts)),
                      dtype=float)
    confidence = result.copy()

    for i, obj in enumerate(analysis.values()):
        for j, attr in enumerate(concepts):
            if attr in obj:
                confidence[i, j] = 1
                result[i, j] = int(obj[attr])
            else:
                confidence[i, j] = 0

    return result, confidence
