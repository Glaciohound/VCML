#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : image_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 10.05.2018
# Last Modified Date: 02.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


import os
from glob import glob
import numpy as np
from copy import deepcopy
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import jaclearn.vision.coco.mask_utils as mask_utils


def _is_object_mask_available(scene):
    objects = scene['objects']
    if isinstance(objects, dict):
        objects = list(objects.values())
    if len(scene['objects']) > 0 and 'mask' in objects[0]:
        return True
    return False


def _get_object_masks(scene, obj_prior=False):
    """Backward compatibility: in self-generated clevr scenes,
    the groundtruth masks are provided;
    while in the clevr test data, we use Mask R-CNN to detect all the masks,
    and stored in `objects_detection`."""
    if obj_prior:
        check_order = ['object_mask', 'masks']
    else:
        check_order = ['masks', 'object_mask']

    for item in check_order:
        if item == 'object_mask' and \
                _is_object_mask_available(scene):
            return list(scene['objects'].values())

        if item == 'masks' and \
                'objects_detection' in scene:
            return scene['objects_detection']

    # when no masks detected
    return []


def _get_object_boxes(scene, obj_prior=True):

    masks = _get_object_masks(scene, obj_prior)
    if masks != []:
        boxes = [mask_utils.toBbox(i['mask'])
                 for i in masks]
    else:
        if len(scene['objects']) > 0 and\
                'x' in list(scene['objects'].values())[0]:
            boxes = [[o['x'], o['y'], o['w'], o['h']]
                     for o in scene['objects'].values()]
        else:
            boxes = []

    if boxes != []:
        boxes = np.array(boxes)
        return boxes

    else:
        return np.zeros((0, 4), dtype='float32')


def annotate_objects(scene, from_shape=None, to_shape=None,
                     obj_prior=True):
    if 'objects' not in scene and 'objects_detection' not in scene:
        return dict()

    boxes = _get_object_boxes(scene, obj_prior)

    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes = boxes.astype('float32')
    if from_shape is not None and to_shape is not None:
        ratio = min(to_shape[1] / from_shape[1], to_shape[0] / from_shape[0])
        boxes *= ratio
    return boxes


def get_imageCrops(scene):
    output = deepcopy(scene)
    image = Image.open(scene['image_filename']).convert('RGB')
    image = np.asarray(image)
    if not _is_object_mask_available(scene):
        match_objects(scene, inplace=True)
    for obj in output['objects'].values():
        obj['crop'] = mask_utils.decode(obj['mask'])[:, :, None] * image

    return output


def get_all_imageNames(path):
    all_filenames = glob(os.path.join(path, '**'), recursive=True)
    all_filenames = [name for name in all_filenames if
                     name.endswith('jpg') or name.endswith('png')]
    return all_filenames


def match_objects(scene, inplace=False):
    if 'objects' not in scene or 'objects_detection' not in scene:
        return scene
    if not inplace:
        output = deepcopy(scene)
    else:
        output = scene

    boxes = [mask_utils.toBbox(i['mask']) for i in scene['objects_detection']]
    if len(boxes) == 0:
        return {'objects': np.zeros((0, 4), dtype='float32')}

    boxes = np.array(boxes)
    centers_mask = boxes[:, [0, 1]] + boxes[:, [2, 3]] / 2
    obj_indexes = sorted(list(scene['objects'].keys()))
    centers_obj = np.array([
        scene['objects'][ind]['pixel_coords'][:2]
        for ind in obj_indexes
    ])
    sqr_distances = np.power(
        centers_obj[:, None] - centers_mask[None], 2
    ).sum(2)
    closest = sqr_distances.argmin(1)
    for i, ind in enumerate(obj_indexes):
        obj = output['objects'][ind]
        detection_index = closest[i]
        obj['mask'] = scene['objects_detection'][detection_index]['mask']
        obj['bbox'] = boxes[detection_index]

    output['objects_detection'] = np.array(
        output['objects_detection']
    )[closest].tolist()
    output['matched'] = True

    return output
