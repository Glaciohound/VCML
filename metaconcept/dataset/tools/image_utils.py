# File   : scene_annotation.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/05/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os
from glob import glob
import numpy as np
from copy import deepcopy
from PIL import Image

import jaclearn.vision.coco.mask_utils as mask_utils


def _is_object_mask_available(scene):
    objects = scene['objects']
    if isinstance(objects, dict):
        objects = list(objects.values())
    if len(scene['objects']) > 0 and 'mask' in objects[0]:
        return True
    return False

def _get_object_masks(scene):
    """Backward compatibility: in self-generated clevr scenes, the groundtruth masks are provided;
    while in the clevr test data, we use Mask R-CNN to detect all the masks, and stored in `objects_detection`."""
    if 'objects_detection' in scene:
        return scene['objects_detection']
    elif _is_object_mask_available(scene):
        return [obj['mask'] for obj in scene['objects'].values()]
    else:
        return []

def _get_object_boxes(scene):

    masks = _get_object_masks(scene)
    if masks != []:
        boxes = [mask_utils.toBbox(i['mask']) for i in _get_object_masks(scene)]
    else:
        if 'x' in list(scene['objects'].values())[0]:
            boxes = [[o['x'], o['y'], o['w'], o['h']]
                     for o in scene['objects'].values()]
        else:
            boxes = []

    if boxes != []:
        boxes = np.array(boxes)
        return boxes

    else:
        return {'objects': np.zeros((0, 4), dtype='float32')}

def annotate_objects(scene, from_shape=None, to_shape=None):
    if 'objects' not in scene and 'objects_detection' not in scene:
        return dict()

    boxes = _get_object_boxes(scene)

    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes = boxes.astype('float32')
    if from_shape and to_shape:
        ratio = min(to_shape[1] / from_shape[1], to_shape[0] / from_shape[0])
        boxes *= ratio
    return {'objects': boxes}


def get_imageCrops(scene):
    output = deepcopy(scene)
    image = Image.open(scene['image_filename']).convert('RGB')
    image = np.asarray(image)
    if not _is_object_annotation_available(scene):
        match_objects(scene, inplace=True)
    for obj in output['objects'].values():
        obj['crop'] = mask_utils.decode(obj['mask'])[:, :, None] * image

    return output


def get_imageNames(path):
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
    centers_obj = np.array([scene['objects'][ind]['pixel_coords'][:2] for ind in obj_indexes])
    sqr_distances = np.power(centers_obj[:, None] - centers_mask[None], 2).sum(2)
    closest = sqr_distances.argmin(1)
    for i, ind in enumerate(obj_indexes):
        obj = output['objects'][ind]
        detection_index = closest[i]
        obj['mask'] = scene['objects_detection'][detection_index]['mask']
        obj['bbox'] = boxes[detection_index]

    output['objects_detection'] = np.array(output['objects_detection'])[closest].tolist()
    output['matched'] = True

    return output
