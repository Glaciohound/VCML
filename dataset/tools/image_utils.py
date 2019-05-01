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

import jaclearn.vision.coco.mask_utils as mask_utils

def _is_object_annotation_available(scene):
    objects = scene['objects']
    if isinstance(objects, dict):
        objects = list(objects.values())
    if len(scene['objects']) > 0 and 'mask' in objects[0]:
        return True
    return False


def _get_object_masks(scene):
    """Backward compatibility: in self-generated clevr scenes, the groundtruth masks are provided;
    while in the clevr test data, we use Mask R-CNN to detect all the masks, and stored in `objects_detection`."""
    if 'objects_detection' not in scene:
        return scene['objects']
    if _is_object_annotation_available(scene):
        return scene['objects']
    return scene['objects_detection']


def annotate_objects(scene, from_shape=None, to_shape=None):
    if 'objects' not in scene and 'objects_detection' not in scene:
        return dict()

    boxes = [mask_utils.toBbox(i['mask']) for i in _get_object_masks(scene)]
    if len(boxes) == 0:
        return {'objects': np.zeros((0, 4), dtype='float32')}
    boxes = np.array(boxes)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes = boxes.astype('float32')
    if from_shape and to_shape:
        ratio = min(to_shape[1] / from_shape[1], to_shape[0] / from_shape[0])
        boxes *= ratio
    return {'objects': boxes}

def get_imageNames(path):
    all_filenames = glob(os.path.join(path, '**'), recursive=True)
    all_filenames = [name for name in all_filenames if
                     name.endswith('jpg') or name.endswith('png')]
    return all_filenames
