#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : sceneGraph_loader.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# utility functions for loading sceneGraphs

import os
import json
import pickle
from glob import glob
from functools import reduce

from . import image_utils
from .sceneGraph_utils import id_from_fileName
from utility.common import union


#  main function for loading sceneGraphs


def load_raw_sceneGraphs(filename, process_fn, logger, args):
    """
    Loading the sceneGraphs from raw file
    """
    logger(f'Loading file: {filename}')
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            loaded = json.load(f)

    logger('Post-processing')
    with logger.levelup():
        processed = process_fn(loaded, filename, logger, args)
        for scene in processed.values():
            scene['info'] = {}
    logger(f'size = {len(processed)}', resume=True)
    return processed


def merge_sceneGraphs(x, y):
    sceneGraphs = {
        image_id: x.get(image_id, y.get(image_id, None))
        for image_id in union(x.keys(), y.keys(), as_set=True)
    }
    return sceneGraphs


def load_multiple_sceneGraphs(path, args, logger, process_fn):
    """
    Loading sceneGraphs from raw files. This function is for building dataset
    only.
    """

    all_files = union(glob(os.path.join(path, '*.pkl')),
                      glob(os.path.join(path, '*.json')))
    logger('Loading the sceneGraph files:')
    logger(all_files, pretty=True, resume=True)
    sceneGraph_list = [
        load_raw_sceneGraphs(
            filename, process_fn, logger, args)
        for filename in all_files
    ]
    logger('Merging sceneGraphs')
    merged = reduce(merge_sceneGraphs, sceneGraph_list)
    logger(f'Matching image file names')
    with logger.levelup():
        match_images(merged, args.image_dir, logger)
    return merged


# matching image file names


def match_images(sceneGraphs, image_dir, logger):
    all_imageNames = image_utils.get_all_imageNames(image_dir)
    logger(f'Image file names from {image_dir}: num = {len(all_imageNames)}')

    counter = 0
    name2id = {id_from_fileName(scene['image_filename']): image_id
               for image_id, scene in sceneGraphs.items()}
    for imageName in all_imageNames:
        image_id = id_from_fileName(imageName)

        if image_id in name2id:
            sceneGraphs[name2id[image_id]]['image_filename'] = imageName
            counter += 1

    logger(f'{counter} out of {len(sceneGraphs)} scenes matched',
           resume=True)
