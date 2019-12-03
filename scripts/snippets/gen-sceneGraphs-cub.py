#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen-sceneGraphs-cub.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 05.08.2019
# Last Modified Date: 09.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#  This file describes code used for generating sceneGraph for CUB-birds
#  dataset

import os
import sys
import numpy as np
from IPython.core import ultratb
from IPython import embed

from utility.common import dump, make_dir

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)


# Backbone functions


def generate_sceneGraphs(
    classes, attributes, parts,
    class_labels, attribute_labels,
    bounding_boxes, part_loc, part_loc_info,
    filenames, train_test_split,
):
    print('Generating sceneGraphs')
    num = len(filenames)
    n_attributes = len(attributes)
    attribute_matrix = attribute_labels[:, 2].reshape((num, n_attributes))
    confidence_matrix = attribute_labels[:, 3].reshape((num, n_attributes))
    sceneGraphs = []
    for i in range(num):
        this_attributes = attribute_matrix[i]
        this_confidence = confidence_matrix[i]
        objects = [
            {
                'name': classes[class_labels[i]],
                'attributes_uncertain': {
                    'attribute_vector': this_attributes,
                    'confidence_vector': this_confidence,
                },
                'x': bounding_boxes[i, 0],
                'y': bounding_boxes[i, 1],
                'w': bounding_boxes[i, 2],
                'h': bounding_boxes[i, 3],
            }
        ]
        filename = filenames[i]
        scene = {
            'image_index': i,
            'image_id': get_image_id(filename),
            'relationships': {},
            'image_filename': filename,
            'split': train_test_split[i],
            'objects': objects,
        }
        sceneGraphs.append(scene)

    return sceneGraphs


def main(
    bird_class_txt, bird_attribute_txt, bird_parts_txt,
    class_labels_txt, attribute_labels_txt,
    bounding_boxes_txt, part_loc_txt,
    filename_txt, image_dir, train_test_split_txt,
    sceneGraph_dir,
):
    print('Main function')
    classes = read_names(bird_class_txt)
    classes = {i+1: name for i, name in enumerate(classes)}
    attributes = read_names(bird_attribute_txt)
    parts = read_names(bird_parts_txt)

    class_labels = read_numbers(class_labels_txt, 1, int)
    attribute_labels = read_numbers(attribute_labels_txt, [0, 1, 2, 3], int)
    bounding_boxes = read_numbers(bounding_boxes_txt, [1, 2, 3, 4], float)
    part_loc = read_numbers(part_loc_txt, [2, 3], float)
    part_loc_info = read_numbers(part_loc_txt, [0, 1, 4], int)

    filenames = read_filenames(filename_txt, image_dir)
    train_test_split = read_numbers(train_test_split_txt, 1, int)
    train_test_split = parse_train_test_split(train_test_split)

    sceneGraphs = generate_sceneGraphs(
        classes, attributes, parts,
        class_labels, attribute_labels,
        bounding_boxes, part_loc, part_loc_info,
        filenames, train_test_split
    )
    train, _, test = split_sceneGraphs(sceneGraphs)
    train_wrap = wrap_sceneGraphs('train', train, attributes)
    test_wrap = wrap_sceneGraphs('test', test, attributes)
    train_pkl = os.path.join(sceneGraph_dir, 'CUB_train_scenes.pkl')
    test_pkl = os.path.join(sceneGraph_dir, 'CUB_test_scenes.pkl')
    make_dir(sceneGraph_dir)

    dump(train_wrap, train_pkl)
    dump(test_wrap, test_pkl)

    embed()


"""
Some utility functions
"""


def read_names(txt_file):
    with open(txt_file, 'r') as f:
        read_lines = f.readlines()

    names = [line.lstrip('0123456789. ').rstrip('\n')
             for line in read_lines]
    return names


def read_numbers(txt_file, select_columns=None, dtype=float):
    with open(txt_file, 'r') as f:
        read_lines = f.readlines()

    data = np.array([
        np.array([
            float(num) for num in line.rstrip('\n').split(' ')
            if num != ''
        ])[select_columns]
        for line in read_lines
    ])
    data = data.astype(dtype)
    return data


def read_filenames(txt_file, image_dir):
    with open(txt_file, 'r') as f:
        read_lines = f.readlines()

    filenames = [line.split(' ')[-1].rstrip('\n') for line in read_lines]
    filenames = list(map(
        lambda s: os.path.join(image_dir, s),
        filenames
    ))
    return filenames


def get_image_id(filename):
    return filename.split('/')[-1].split('.')[0]


def parse_train_test_split(train_test_split):
    split_index = ['train', 'test']
    split = [split_index[ind] for ind in train_test_split]
    return split


def split_sceneGraphs(sceneGraphs):
    splits = ('train', 'val', 'test')
    splits = [
        [scene for scene in sceneGraphs
         if scene['split'] == _split]
        for _split in splits
    ]
    return splits


def wrap_sceneGraphs(split, sceneGraphs, attributes):
    doc_string = (
        'Self-made sceneGraph dataset for CUB-200-2011'
        'dataset. This files belongs to {split} slplit.'
    )
    output = {
        'info': doc_string,
        'scenes': sceneGraphs,
        'attributes': attributes,
    }
    return output


# main program
if __name__ == '__main__':
    print('Program Starts')

    root_dir = '.'
    # relative path
    data_dir = '../data/cub'

    bird_class_txt = 'raw/classes.txt'
    bird_attribute_txt = 'raw/attributes/attributes.txt'
    bird_parts_txt = 'raw/parts/parts.txt'

    class_labels_txt = 'raw/image_class_labels.txt'
    attribute_labels_txt = 'raw/attributes/image_attribute_labels.txt'
    part_loc_txt = 'raw/parts/part_locs.txt'

    filename_txt = 'raw/images.txt'
    image_dir = 'raw/images'
    train_test_split_txt = 'raw/train_test_split.txt'
    bounding_boxes_txt = 'raw/bounding_boxes.txt'
    sceneGraph_dir = 'processed/sceneGraphs/'

    # get absolute path
    data_dir = os.path.join(root_dir, data_dir)

    bird_class_txt = os.path.join(data_dir, bird_class_txt)
    bird_attribute_txt = os.path.join(data_dir, bird_attribute_txt)
    bird_parts_txt = os.path.join(data_dir, bird_parts_txt)

    class_labels_txt = os.path.join(data_dir, class_labels_txt)
    attribute_labels_txt = os.path.join(data_dir, attribute_labels_txt)
    bounding_boxes_txt = os.path.join(data_dir, bounding_boxes_txt)
    part_loc_txt = os.path.join(data_dir, part_loc_txt)

    filename_txt = os.path.join(data_dir, filename_txt)
    image_dir = os.path.join(data_dir, image_dir)
    train_test_split_txt = os.path.join(data_dir, train_test_split_txt)
    sceneGraph_dir = os.path.join(data_dir, sceneGraph_dir)

    # run main function
    main(
        bird_class_txt, bird_attribute_txt, bird_parts_txt,
        class_labels_txt, attribute_labels_txt,
        bounding_boxes_txt, part_loc_txt,
        filename_txt, image_dir, train_test_split_txt,
        sceneGraph_dir,
    )
