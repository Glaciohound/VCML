#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : eval-classification.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/31/2019
#
# This file is part of eval-clevr-instance-retrieval.
# Distributed under terms of the MIT license.

import six
import functools

import numpy as np
import jacinle.io as io
import jacinle.random as random
from jacinle.cli.argument import JacArgumentParser
from jacinle.utils.tqdm import tqdm_gofor, get_current_tqdm
from jacinle.utils.meter import GroupMeters

parser = JacArgumentParser()
parser.add_argument('--scene-json', required=True, type='checked_file')
parser.add_argument('--preds-json', required=True, type='checked_file')
args = parser.parse_args()


class Definition(object):
    annotation_attribute_names = ['color', 'material', 'shape', 'size']
    annotation_relation_names = ['behind', 'front', 'left', 'right']
    concepts = {
        'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
        'material': ['rubber', 'metal'],
        'shape': ['cube', 'sphere', 'cylinder'],
        'size': ['small', 'large']
    }
    concept2attribute = {
        v: k for k, vs in concepts.items() for v in vs
    }
    relational_concepts = {
        'spatial_relation': ['left', 'right', 'front', 'behind']
    }
    synonyms = {
        "thing": ["thing", "object"],
        "sphere": ["sphere", "ball", "spheres", "balls"],
        "cube": ["cube", "block", "cubes", "blocks"],
        "cylinder": ["cylinder", "cylinders"],
        "large": ["large", "big"],
        "small": ["small", "tiny"],
        "metal": ["metallic", "metal", "shiny"],
        "rubber": ["rubber", "matte"],
    }

    word2lemma = {
        v: k for k, vs in synonyms.items() for v in vs
    }


def_ = Definition()

def transpose_scene(scene):
    ret = dict()
    for k in scene['0']:
        ret[k] = np.array([scene[str(o)][k] for o in range(len(scene))])
    return ret

def main():
    scenes = io.load_json(args.scene_json)['scenes']
    preds = io.load(args.preds_json)
    if isinstance(preds, dict):
        preds = list(preds.values())
    if False:
        preds = [transpose_scene(s) for s in preds]
    meter = GroupMeters()

    flattened_objs = [o for s in scenes for o in s['objects']]
    flattened_preds = {
        k: np.concatenate([np.array(p[k]) for p in preds], axis=0)
        for k in preds[0]
    }

    for k, preds in flattened_preds.items():
        kk = def_.word2lemma.get(k, k)
        for i, o in tqdm_gofor(flattened_objs, desc='{}(lemma: {})'.format(k, kk), leave=False):
            meter.update('acc', (preds[i] > 0) == (kk == o[def_.concept2attribute[kk]]))
            meter.update(f'acc/{k}', (preds[i] > 0) == (kk == o[def_.concept2attribute[kk]]))
    print(meter.format_simple('Results:', compressed=False))

if __name__ == '__main__':
    main()
