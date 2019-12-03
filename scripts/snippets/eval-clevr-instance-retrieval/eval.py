#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : eval.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/30/2019
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
        "sphere": ["sphere", "ball"],
        "cube": ["cube", "block"],
        "cylinder": ["cylinder"],
        "large": ["large", "big"],
        "small": ["small", "tiny"],
        "metal": ["metallic", "metal", "shiny"],
        "rubber": ["rubber", "matte"],
    }

    word2lemma = {
        v: k for k, vs in synonyms.items() for v in vs
    }


def_ = Definition()


def get_desc(obj):
    names = [obj[k] for k in def_.annotation_attribute_names]
    for i, n in enumerate(names):
        if n in def_.synonyms:
            names[i] = random.choice_list(def_.synonyms[n])
    return names


def run_desc_obj(obj, desc):
    for d in desc:
        dd = def_.word2lemma.get(d, d)
        if dd != obj[def_.concept2attribute[dd]]:
            return False
    return True


def run_desc_pred(all_preds, desc):
    if True:
        s = 10000
        for d in desc:
            s = np.fmin(s, all_preds[d])
        return s


    desc = tuple(desc)
    if desc not in run_desc_pred._cache:
        s = 10000
        for d in desc:
            s = np.fmin(s, all_preds[d])
        run_desc_pred._cache[desc] = s

    return run_desc_pred._cache[desc]

run_desc_pred._cache = dict()


def test(index, all_objs, all_preds, meter):
    obj = all_objs[index]
    nr_descriptors = random.randint(2, 5)
    desc = random.choice_list(get_desc(obj), size=nr_descriptors)

    filtered_objs = [i for i, o in enumerate(all_objs) if not run_desc_obj(o, desc)]
    all_scores = run_desc_pred(all_preds, desc)
    rank = (all_scores[filtered_objs] > all_scores[index]).sum()

    # print(desc)
    # print(all_scores)
    # print(all_scores[index])

    meter.update('r@01', rank <= 1)
    meter.update('r@05', rank <= 5)
    meter.update('r@10', rank <= 10)
    meter.update('r@50', rank <= 50)



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
    scenes = scenes[:1000]
    preds = preds[:1000]

    flattened_objs = [o for s in scenes for o in s['objects']]
    flattened_preds = {
        k: np.concatenate([np.array(p[k]) for p in preds], axis=0)
        for k in preds[0]
    }
    meter = GroupMeters()

    for i, obj in tqdm_gofor(flattened_objs, mininterval=0.5):
        test(i, flattened_objs, flattened_preds, meter)

    print(meter.format_simple('Results:', compressed=False))


if __name__ == '__main__':
    main()

