#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : eval-referential.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 30.07.2019
# Last Modified Date: 16.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


# -*- coding: utf-8 -*-
# File   : eval-referential.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/30/2019
#
# This file is part of eval-clevr-instance-retrieval.
# Distributed under terms of the MIT license.

import six
import functools
import sys
from IPython.core import ultratb

import numpy as np
import jacinle.io as io
import jacinle.random as random
from jacinle.cli.argument import JacArgumentParser
from jacinle.utils.tqdm import tqdm_gofor, get_current_tqdm
from jacinle.utils.meter import GroupMeters

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=True)

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
    s = 10000
    for d in desc:
        s = np.fmin(s, all_preds[d])
    return s


def test(index, all_objs, all_preds, meter):
    obj = all_objs[index]
    nr_descriptors = random.randint(1, 3)
    desc = random.choice_list(get_desc(obj), size=nr_descriptors)
    if isinstance(desc, six.string_types):
        desc = [desc]

    filtered_objs = [i for i, o in enumerate(all_objs) if not run_desc_obj(o, desc)]
    all_scores = run_desc_pred(all_preds, desc)
    rank = (all_scores[filtered_objs] > all_scores[index]).sum()

    # print(desc)
    # print(all_scores)
    # print(all_scores[index])

    meter.update('r@01', rank <= 1)
    meter.update('r@02', rank <= 2)
    meter.update('r@03', rank <= 3)
    meter.update('r@04', rank <= 4)
    meter.update('r@05', rank <= 5)


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

    # flattened_objs = [o for s in scenes for o in s['objects']]
    # flattened_preds = {
    #     k: np.concatenate([np.array(p[k]) for p in preds], axis=0)
    #     for k in preds[0]
    # }
    meter = GroupMeters()

    '''
    for i, scene in tqdm_gofor(scenes, mininterval=0.5):
        for j in range(len(scene['objects'])):
            test(j, scene['objects'], preds[i], meter)
    '''
    for i, pred in tqdm_gofor(preds, mininterval=0.5):
        scene = scenes[i]
        for j in range(len(scene['objects'])):
            test(j, scene['objects'], pred, meter)

    print(meter.format_simple('Results:', compressed=False))


if __name__ == '__main__':
    main()
