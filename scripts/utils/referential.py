#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : referential.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 23.10.2019
# Last Modified Date: 23.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


import six
import sys
import torch
import numpy as np

from IPython.core import ultratb

import jacinle.random as random
from jacinle.utils.meter import GroupMeters

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=True)


class Definition(object):
    annotation_attribute_names = ['color', 'material', 'shape', 'size']
    annotation_relation_names = ['behind', 'front', 'left', 'right']
    concepts = {
        'color': ['gray', 'red', 'blue', 'green',
                  'brown', 'purple', 'cyan', 'yellow'],
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

    filtered_objs = np.array([
        i for i, o in enumerate(all_objs)
        if not run_desc_obj(o, desc)], dtype=int)
    all_scores = run_desc_pred(all_preds, desc)
    rank = (all_scores[filtered_objs] > all_scores[index]).sum()

    meter.update('r@01', rank <= 1)
    meter.update('r@02', rank <= 2)
    meter.update('r@03', rank <= 3)
    meter.update('r@04', rank <= 4)
    meter.update('r@05', rank <= 5)


def get_preds(model, ref_dataset, logger):
    device = model.device
    args = model.args
    concepts = model.tools.concepts

    preds = {}

    def add_preds(data_batch):
        batch = {
            'batch_size': len(data_batch),
            'image': torch.Tensor(
                [data['image'] for data in data_batch]
            ).to(device),
            'objects': torch.Tensor(
                np.concatenate([data['objects'] for data in data_batch])
            ).to(device),
            'object_length': torch.LongTensor(
                [data['object_length'] for data in data_batch]),
            'image_id': [data['image_id'] for data in data_batch],
        }
        features = model.process_objects(batch)
        concept_embedding = model.embedding.get_all_concept_embeddings()
        for image_id, one_feature in zip(batch['image_id'], features):
            logits = model.embedding.logit_fn(
                one_feature, concept_embedding
            ).transpose(0, 1).detach().cpu().numpy()
            preds[image_id] = dict(zip(
                concepts, logits
            ))

    with torch.no_grad():
        pbar = logger.tqdm(ref_dataset.indexes)
        data_batch = []

        for image_index in pbar:
            data_batch.append(ref_dataset[image_index])
            if len(data_batch) == args.batch_size:
                add_preds(data_batch)
                data_batch.clear()
        if len(data_batch) != 0:
            add_preds(data_batch)
        data_batch.clear()

    return preds


def ref_epoch(coach, prepare_fn, recording, ref_dataset):
    prepare_fn()
    model = coach.model
    recording.reset()

    preds = get_preds(model, ref_dataset, coach.logger)

    meter = GroupMeters()

    for image_id, pred in coach.logger.tqdm(preds.items(), mininterval=0.5):
        scene = ref_dataset.sceneGraphs[image_id]
        for j in range(len(scene['objects'])):
            test(j, list(scene['objects'].values()), pred, meter)

    recording.record(meter.avg)
