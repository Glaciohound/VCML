#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-image-retrieval.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/16/2018
#
# This file is part of VisualReasoning-PyTorch.
# Distributed under terms of the MIT license.

import six
import functools

import jacinle.io as io
import jacinle.random as random
from jacinle.cli.argument import JacArgumentParser
from jacinle.utils.tqdm import tqdm_gofor, get_current_tqdm

parser = JacArgumentParser()
parser.add_argument('--scene-json', required=True, type='checked_file')
parser.add_argument('--output-json', required=True, type=str)
parser.add_argument('--qa-compatible', action='store_true', help='if True, requires object A is unique.')
parser.add_argument('--full', action='store_true', help='if True, use all scenes.')
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


def_ = Definition()


def test_filter(o, f):
    return f == o[def_.concept2attribute[f]]


def gen_test_filter(f):
    return functools.partial(test_filter, f=f)


def filter_objects(scene, filters):
    objects = list(scene['objects'])
    if isinstance(filters, six.string_types):
        filters = [filters]
    for f in filters:
        objects = list(filter(gen_test_filter(f), objects))
    return objects


def filter_objects_id(scene, filters):
    objects = list(enumerate(scene['objects']))
    if isinstance(filters, six.string_types):
        filters = [filters]
    for f in filters:
        objects = [(x, y) for x, y in objects if test_filter(y, f)]
    return objects


def describe_object(obj):
    return [obj[k] for k in def_.annotation_attribute_names]


def gen_hard_negative_object(scene, target):
    objects = list(scene['objects'])
    for _ in range(100):
        desc = describe_object(objects[target]).copy()
        attr_t = random.choice(len(def_.annotation_attribute_names))
        attr = def_.annotation_attribute_names[attr_t]
        known_values = def_.concepts[attr].copy()
        assert desc[attr_t] in known_values
        known_values.remove(desc[attr_t])
        desc[attr_t] = random.choice(known_values)
        if len(filter_objects(scene, desc)) == 0:
            return desc
    return None


def exe(scene, q):
    xs = filter_objects_id(scene, q['object_a'])
    ys = filter_objects_id(scene, q['object_b'])
    answer = False
    for x, _ in xs:
        for y, _ in ys:
            if y in scene['relationships'][q['relation']][x]:
                answer = True
    return answer


def gen(scene):
    questions = list()
    all_relations = list()
    for k in def_.annotation_relation_names:
        for i, js in enumerate(scene['relationships'][k]):
            for j in js:
                all_relations.append((i, j, k))

    for pos_id in range(10):
        for _ in range(100):
            i, j, k = random.choice_list(all_relations)
            desc_a = describe_object(scene['objects'][i])
            desc_b = describe_object(scene['objects'][j])
            if not args.qa_compatible or len(filter_objects(scene, desc_a)) == 1:
                questions.append({
                    'image_index': scene['image_index'],
                    'image_filename': scene['image_filename'],
                    'object_a': desc_a,
                    'object_b': desc_b,
                    'relation': k,
                    'answer': True
                })
                break

    for neg_id in range(5):
        for _ in range(100):
            a, b = random.choice_list(list(range(len(scene['objects']))), size=2, replace=False)
            for k in ['behind', 'front', 'left', 'right']:
                if b not in scene['relationships'][k][a]:
                    break
            assert b not in scene['relationships'][k][a]
            desc_a = describe_object(scene['objects'][a])
            desc_b = describe_object(scene['objects'][b])
            if not args.qa_compatible or len(filter_objects(scene, desc_a)) == 1:
                questions.append({
                    'image_index': scene['image_index'],
                    'image_filename': scene['image_filename'],
                    'object_a': desc_a,
                    'object_b': desc_b,
                    'relation': k,
                    'answer': False
                })
                # this is just very hard to ensure False by a rule-based generator...
                questions[-1]['answer'] = exe(scene, questions[-1])
                break

    for neg_id in range(5):
        done_flag = False
        for _ in range(100):
            i, j, k = random.choice_list(all_relations)
            desc_a = describe_object(scene['objects'][i])
            desc_b = describe_object(scene['objects'][j])
            if random.rand() < 0.5:
                desc_a = gen_hard_negative_object(scene, i)
            else:
                desc_b = gen_hard_negative_object(scene, j)
                if args.qa_compatible and len(filter_objects(scene, desc_a)) != 1:
                    continue
            if desc_a is not None and desc_b is not None:
                done_flag = True
                questions.append({
                    'image_index': scene['image_index'],
                    'image_filename': scene['image_filename'],
                    'object_a': desc_a,
                    'object_b': desc_b,
                    'relation': k,
                    'answer': False
                })
                break

    for i, q in enumerate(questions):
        answer = exe(scene, q)
        if answer != q['answer']:
            print(i, q)
            assert False

    return questions


def main():
    scenes = io.load_json(args.scene_json)['scenes']
    if not args.full:
        scenes = scenes[-1000:]
    questions = list()
    for i, scene in tqdm_gofor(scenes):
        gg = gen(scene)
        questions.extend(gg)
        get_current_tqdm().set_description('index = {}, quesiton/count = {}'.format(i, len(gg)))

    io.dump_json(args.output_json, {'questions': questions})


if __name__ == '__main__':
    main()

