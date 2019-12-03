#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : split_visual.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 04.08.2019
# Last Modified Date: 20.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license

# import numpy as np

from utility.common import\
    random_choice_ratio, difference, belongs_to, \
    load_knowledge, contains, union, random_choice


def split_train_test(visual_dataset, logger, args):
    """
    This splitting method takes only the train & val part of the visual
    dataset. The train split is re-split into train and val, and the original
    val split is regarded as test split.
    """
    logger('Splitting with train-test parts')

    train_dataset = visual_dataset.copy().filter(
        lambda scene: scene['split'] == 'train'
    )
    test_dataset = visual_dataset.copy().filter(
        lambda scene: scene['split'] == 'test'
    )

    val_dataset = train_dataset.copy()
    train_dataset.set_indexes(
        random_choice_ratio(train_dataset.indexes, 6/7)
    )
    val_dataset.set_indexes(
        difference(val_dataset.indexes, train_dataset.indexes)
    )

    visual_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    show_split_sizes(visual_datasets, logger)
    return visual_datasets


def split_train_val(visual_dataset, logger, args):
    """
    This splitting method takes only the train & val part of the visual
    dataset. The train split is re-split into train and val, and the original
    val split is regarded as test split.
    """

    logger('Splitting with train-val parts')
    train_dataset = visual_dataset.copy().filter(
        lambda scene: scene['split'] == 'train'
    )
    test_dataset = visual_dataset.copy().filter(
        lambda scene: scene['split'] == 'val'
    )

    val_dataset = train_dataset.copy()
    train_dataset.set_indexes(
        random_choice_ratio(train_dataset.indexes, 6/7)
    )
    val_dataset.set_indexes(
        difference(val_dataset.indexes, train_dataset.indexes)
    )

    visual_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    show_split_sizes(visual_datasets, logger)
    return visual_datasets


def original_split(visual_dataset, logger, args):
    """
    This splitting method maintains the original splits
    """

    logger('Taking original splits')
    train_dataset = visual_dataset.copy().filter(
        lambda scene: scene['split'] == 'train'
    )
    val_dataset = visual_dataset.copy().filter(
        lambda scene: scene['split'] == 'val'
    )
    test_dataset = visual_dataset.copy().filter(
        lambda scene: scene['split'] == 'test'
    )

    visual_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    show_split_sizes(visual_datasets, logger)
    return visual_datasets


def cub_split(visual_dataset, logger, args):
    """
    Splitting method specially for CUB dataset
    """

    logger('Special splitting function for CUB dataset\n'
           'Splitting by ratio specified in args')

    def get_species(scene):
        return scene['objects']['0']['name']
    species_by_index = {
        image_id: get_species(scene)
        for image_id, scene in visual_dataset.sceneGraphs.items()
    }
    index_by_species = {
        species: [
            image_id
            for image_id, name in species_by_index.items()
            if name == species
        ]
        for species in set(species_by_index.values())
    }
    train_indexes = union(
        *tuple(
            random_choice_ratio(indexes, args.split_ratio['train'])
            for indexes in index_by_species.values()
        ),
        as_set=True,
    )
    val_indexes = union(
        *tuple(
            random_choice_ratio(
                difference(indexes, train_indexes),
                args.split_ratio['val'] / (1 - args.split_ratio['train'])
            )
            for indexes in index_by_species.values()
        ),
        as_set=True
    )
    test_indexes = difference(visual_dataset.indexes,
                              union(train_indexes, val_indexes))
    train_dataset = visual_dataset.copy().set_indexes(train_indexes)
    val_dataset = visual_dataset.copy().set_indexes(val_indexes)
    test_dataset = visual_dataset.copy().set_indexes(test_indexes)

    visual_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    show_split_sizes(visual_datasets, logger)
    return visual_datasets


def combine_results(old_split, new_split):
    if old_split is None:
        return new_split
    elif new_split is None:
        return old_split
    elif old_split == new_split:
        return old_split
    else:
        return 'remove'


def from_one_rule(attrs, train_concept_set, test_concept_set):
    if contains(attrs, train_concept_set):
        return 'train'
    elif contains(attrs, test_concept_set):
        return 'test'
    else:
        return None


def get_split_by_visual_bias(scene, visual_bias,
                             isinstanceof_stats):
    if 'objects' not in scene:
        return False

    split = None

    for obj in scene['objects'].values():
        attrs = obj['concepts_contained']

        for attr_if, attr_then in visual_bias.items():
            cat = belongs_to(isinstanceof_stats, attr_then[0])
            attr_else = difference(isinstanceof_stats[cat],
                                   attr_then)
            if attr_if in attrs:
                split = combine_results(
                    split,
                    from_one_rule(attrs, attr_then, attr_else)
                )

    if split is None:
        if scene['split'] in ['train', 'val']:
            split = 'train'
        elif scene['split'] == 'test':
            split = 'test'
    return split


def split_by_visual_bias(visual_dataset, logger, args, visual_bias):
    """
    Splitting the visual dataset as sepcified in `default_bias`
    """

    logger('Splitting by visual bias')
    logger(visual_bias, resume=True, pretty=True)

    isinstanceof_stats = load_knowledge(args.task, 'isinstanceof')

    def resplit_fn(scene):
        return get_split_by_visual_bias(
            scene, visual_bias[args.task], isinstanceof_stats)

    resplited = visual_dataset.resplit(resplit_fn)
    train_dataset = resplited['train']
    val_dataset = train_dataset.copy()
    test_dataset = resplited['test']

    train_dataset.set_indexes(
        random_choice_ratio(train_dataset.indexes, 6/7)
    )
    val_dataset.set_indexes(
        difference(val_dataset.indexes, train_dataset.indexes)
    )

    visual_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    show_split_sizes(visual_datasets, logger)
    return visual_datasets


def split_by_visual_bias_leaked(visual_dataset, logger, args, visual_bias):
    """
    Splitting the visual dataset as sepcified in `default_bias`
    """

    logger('Splitting by visual bias, with a few unbiased samples')
    logger(visual_bias, resume=True, pretty=True)

    isinstanceof_stats = load_knowledge(args.task, 'isinstanceof')

    def resplit_fn(scene):
        raw_split = get_split_by_visual_bias(
            scene, visual_bias[args.task], isinstanceof_stats)
        '''
        if raw_split == 'test' and \
                np.random.rand() < args.debiasing_leak:
            return 'train'
        else:
        '''
        return raw_split

    resplited = visual_dataset.resplit(resplit_fn)
    train_dataset = resplited['train']
    val_dataset = train_dataset.copy()
    test_dataset = resplited['test']

    train_dataset.set_indexes(
        random_choice_ratio(train_dataset.indexes, 6/7)
    )
    val_dataset.set_indexes(
        difference(val_dataset.indexes, train_dataset.indexes)
    )

    leaked_indexes = random_choice(test_dataset.indexes, args.debiasing_leak)
    train_dataset.add_indexes(leaked_indexes)
    test_dataset.remove_indexes(leaked_indexes)

    visual_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    show_split_sizes(visual_datasets, logger)
    return visual_datasets


def show_split_sizes(visual_datasets, logger):
    """ print the visual split sizes """
    logger('SceneGraph split sizes: ')
    logger(
        {
            split: len(dataset_split)
            for split, dataset_split in visual_datasets.items()
        },
        resume=True
    )
