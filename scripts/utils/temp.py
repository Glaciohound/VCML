#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : temp.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 31.07.2019
# Last Modified Date: 09.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# These are some temporary testing codes

import torch
import numpy as np
from tqdm import tqdm

from utility.common import detach, load_knowledge
from metaconcept import info, args

from dataset.visual_dataset.utils.sceneGraph_processing import \
    split_by_visual_bias
from experiments.utils import visualize_utils


def select_by_visual_bias(visual_dataset, visual_bias):
    iio_stats = load_knowledge(args.task, 'isinstanceof')
    splits = {
        image_id: split_by_visual_bias(
            scene, visual_bias, iio_stats
        )
        for image_id, scene in visual_dataset.sceneGraphs.items()
        # for image_id, scene in visual_dataset.local_sceneGraphs.items()
    }
    resplit = {
        split: [image_id
                for image_id, this_split in splits.items()
                if this_split == split]
        for split in ['train', 'val', 'test']
    }
    return resplit


def multiple_check(coach, test_concepts, epochs, indexes, images):
    output = dict()
    for epoch in epochs:
        results = []
        for ind in indexes:
            coach.index = ind
            coach.load(epoch)
            # coach.load(epoch, ind)
            results.append(check_PrecRec(
                coach, test_concepts, images))
        gather = {
            concept: {
                metric: np.array([
                    one_result[concept][metric]
                    for one_result in results
                ]).mean()
                for metric in ['precision', 'recall']
            }
            for concept in test_concepts
        }
        output[epoch] = gather
        print(gather)
    return output


def pr_curve(coach, test_concepts, images, plt):
    all_logits = {
        concept: []
        for concept in test_concepts
    }
    model = coach.model
    concepts_in_argument = torch.tensor([
        info.tools.arguments[concept]
        for concept in test_concepts
    ])

    for image_id in tqdm(images, leave=False):
        scene = info.visual_dataset[image_id]
        data = {
            'object_length': torch.tensor([scene['object_length']]),
            'image': torch.tensor(scene['image'])[None],
            'objects': torch.tensor(scene['objects']),
            'batch_size': 1,
        }
        _, recognized = model.resnet_model(data)
        feature = model.feature_mlp(recognized[0][1])
        if args.model == 'h_embedding_v1':
            conditional_logits, _ = \
                model.embedding.calculate_logits(
                    feature,
                    concepts_in_argument,
                    model.scale,
                    model.inf
                )
            logits = torch.stack(conditional_logits)
        else:
            logits = model.embedding.calculate_logits(
                feature, concepts_in_argument,
            )
            logits = torch.stack(logits)
        recog = detach(torch.sigmoid(logits))
        gt = info.visual_dataset.get_classification(
            info.visual_dataset.sceneGraphs[image_id],
            test_concepts
        ).astype(int).transpose(1, 0)
        for i, concept in enumerate(test_concepts):
            all_logits[concept] += list(zip(recog[i], gt[i]))

    pr_data = get_pr(all_logits)

    return pr_data, all_logits


def get_pr(all_logits, samples=None):
    all_logits = {
        concept: np.stack(item)
        for concept, item in all_logits.items()
    }
    if samples is not None:
        all_logits = {
            concept: all_logits[concept][sample]
            for concept, sample in samples.items()
        }
    all_logits = {
        concept: sorted(item, key=lambda x: x[0])
        for concept, item in all_logits.items()
    }
    pr_data = dict()

    for concept in all_logits.keys():
        this_data = np.zeros((len(all_logits[concept]), 2))
        total = len([pair for pair in all_logits[concept] if pair[1] == 1])
        count = 0
        for i, obj in enumerate(all_logits[concept]):
            count += obj[1] == 1
            this_data[i] = [count / total, count / (i + 1)]
        pr_data[concept] = this_data

    return pr_data


def check_PrecRec(coach, test_concepts, images):

    model = coach.model
    concepts_in_argument = torch.tensor([
        info.tools.arguments[concept]
        for concept in test_concepts
    ])
    num = len(test_concepts)
    total = np.zeros(num)
    true_positive = np.zeros(num)
    positive = np.zeros(num)

    with torch.no_grad():
        for image_id in tqdm(images, leave=False):
            scene = info.visual_dataset[image_id]
            data = {
                'object_length': torch.tensor([scene['object_length']]),
                'image': torch.tensor(scene['image'])[None],
                'objects': torch.tensor(scene['objects']),
                'batch_size': 1,
            }
            _, recognized = model.resnet_model(data)
            feature = model.feature_mlp(recognized[0][1])
            if args.model == 'h_embedding_v1':
                conditional_logits, _ = \
                    model.embedding.calculate_logits(
                        feature,
                        concepts_in_argument,
                        model.scale,
                        model.inf
                    )
                logits = torch.stack(conditional_logits)
            else:
                logits = model.embedding.calculate_logits(
                    feature, concepts_in_argument,
                )
                logits = torch.stack(logits)
            recog = detach(torch.sigmoid(logits))
            gt = info.visual_dataset.get_classification(
                info.visual_dataset.sceneGraphs[image_id],
                test_concepts
            ).astype(int).transpose(1, 0)
            total += gt.sum(1)
            true_positive += (gt * recog).sum(1)
            positive += recog.sum(1)

    output = {
        concept: {
            'precision': true_positive[i] / positive[i],
            'recall': true_positive[i] / total[i],
        }
        for i, concept in enumerate(test_concepts)
    }
    return output


def for_jiayuan(coach, visual_dataset):
    model = coach.model
    concepts_in_argument = torch.tensor([
        info.tools.arguments[concept]
        for concept in info.tools.concepts
    ])
    output = {}

    with torch.no_grad():
        for image_id in tqdm(visual_dataset.sceneGraphs.keys(),
                             leave=False):
            scene = visual_dataset[image_id]
            data = {
                'object_length': torch.tensor([scene['object_length']]),
                'image': torch.tensor(scene['image'])[None],
                'objects': torch.tensor(scene['objects']),
                'batch_size': 1,
            }
            _, recognized = model.resnet_model(data)
            feature = model.feature_mlp(recognized[0][1])
            conditional_logits, raw_logits = \
                model.embedding.calculate_logits(
                    feature,
                    concepts_in_argument,
                    model.scale,
                    model.inf
                )
            concept_dict = {
                concept: conditional_logits[i].detach().cpu().numpy()
                for i, concept in enumerate(info.tools.concepts)
            }
            output[image_id] = concept_dict

    return output


def metric(data, logits, samples=None, th=None):
    output = dict()
    for concept in data.keys():
        num = len([pair
                   for i, pair in enumerate(logits[concept])
                   if pair[0] < th and
                   (samples is None or i in samples[concept])])
        precision = data[concept][num, 1]
        recall = data[concept][num, 0]
        f1 = 2 / (1 / precision + 1 / recall)
        output[concept] = {'precision': precision, 'recall': recall, 'f1': f1}

    return output


def calculate_auc(data):
    output = dict()
    for concept, pr in data.items():
        last = pr[:-1]
        this = pr[1:]
        area = (last[:, 1] + this[:, 1]) / 2 * (this[:, 0] - last[:, 0])
        area = area.sum()
        output[concept] = area
    return output


def concat_logits(data):
    output = {
        concept: np.stack(this_data)
        for concept, this_data in data.items()
    }
    return output


def sample_to_ratio(source, target):
    output = dict()
    for concept, this_data in source.items():
        ratio = target[concept][:, 1].mean()
        true_indexes = np.array(this_data[:, 1].nonzero()[0])
        false_indexes = np.array((1 - this_data[:, 1]).nonzero()[0])
        num_false = int(true_indexes.shape[0] * (1 - ratio) / ratio)
        false_selection = np.random.choice(
            false_indexes, num_false)
        output[concept] = np.concatenate([true_indexes, false_selection])
    return output


def plot(data, legends, concepts, plt):
    for concept in concepts:
        fig = info.plt.figure()
        ax = fig.add_subplot(111)
        for i, setting in enumerate(data):
            ax.plot(setting[concept][:, 0], setting[concept][:, 1])
        ax.legend(legends, loc='upper left')
        visualize_utils.savefig(fig, concept, 'mix')
        plt.clf()
