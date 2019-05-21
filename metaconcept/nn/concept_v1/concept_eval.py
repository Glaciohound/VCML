#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : concept_eval.py
# Author : Chi Han, Jiayuan Mao
# Email  : hanchier@gmail.com, maojiayuan@gmail.com
# Date   : 05/17/2019
#
# This file is part of MetaConcept.
# Distributed under terms of the MIT license.

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jacinle.logging import get_logger; logger = get_logger(__file__)
from jactorch.utils.meta import as_float

from metaconcept import info


class ConceptEvaluation(nn.Module):
    def __init__(self, concept_embeddings, training):
        super().__init__()
        self.concept_embeddings = concept_embeddings
        self.concepts = info.vocabulary.concepts
        self.training = training

    def forward(self, scene_graphs, object_classes):
        monitors = defaultdict(list)
        object_features = torch.cat([sng[1] for sng in scene_graphs], dim=0)
        object_classes = torch.cat(object_classes, dim=0)

        for i, c in enumerate(self.concepts):
            belong = info.vocabulary.belongs_to(c)

            concept_embedding = self.concept_embeddings.get_concept(c).unsqueeze(0)
            result = self.concept_embeddings.infer_metaconcept('object_classify', object_features, concept_embedding)

            loss = F.binary_cross_entropy_with_logits(result, object_classes[:, i].to(result))
            acc = as_float(((result > 0).float() == object_classes[:, i].to(result)).float().mean())

            monitors['acc.concept.' + belong + '.' + c] = acc
            monitors['acc.concept.' + belong].append(acc)
            monitors['acc.concept'].append(acc)

            monitors['loss.concept.' + belong + '.' + c] = loss
            monitors['loss.concept.' + belong].append(loss)
            monitors['loss.concept'].append(loss)

        return monitors


def pn_bce_logits(logits, target):
    pos = (target == 1).sum().item()
    neg = (target == 0).sum().item()
    loss = F.binary_cross_entropy_with_logits(result, object_classes[:, i].to(result), weight=torch.tensor(
        [1 / pos, 1 / neg]
    ).to(result))


class ConceptIsinstanceEvaluation(nn.Module):
    def __init__(self, concept_embeddings, training):
        super().__init__()
        self.concept_embeddings = concept_embeddings
        self.types = list(info.vocabulary.records.keys())
        self.concepts = info.vocabulary.concepts
        self.training = training

    def forward(self):
        all_concept_names, all_concepts = self.concept_embeddings.get_all_concepts()
        concept2idx = {v: i for i, v in enumerate(all_concept_names)}
        max_len_name = max(map(len, all_concept_names))

        log_lines = ['Concept Isistance Evaluation:']
        for concept_name in self.concepts:
            score = self.concept_embeddings.infer_metaconcept('isinstance', self.concept_embeddings.get_concept(concept_name).unsqueeze(0), all_concepts)
            score = F.softmax(score, dim=-1)

            type_scores = dict()
            for t in self.types:
                type_scores[t] = as_float(score[concept2idx[t]])

            log_lines.append(('  {:' + str(max_len_name) + '}: ').format(concept_name) + '; '.join([f'{k}={v:.4f}' for k, v in type_scores.items()]))
        logger.info('\n'.join(log_lines))

