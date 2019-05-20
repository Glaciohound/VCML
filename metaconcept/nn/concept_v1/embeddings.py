#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : embeddings.py
# Author : Chi Han, Jiayuan Mao
# Email  : hanchier@gmail.com, maojiayuan@gmail.com
# Date   : 05/16/2019
#
# This file is part of MetaConcept.
# Distributed under terms of the MIT license.

import math
import torch
import torch.nn as nn
import jactorch.nn as jacnn


class ConceptEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.all_concepts = list()
        self._all_concept_buffer = None
        self._all_concept_buffer_dirty = False

    def init_concept(self, name, embedding_dim):
        self.all_concepts.append(name)
        self.register_parameter('concept_' + name, nn.Parameter(torch.randn(embedding_dim)))
        self._all_concept_buffer_dirty = True

    def init_metaconcept(self, name, embedding_dim):
        self.register_module('metaconcept_' + name, jacnn.MLPLayer(embedding_dim * 4, 1, [embedding_dim * 2], activation='relu'))

    def get_all_concepts(self):
        if self._all_concept_buffer is None or self._all_concept_buffer_dirty:
            all_concepts = sorted(self.all_concepts)
            self._all_concept_buffer = all_concepts, torch.stack([self.get_concept(name) for name in self.all_concepts], dim=0)
            self._all_concept_buffer_dirty = False
        return self._all_concept_buffer

    def get_concept(self, name):
        return getattr(self, 'concept_' + name)

    def get_metaconcept(self, name):
        return getattr(self, 'metaconcept_' + name)

    def infer_metaconcept(self, name, a, b):
        # a.shape = (batch_size, embedding_dim)
        # b.shape = (batch_size, embedding_dim)
        max_size = tuple(map(max, zip(a.shape, b.shape)))
        a = a.expand(max_size)
        b = b.expand(max_size)

        # if name == 'object_classify':
        #     return (a * b).sum(dim=-1) / math.sqrt(a.shape[-1])

        abdp = torch.cat([a, b, a - b, a * b], dim=-1)
        return self.get_metaconcept(name)(abdp).squeeze(-1)

    def register_module(self, name, module):
        setattr(self, name, module)

