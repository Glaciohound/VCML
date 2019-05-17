#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : reasoning.py
# Author : Chi Han, Jiayuan Mao
# Email  : hanchier@gmail.com, maojiayuan@gmail.com
# Date   : 05/16/2019
#
# This file is part of MetaConcept.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn


class ProgramExecutor(nn.Module):
    def __init__(self, scene, concept_embeddings, training):
        super().__init__()
        self.scene = scene
        self.concept_embeddings = concept_embeddings
        self.training = training

    def forward(self, program):
        stack = list()
        buffer = list()

        object_features = self.scene[1]
        nr_objects = object_features.size(0)

        for pblock in program:
            op = pblock['operation']
            arg = pblock['argument']

            if op == 'select':
                if arg == 'object_only':
                    stack.append(torch.zeros((nr_objects, ), dtype=torch.float32, device=object_features.device) + 10)
                elif arg == 'concept_only':
                    stack.append(None)
                else:
                    raise ValueError('Unknown select argument: {}.'.format(arg))
            elif op == 'filter':
                concept = self.concept_embeddings.get_concept(arg).unsqueeze(0)
                stack.append(torch.min(
                    stack.pop(),
                    self.concept_embeddings.infer_metaconcept('object_classify', object_features, concept)
                ))
            elif op == 'exist':
                v = stack.pop().max(dim=0)[0] * 0.5
                stack.append((
                    torch.stack([v, -v], dim=0),
                    {'yes': 0, 'no': 1},
                    {0: 'yes', 1: 'no'}
                ))
            elif op == 'choose':
                stack.pop()
                stack.append(self.concept_embeddings.get_concept(arg))
            elif op == 'transfer_cc':
                all_concept_names, all_concepts = self.concept_embeddings.get_all_concepts()
                stack.append((
                    self.concept_embeddings.infer_metaconcept(
                        arg,
                        stack.pop().unsqueeze(0),
                        all_concepts
                    ),
                    {v: i for i, v in enumerate(all_concept_names)},
                    {i: v for i, v in enumerate(all_concept_names)},
                ))
            else:
                raise NotImplementedError('Unimplemented op: {}.'.format(op))

            buffer.append(stack[-1])

        return buffer, stack[-1]

