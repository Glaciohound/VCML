#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : reasoning.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 23.07.2019
# Last Modified Date: 21.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.common import min_fn, log, detach


INF = 100


class ProgramExecutor(nn.Module):

    def __init__(self, args, tools, device):
        super().__init__()
        if not args.not_build_reasoning:
            self.build()
        self.args = args
        self.tools = tools
        self.device = device

    def build(self):
        self.exist_offset = nn.Parameter(torch.tensor(0.))
        self.exist_scale = nn.Parameter(torch.tensor(1.))

    def forward(
        self,
        program, answer, question_cat,
        objects, logits,
        embedding,
    ):
        result = None

        for j, op in enumerate(program):
            operation = op['operation']
            argument = op['argument']

            if operation == 'select_object':
                result = self.select_object_fn(objects.shape[0], INF)

            elif operation == 'select_concept':
                result = self.select_concept_fn(embedding, argument)

            elif operation == 'unique_object':
                result = self.unique_object_fn(result, objects)

            elif operation == 'query':
                raise NotImplementedError('Querying module not implemented')

            elif operation == 'filter':
                result = self.filter_fn(result, logits[j])

            elif operation == 'exist':
                result = self.exist_fn(result)

            elif operation in ['synonym', 'hypernym',
                               'samekind', 'meronym']:
                result = self.judge_relation(
                    result, argument, embedding, operation)

            elif operation == 'isinstanceof':
                result = self.isinstanceof_fn(
                    result, embedding)

            elif operation in ['<END>']:
                pass

            else:
                raise Exception('unsupported opeartion: {}'.format(op))

        """ modifying values"""
        loss, output, debug = self.analyze(result, answer)
        if question_cat == 'conceptual':
            loss = loss * self.args.conceptual_weight
        return loss, output, debug

    # the following are operation modules

    def select_object_fn(self, n, INF):
        return 'object_logits', torch.ones(n).to(self.device) * INF

    def select_concept_fn(self, embedding, argument):
        return 'concept_embedding', \
            embedding.get_embedding('concept', argument)

    def unique_object_fn(self, result, objects):
        weighted_sum = (F.softmax(result[1], dim=0)[None] *
                        objects).sum(0)
        return 'object_embedding', weighted_sum

    def unique_concept_fn(self, result, embedding):
        weighted_sum = (F.softmax(result[1], dim=0)[None] *
                        embedding.all_concept_embeddings).sum(0)
        return 'concept_embedding', weighted_sum

    def filter_fn(self, results, logits):
        filtered_logits = min_fn(results[1], logits)
        return 'object_logits', filtered_logits

    def exist_fn(self, result):
        if self.args.not_build_reasoning:
            output = result[1].max()
        else:
            output = (result[1].max() + self.exist_offset) * self.exist_scale
        return ('boolean', output)

    def isinstanceof_fn(self, result, embedding):
        attributes = embedding.all_attribute_embeddings
        detach_concept = self.args.detach_in_rel
        result = embedding.determine_relation(
            result[1], attributes,
            detach=(detach_concept, False),
        )[:, 1]
        result = (
            'attribute_logits',
            result,
        )
        return result

    def judge_relation(self, result, argument, embedding, operation):
        metaconcept_index = {
            'synonym': 0, 'hypernym': 2, 'samekind': 3, 'meronym': 4,
        }[operation]
        another_concept = embedding.get_embedding('concept', argument)

        detach_concept = self.args.detach_in_rel
        judgement = embedding.determine_relation(
            result[1], another_concept,
            detach=(detach_concept, detach_concept),
        )
        result = (
            'boolean',
            judgement[metaconcept_index]
        )
        return result

    # analyzing outputs

    def boolean_analyze(self, result, answer):
        output = {
            'yes': detach(torch.sigmoid(result[1])),
            'no': detach(torch.sigmoid(-result[1])),
        }

        if answer == 'yes':
            loss = -log(result[1])
        else:
            loss = -log(-result[1])

        return loss, output, {}

    def attribute_logits_analyze(self, result, answer):
        logs = F.log_softmax(result[1], dim=0)
        output = dict(zip(
            self.tools.attributes,
            detach(logs.exp())
        ))
        index = self.tools.attributes[answer]
        target = torch.LongTensor([index]).to(self.device)
        loss = F.nll_loss(logs[None], target)

        return loss, output, {}

    def analyze(self, result, answer):
        if result[0] == 'boolean':
            return self.boolean_analyze(result, answer)
        elif result[0] == 'attribute_logits':
            return self.attribute_logits_analyze(result, answer)
        else:
            raise Exception('result type error: wrong type %s' % result[0])

    def init(self):
        pass


class Classification(nn.Module):
    def __init__(self, args, tools, device):
        super().__init__()
        self.args = args
        self.tools = tools
        self.device = device

    def forward(self, logits, answer, argument_index):

        target = answer.to(self.device)
        classify_logits = torch.stack(logits).transpose(1, 0)
        binary_loss = F.binary_cross_entropy_with_logits(
            classify_logits, target, reduction='none'
        )

        output = detach(torch.sigmoid(classify_logits))

        return binary_loss, output, {}

    def init(self):
        pass
