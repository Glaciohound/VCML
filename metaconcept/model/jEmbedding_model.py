#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : jEmbedding_model.py
# Author : Chi Han, Jiayuan Mao
# Email  : hanchier@gmail.com, maojiayuan@gmail.com
# Date   : 05/16/2019
#
# This file is part of MetaConcept.
# Distributed under terms of the MIT license.

import os
import pprint
from copy import deepcopy
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import jacinle
import jactorch
import jactorch.nn as jacnn

from metaconcept import info, args
from metaconcept.nn.scene_graph import ResNetSceneGraph
from metaconcept.nn.concept_v1.embeddings import ConceptEmbedding
from metaconcept.nn.concept_v1.reasoning import ProgramExecutor
from metaconcept.nn.concept_v1.concept_eval import ConceptEvaluation
from metaconcept.utils.common import to_numpy, to_normalized, min_fn, matmul, to_tensor, vistb, arange, logit_exist, log_or, logit_xand


class JEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet_model = ResNetSceneGraph(relation=False)
        self.object_feature = nn.Linear(256, 100)
        self.concept_embeddings = ConceptEmbedding()
        self.init_concept_embeddings()
        self.inf = 100

    def init_concept_embeddings(self):
        for k, vs in info.vocabulary.records.items():
            self.concept_embeddings.init_concept(k, args.embed_dim)
            for v in vs:
                self.concept_embeddings.init_concept(v, args.embed_dim)
        # TODO(Jiayuan Mao @ 05/16): use the value from the dataset.
        for k in ['object_classify', 'isinstance', 'synonym', 'hypernym', 'hyponym']:
            self.concept_embeddings.init_metaconcept(k, args.embed_dim)

    def forward(self, data):
        batch_size = data['answer'].shape[0]

        monitors, outputs = defaultdict(list), defaultdict(list)

        features, scene_graphs = self.resnet_model(data)
        scene_graphs = [
            [None, self.object_feature(sg[1]), None]
            for sg in scene_graphs
        ]

        for i in range(batch_size):
            sg = scene_graphs[i]
            executor = ProgramExecutor(self.concept_embeddings, self.training)
            buffer, answer = executor(sg, data['program'][i])

            logits, word2idx, idx2word = answer
            assert logits.dim() == 1
            pred = logits.argmax().item()

            if 'answer' in data:
                qa_type = 'conceptual' if data['type'][i] in args.conceptual_subtasks else 'visual'

                acc = float(pred == word2idx[data['answer'][i]])
                monitors['acc.qa'].append(acc)
                monitors['acc.qa.' + qa_type].append(acc)
                monitors['acc.qa.' + qa_type + '.' + data['type'][i]].append(acc)

                if self.training:
                    log_softmax = F.log_softmax(logits, dim=-1)
                    loss = -log_softmax[word2idx[data['answer'][i]]]

                    if qa_type == 'conceptual':
                        monitors['loss.qa'].append(loss * args.conceptual_weight)
                    else:
                        monitors['loss.qa'].append(loss)

                    monitors['loss.qa.' + qa_type].append(loss)
                    monitors['loss.qa.' + qa_type + '.' + data['type'][i]].append(loss)

            outputs['answer'] = idx2word[pred]

        concept_eval = ConceptEvaluation(self.concept_embeddings, self.training)
        monitors.update(concept_eval(scene_graphs, data['object_classes']))

        canonize_monitors(monitors)

        if self.training:
            # monitors['loss'] = monitors['loss.qa'] + monitors['loss.concept']
            monitors['loss'] = monitors['loss.qa']
            monitors['acc'] = monitors['acc.qa']
            return monitors['loss'], monitors, outputs
        else:
            return None, monitors, outputs

    def init(self):
        # inited = []
        # for name, param in self.named_parameters():
        #     if not name.startswith('resnet_model'):
        #         inited.append(name)
        #         if info.new_torch:
        #             init.normal_(param, 0, args.init_variance)
        #         else:
        #             init.normal(param, 0, args.init_variance)

        # print('initalized parameters:', end='')
        # jacinle.stprint(inited)
        self.new_optimizer()

    def new_optimizer(self):
        info.optimizer = optim.Adam(self.parameters(),
                                   lr=args.lr)
        info.scheduler = ReduceLROnPlateau(info.optimizer, patience=2, verbose=True)

    def save(self, name):
        torch.save({'model': self.state_dict(),
                    'optimizer': info.optimizer.state_dict(),
                    'scheduler': info.scheduler.state_dict(),
                    'protocol': (info.protocol['operations'], info.protocol['concepts'])},
                   os.path.join(args.ckpt_dir, name+'.tar'))

    def load(self, name, retrain=False):
        ckpt = torch.load(os.path.join(args.ckpt_dir, name+'.tar'))
        info.model.load_state_dict(ckpt['model'])
        if retrain:
            self.new_optimizer()
        else:
            info.optimizer.load_state_dict(ckpt['optimizer'])
            info.scheduler.load_state_dict(ckpt['scheduler'])
            for state in info.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        protocol = ckpt['protocol']
        old_protocol = deepcopy(info.protocol)
        info.protocol.reset()
        [info.protocol['operations', o] for o in protocol[0]]
        [info.protocol['concepts', c] for c in protocol[1]]
        [info.protocol['operations', o] for o in old_protocol['operations']]
        [info.protocol['concepts', c] for c in old_protocol['concepts']]


def canonize_monitors(monitors):
    for k, v in monitors.items():
        if isinstance(v, list):
            if len(v) == 0:
                monitors[k] = 0
            elif isinstance(v[0], (int, float)):
                monitors[k] = sum(v) / max(len(v), 1e-3)
            else:
                monitors[k] = torch.stack(v).mean()

