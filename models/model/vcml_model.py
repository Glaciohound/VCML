#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : vcml_model.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 03.12.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.nn.scene_graph import ResNetSceneGraph
from models.nn.framework import reasoning, embedding

from utility.common import detach


class VCML_Model(nn.Module):
    """
    Currently used h_embedding models.
    Variants include: h_embedding_bert & h_embedding_nscl.
    """
    def __init__(self, args, tools, device, logger):
        super().__init__()
        self.args = args
        self.tools = tools
        self.device = device
        self.version = args.model
        self.logger = logger
        self.build()

    def build(self):
        args = self.args
        self.embedding = embedding.ConceptEmbedding(
            args, self.tools, self.device, self.version
        )
        self.reasoning = reasoning.ProgramExecutor(
            args, self.tools, self.device
        )
        self.classify = reasoning.Classification(args, self.tools, self.device)
        self.resnet_model = ResNetSceneGraph(
            self.device, relation=False, dropout_rate=args.dropout)

        self.feature_mlp = self.sub_net(
            args.feature_dim, args.hidden_dim, args.embed_dim)

    def sub_net(self, in_dim, hidden_dim, out_dim):
        if hidden_dim != 0:
            net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            net = nn.Linear(in_dim, out_dim)
        return net

    def get_embedding(self, *arg, **kwarg):
        return self.embedding.get_embedding(*arg, **kwarg)

    def init(self):
        if self.args.fix_resnet:
            for name, param in self.resnet_model.named_parameters():
                param.requires_grad_(False)
        self.embedding.init()
        self.reasoning.init()
        self.classify.init()
        return self.new_optimizer()

    def train(self, mode=True):
        super().train(mode=mode)
        if self.args.fix_resnet:
            self.resnet_model.eval()

    # running a batch
    def forward(self, data):
        batch_size = data['batch_size']
        if self.args.use_gt_program:
            program = data['program']
            program_encoded = data['program_encoded']
        else:
            program = data['program_parsed']
            program_encoded = data['program_parsed_encoded']

        objects = self.process_objects(data)

        # running
        losses, outputs, debugs = [], [], []
        for i in range(batch_size):
            debug = {}

            logits = self.embedding.calculate_logits(
                objects[i],
                program_encoded[i][:, 1],
            )
            if data['type'][i] == 'classification':
                piece_result = self.classify(
                    logits,
                    data['answer_encoded'][i],
                    program_encoded[i][:, 1],
                )
            else:
                try:
                    piece_result = self.reasoning(
                        program[i],
                        data['answer'][i], data['category'][i],
                        objects[i], logits,
                        self.embedding,
                    )
                except Exception:
                    piece_result = (0, {'yes': 0.5, 'no': 0.5}, {})

            loss, output, this_debug = piece_result
            debug.update(this_debug)

            losses.append(loss)
            outputs.append(output)
            debugs.append(debug)

        return losses, outputs, debugs, objects

    # getting object features
    def process_objects(self, data):
        def feature_mlp(feature):
            if feature is None:
                return None
            else:
                return self.feature_mlp(feature)

        _, _, recognized = self.resnet_model(data)
        objects = [
            feature_mlp(feature[1])
            for feature in recognized
        ]

        return objects

    def new_optimizer(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=20, verbose=True)
        return optimizer, scheduler

    def set_coach(self, coach):
        self.coach = coach

    @property
    def max_len(self):
        output = detach(
            self.embedding.concept_embedding.weight.
            pow(2).sum(-1).max().sqrt()
        )
        return output

    def visualize(self, path, plt):
        self.embedding.visualize(path, plt)

    def penalty(self):
        return self.embedding.penalty()

    def update(self):
        pass
