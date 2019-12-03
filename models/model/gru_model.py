#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gru_model.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 26.07.2019
# Last Modified Date: 21.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file defines a GRU model, with two options: w/o pretrained word
# embeddings.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import jactorch
import jactorch.nn as jacnn

from utility import load_ckpt
from utility.common import detach


class GRUModel(nn.Module):
    def __init__(self, args, tools, device, logger,
                 use_vision=True, fix_vision=False,
                 use_pretrained=False, finetune=True, use_lm=False):
        super().__init__()
        self.tools = tools
        self.device = device
        self.args = args
        self.logger = logger
        self.use_vision = use_vision
        self.fix_vision = fix_vision
        self.use_lm = use_lm
        self.set_lm = use_lm
        self.use_pretrained = use_pretrained
        self.finetune = finetune
        self.dim = args.embed_dim
        self.num_vocab = len(self.tools.words)

        self.build()

    def build(self):

        if self.use_vision:
            import jactorch.models.vision.resnet as resnet
            self.resnet = resnet.resnet34(
                pretrained=True, incl_gap=False, num_classes=None)
            self.resnet.layer4 = jacnn.Identity()

            self.mlp = jacnn.MLPLayer(
                256 + 128 * 2,
                len(self.tools.answers),
                [512])
        else:
            self.mlp = jacnn.MLPLayer(
                128 * 2,
                len(self.tools.answers),
                [256]
            )

        padding_idx = self.tools.words['<NULL>']
        self.embedding = nn.Embedding(
            self.num_vocab, self.dim, padding_idx=padding_idx)

        self.gru = jacnn.GRULayer(
            self.dim, 128, 1,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        self.loss_fn = F.nll_loss

        if self.use_lm:
            self.gru_dropout = nn.Dropout(0.1)
            self.decode = nn.Linear(128 * 2, self.num_vocab)
            self.decode.bias.data.zero_()
            self.decode_loss = jacnn.CrossEntropyLoss(average='none')

    def init(self):
        if self.use_pretrained:
            # when using pretrained embeddings, the weights do not require grad
            self.load_pretrained_embedding('glove')
            if not self.finetune:
                self.logger('Fixing word embeddings')
                self.embedding.weight.requires_grad_(False)
        else:
            init.kaiming_normal_(self.embedding.weight)
        if self.fix_vision and self.use_vision:
            # fix_vision turns the gradient of the vision part off
            self.logger('Fixing vision part')
            for name, parameter in self.resnet.named_parameters():
                parameter.requires_grad = False
        return self.new_optimizer()

    def load_pretrained_embedding(self, name):
        if name == 'glove':
            self.logger('Loading Glove embeddings')
            filename = os.path.join(
                self.args.glove_pretrained_dir,
                f'glove.6B.{self.dim}d.txt'
            )
            with self.logger.levelup():
                table = load_ckpt.load_embedding(filename, self.logger)
                aligned_weights = load_ckpt.align_weights(
                    table, self.tools.words,
                    np.zeros(self.dim),
                    self.logger
                )
                self.embedding.from_pretrained(
                    torch.tensor(aligned_weights))
        else:
            raise NotImplementedError()

    def forward_lm(self, question, question_length, f):
        f = self.gru_dropout(f)
        next_token = self.decode(f)

        assert self.training
        pred = next_token[:, :-1]
        label = question[:, 1:]
        length = question_length - 1
        losses = self.decode_loss(
            pred, label,
            jactorch.length2mask(length, pred.size(1))
        )
        return losses

    def train(self, mode=True):
        super().train(mode=mode)
        if self.use_vision:
            self.resnet.eval()
        self.use_lm = self.set_lm

    def eval(self):
        super().eval()
        self.use_lm = False

    def encode_sentence(self, sent, sent_length):
        f = self.embedding(sent)
        return self.gru(f, sent_length)

    def forward(self, data):
        # pre-processing
        batch_size = data['batch_size']
        data['image'] = data['image'].to(self.device)
        data['question'] = data['question_encoded'].long().to(self.device)
        questions = data['question']
        answers = torch.stack(data['answer_encoded']).to(self.device)

        is_conceptual = [
            data['category'][i] == 'conceptual'
            for i in range(batch_size)
        ]
        is_conceptual = torch.tensor(
            is_conceptual,
            device=self.device,
            dtype=torch.float
        )

        # running language phase
        question_length = (questions > 0).int().sum(1)
        question_states, last_state = \
            self.encode_sentence(questions, question_length)
        last_state = last_state.view(batch_size, -1)

        # visual phase
        if self.use_vision:
            features = self.resnet(data['image'])
            features = features.mean(dim=-1).mean(dim=-1)
            features = features * (1 - is_conceptual).unsqueeze(-1)
            logits = self.mlp(torch.cat([features, last_state], dim=-1))
        else:
            logits = self.mlp(last_state)

        logs = F.log_softmax(logits, dim=1)

        outputs = [
            dict(zip(self.tools.answers,
                     detach(one_log.exp())))
            for one_log in logs
        ]

        # calculating losses
        losses = self.loss_fn(logs, answers, reduction='none')
        if self.use_lm:
            lm_losses = self.forward_lm(
                questions, question_length, question_states)
            losses = losses + lm_losses.sum(1)

        return losses, outputs, None

    def new_optimizer(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=20, verbose=True)
        return optimizer, scheduler

    def set_coach(self, coach):
        self.coach = coach

    def visualize(self, path, plt):
        pass

    def penalty(self):
        return 0

    def update(self):
        pass
