#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : bert_model.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 16.07.2019
# Last Modified Date: 21.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file describes a BERT-based language baseline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_transformers import BertTokenizer, BertModel

from utility.common import detach


class BERTModel(nn.Module):
    def __init__(self, args, tools, device, logger):
        super().__init__()

        self.tools = tools
        self.device = device
        self.args = args
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        self.bert = model
        _, self.bert_dim = model.embeddings.position_embeddings.weight.shape

        self.mlp = self.sub_net(
            self.bert_dim, self.args.hidden_dim, len(tools.answers)
        )
        self.trainining_parameters = self.mlp.parameters()

    def sub_net(self, in_dim, hidden_dim, out_dim):
        net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        return net

    def forward(self, data):
        batch_size = data['batch_size']
        assert not any(qtype == 'classifiction' for qtype in data['type']),\
            'BERT model can not handle classification tasks'
        questions_encoded = [
            self.tokenizer.encode(question)
            for question in data['question']
        ]
        max_length = max(len(encoded) for encoded in questions_encoded)
        questions_encoded = torch.stack([
            torch.tensor(self.pad_to_length(encoded, max_length),
                         device=self.device)
            for encoded in questions_encoded
        ])
        answer_encoded = torch.stack(data['answer_encoded']).to(self.device)
        batch_size = data['batch_size']

        bert_attention, last_state = self.bert(questions_encoded)
        logits = self.mlp(last_state)
        logs = F.log_softmax(logits, dim=1)
        outputs = [
            dict(zip(
                self.tools.answers,
                detach(one_log.exp())
            ))
            for one_log in logs
        ]
        losses = F.nll_loss(logs, answer_encoded, reduction='none')
        debugs = [{} for i in range(batch_size)]

        return losses, outputs, debugs

    def init(self):
        self.bert.eval()
        for name, param in self.mlp.named_parameters():
            try:
                init.kaiming_normal_(param)
            except Exception:
                init.normal_(param, 0, self.args.init_variance)
        for param in self.bert.parameters():
            param.requires_grad_(False)

        return self.new_optimizer()

    def new_optimizer(self):
        optimizer = optim.Adam(self.trainining_parameters, lr=self.args.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=20, verbose=True)

        return optimizer, scheduler

    def set_coach(self, coach):
        self.coach = coach

    def train(self, mode=True):
        super().train(mode=mode)
        self.bert.eval()

    def pad_to_length(self, sequence, to_length):
        pad_token = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        output = sequence + [pad_token] * (to_length - len(sequence))
        return output

    def visualize(self, path, plt):
        pass

    def penalty(self):
        return 0

    def update(self):
        pass
