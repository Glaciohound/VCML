#!/usr/bin/env python
# coding=utf-8

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)
import torch
import torch.nn as nn
import torch.utils.data

class Transition(nn.Module):
    def __init__(self, args, info=None):
        super(Transition, self).__init__()
        self.args = args
        self.info = info

        self.relation_register = []
        self.rel_subj = []
        self.rel_obj = []
        self.relation_encoding = nn.Parameter(torch.randn((args.max_relations, args.relation_dim)))
        self.relation_embedding = nn.Linear(args.relation_dim, args.attention_dim)

        self.state_embedding = nn.Embedding(args.size, args.state_embed_dim)
        self.state2Rel_linear1 = nn.Linear(2 * args.state_embed_dim, args.hidden_dim)
        self.state2Rel_linear2 = nn.Linear(args.hidden_dim, args.attention_dim)
        self.state2Rel = lambda x: self.state2Rel_linear2(torch.relu(self.state2Rel_linear1(x)))

        self.action_embedding = nn.Embedding(args.num_action, args.attention_dim)
        self.position_embedding = nn.Embedding(args.max_length, args.attention_dim)
        self.metaMode_init = nn.Parameter(torch.randn(args.attention_dim))
        self.metaMode_transitionLinear1G = nn.Linear(args.attention_dim + args.size,
                                              args.hidden_dim)
        self.metaMode_transitionLinear2G = nn.Linear(args.hidden_dim, args.attention_dim)
        self.metaMode_transitionG = lambda x:\
            self.metaMode_transitionLinear2G(torch.relu(self.metaMode_transitionLinear1G(x)))

        self.query_linear = nn.Linear(args.attention_dim, 2 * args.attention_dim)
        self.metaMode_transitionLinear1S = nn.Linear(3 * args.attention_dim,
                                              args.hidden_dim)
        self.metaMode_transitionLinear2S = nn.Linear(args.hidden_dim, args.attention_dim)
        self.metaMode_transitionS = lambda x:\
            self.metaMode_transitionLinear2S(torch.relu(self.metaMode_transitionLinear1S(x)))


    def forward(self, x):
        args = self.args
        info = self.info
        init = x[:, :args.size].float()
        actions = x[:, args.size:].long()
        positions = torch.arange(args.length)[None].repeat((x.shape[0], 1))

        actions_embedded = self.action_embedding(actions)
        position_embedded = self.position_embedding(positions.to(info.device))
        sentence = torch.cat((actions_embedded, position_embedded), 2).float().to(info.device)
        output = torch.zeros((x.shape[0], args.length, args.size))

        state = init
        metaMode = self.metaMode_init[None].repeat((x.shape[0], 1))

        for i in range(args.length):
            if args.run_mode == 'auto':
                metaMode = self.metaMode_transitionG(torch.cat((state, metaMode), 1))
            elif args.run_mode == 'program':
                metaMode = self.action_embedding(actions[:, i])
            elif args.run_mode == 'attention':
                query = self.query_linear(metaMode)
                attention = (query[:, None] * sentence).sum(2)
                attention = torch.softmax(attention, 1)
                attended = (attention[:, :, None] * sentence).sum(1)
                metaMode = self.metaMode_transitionS(torch.cat((metaMode, attended), 1))

            if args.relation_mode == 'stored':
                relation = self.relation_embedding(self.relation_encoding[:self.num_relations])
            elif args.relation_mode == 'deducted':
                obj = self.state_embedding(torch.LongTensor(self.rel_obj).to(info.device))
                subj = self.state_embedding(torch.LongTensor(self.rel_subj).to(info.device))
                state_stack = torch.cat((subj, obj), 1)
                relation = self.state2Rel(state_stack)

            expand_relation = relation[None].repeat((x.shape[0], 1, 1))
            expand_action = metaMode[:, None].repeat((1, self.num_relations, 1))

            h = (expand_relation * expand_action).sum(2)
            h = torch.sigmoid(h)
            next_state = torch.zeros_like(state)
            for j in range(self.num_relations):
                next_state[:, self.rel_obj[j]] += state[:, self.rel_subj[j]] *\
                    h[:, j]

            output[:, i, :] = next_state
            state = next_state

        return output

    @property
    def num_relations(self):
        return len(self.relation_register)

    def register(self, data, target):
        args = self.args
        history = torch.cat((data[:, None, :args.size].long(), target), 1)
        active_nodes = [history[b].sum(0).nonzero()[:, 0] for b in range(history.shape[0])]
        active_rels = set([(j1, j2)
                           for i in range(len(active_nodes))
                           for j1 in active_nodes[i].cpu().numpy()
                           for j2 in active_nodes[i].cpu().numpy()])
        for rel in active_rels:
            if not rel in self.relation_register:
                self.relation_register.append(rel)
                self.rel_subj.append(rel[0])
                self.rel_obj.append(rel[1])
        assert self.num_relations < args.max_relations, 'too many relations'
