#!/usr/bin/env python
# coding=utf-8

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from numpy import random

class RelationModel(nn.Module):
    def __init__(self, args, info=None):
        super(RelationModel, self).__init__()
        self.args = args
        self.info = info

        self.class_embeddingIn = nn.Embedding(args.num_classes, args.embed_dim)
        self.attribute_embeddingIn = nn.Embedding(args.num_attributes, args.embed_dim)
        self.concept_embeddingIn = nn.Embedding(args.max_concepts, args.embed_dim)
        if args.relation_direction=='directed':
            self.class_embeddingOut = nn.Embedding(args.num_classes, args.embed_dim)
            self.attribute_embeddingOut = nn.Embedding(args.num_attributes, args.embed_dim)
            self.concept_embeddingOut = nn.Embedding(args.max_concepts, args.embed_dim)
        else:
            self.class_embeddingOut, self.attribute_embeddingOut, self.concept_embeddingOut =\
                (self.class_embeddingIn, self.attribute_embeddingIn, self.concept_embeddingIn)
        self.operation_embedding = nn.Embedding(len(info.protocol['operations']),
                                                args.operation_dim)

        self.metaMode_init = nn.Parameter(torch.randn(args.embed_dim))
        self.object_init = nn.Parameter(torch.randn(args.embed_dim))
        self.attention_init = nn.Parameter(torch.randn(args.attention_dim))

        self.out_linear1 = nn.Linear(args.attention_dim + args.operation_dim + 2*args.embed_dim,
                                     args.hidden_dim)
        self.out_linear2 = nn.Linear(args.hidden_dim, args.embed_dim)
        self.out_linear = lambda x: self.out_linear2(torch.relu(self.out_linear1(x)))

    def forward(self, data):
        args = self.args
        batch_size = data.answer.shape[0]
        dim_concept = args.max_objects + args.max_concepts

        normalize = lambda x: x/x.sum()
        def embed_without_bg(embedding, x):
            non_bg = (x != -1).astype(int)
            x = torch.LongTensor(x+1).cuda()
            non_bg = torch.Tensor(non_bg).cuda()
            if x.dim() == 1:
                return embedding(x)
            else:
                return (embedding(x) * non_bg[:,:,None]).sum(1)

        objectsIn = self.object_init[None, None].repeat((batch_size, args.max_objects, 1))
        if args.relation_direction == 'directed':
            objectsOut = self.object_init[None, None].repeat((batch_size, args.max_objects, 1))
        else:
            objectsOut = objectsIn

        for i in range(batch_size):
            num_objects = data.scene[i].gt_classes.shape[0]
            objectsIn[i, :num_objects] = embed_without_bg(self.class_embeddingIn, data.scene[i].gt_classes) +\
                embed_without_bg(self.attribute_embeddingIn, data.scene[i].gt_attributes)
            objectsOut[i, :num_objects] = embed_without_bg(self.class_embeddingOut, data.scene[i].gt_classes) +\
                embed_without_bg(self.attribute_embeddingOut, data.scene[i].gt_attributes)

        thoughtIn = torch.cat((self.concept_embeddingIn.weight[None].repeat((batch_size, 1, 1)), objectsIn), 1)
        thoughtOut = torch.cat((self.concept_embeddingIn.weight[None].repeat((batch_size, 1, 1)), objectsOut), 1)

        attention = self.attention_init[None, None].repeat((batch_size, dim_concept, 1))
        operations = self.operation_embedding(torch.LongTensor(data.program[:, :, 0]).cuda())
        arguments = self.concept_embeddingOut(torch.LongTensor(data.program[:, :, 1]).cuda())

        for i in range(args.max_program_length):
            gather = attention.mean(1)
            attention_scalar = attention.abs().sum(2)
            selected = np.stack([
                random.choice(np.arange(args.max_concepts),
                              args.size_attention,
                              p=normalize(attention_scalar[i, :args.max_concepts]\
                                          .cpu().detach().numpy()))
                for i in range(batch_size)
            ])
            def get_selected(original):
                return torch.stack([
                    torch.stack([
                        original[i, selected[i, j]]
                        for j in range(args.size_attention)
                    ]) for i in range(batch_size)
                ])
            attention_selected = get_selected(attention)
            thoughtOut_selected = get_selected(thoughtOut)
            meta = torch.cat((gather, operations[:, i], arguments[:, i]), 1)
            meta = meta[:, None].repeat((1, args.size_attention, 1))
            axon = self.out_linear(torch.cat((thoughtOut_selected, meta), 2))
            new_attention = torch.Tensor(attention.shape).cuda()
            for j in range(dim_concept):
                new_attention[:, j] = ((thoughtIn[:, j:j+1] * axon).mean(2)[:, :, None]\
                                        * attention_selected).mean(1)
            attention = torch.relu(new_attention)

        output_length = attention.pow(2).mean(2)
        output_softmax = F.log_softmax(output_length, 1)
        return output_softmax

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
