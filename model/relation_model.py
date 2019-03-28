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
#from numpy import random

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
        elif args.relation_direction == 'undirected':
            self.class_embeddingOut, self.attribute_embeddingOut, self.concept_embeddingOut =\
                (self.class_embeddingIn, self.attribute_embeddingIn, self.concept_embeddingIn)
        self.operation_embedding = nn.Embedding(len(info.protocol['operations']),
                                                args.operation_dim)

        self.meta_init = nn.Parameter(torch.randn(args.attention_dim))
        self.object_init = nn.Parameter(torch.randn(args.embed_dim))
        self.attention_init = nn.Parameter(torch.randn(args.attention_dim))

        def build_linear(dim_in, dim_hidden, dim_out, name):
            linear1 = nn.Linear(dim_in, dim_hidden)
            linear2 = nn.Linear(dim_hidden, dim_out)
            setattr(self, name+'_linear1', linear1)
            setattr(self, name+'_linear2', linear2)
            return lambda x: linear2(torch.relu(linear1(x)))

        self.axon_linear = build_linear(args.attention_dim+args.embed_dim,
                                        args.hidden_dim,
                                        args.embed_dim,
                                        'axon')
        self.meta_linear = build_linear(args.embed_dim+args.attention_dim,
                                        args.hidden_dim,
                                        args.attention_dim,
                                        'meta')

    def forward(self, data):
        args = self.args
        info = self.info
        batch_size = data.answer.shape[0]
        dim_concept = args.max_objects + args.max_concepts

        is_mode = (data.program[:, :, 0] == info.protocol['operations', 'mode']).astype(int)
        is_insert = (data.program[:, :, 0] == info.protocol['operations', 'insert']).astype(int)
        is_transfer = (data.program[:, :, 0] == info.protocol['operations', 'transfer']).astype(int)

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
            objectsIn[i, :num_objects] = (embed_without_bg(self.class_embeddingIn, data.scene[i].gt_classes) +\
                                          embed_without_bg(self.attribute_embeddingIn, data.scene[i].gt_attributes))\
                [:args.max_objects]
            objectsOut[i, :num_objects] = (embed_without_bg(self.class_embeddingOut, data.scene[i].gt_classes) +\
                                           embed_without_bg(self.attribute_embeddingOut, data.scene[i].gt_attributes))\
                [:args.max_objects]

        thoughtIn = torch.cat((self.concept_embeddingIn.weight[None].repeat((batch_size, 1, 1)), objectsIn), 1)
        thoughtOut = torch.cat((self.concept_embeddingOut.weight[None].repeat((batch_size, 1, 1)), objectsOut), 1)
        meta = self.meta_init[None].repeat((batch_size, 1))

        attention = self.attention_init[None, None].repeat((batch_size, dim_concept, 1))
        arguments = self.concept_embeddingOut(torch.LongTensor(data.program[:, :, 1]).cuda())
        attentions = [attention]

        for i in range(args.max_program_length):
            '''
            attention_scalar = attention.abs().sum(2)
            normalize = lambda x: x/x.sum()
            selected = np.stack([
                random.choice(np.arange(args.max_concepts),
                              args.size_attention,
                              p=normalize(attention_scalar[i, :args.max_concepts]\
                                          .cpu().detach().numpy()))
                for i in range(batch_size)
            ])
            def get_selected(original):
                return torch.stack([
                    original[i, selected[i]] for i in range(batch_size)
                ])
            attention_selected = get_selected(attention)
            thoughtOut_selected = get_selected(thoughtOut)
            '''

            is_mode_ = torch.Tensor(is_mode[:, i, None]).to(args.device)
            is_insert_ = torch.Tensor(is_insert[:, i, None, None]).to(args.device)
            is_transfer_ = torch.Tensor(is_transfer[:, i, None, None]).to(args.device)

            new_meta = self.meta_linear(torch.cat((meta, arguments[:, i]), 1))
            meta = is_mode_ * new_meta + (1-is_mode_) * meta
            #meta = meta[:, None].repeat((1, args.size_attention, 1))
            meta_broadcast = meta[:, None].repeat((1, dim_concept, 1))

            attention_insert = meta_broadcast * (arguments[:, i, None] * thoughtOut).mean(2)[:, :, None]

            #axon = self.out_linear(torch.cat((thoughtOut_selected, meta), 2))
            axon = self.axon_linear(torch.cat((thoughtOut, meta_broadcast), 2))
            attention_transfer = torch.Tensor(attention.shape).cuda()
            for j in range(dim_concept):
                attention_transfer[:, j] = ((thoughtIn[:, j, None] * axon).sum(2)[:, :, None]\
                                        * attention).mean(1)

            attention = attention + is_insert_ * attention_insert + is_transfer_ * attention_transfer
            attentions.append(attention)

        output_length = attention.pow(2).mean(2)
        output_softmax = F.log_softmax(output_length, 1)
        return output_softmax, attentions