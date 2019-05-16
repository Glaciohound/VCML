#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init

class UEmbedding(nn.Module):
    def __init__(self, args, info=None):
        super(UEmbedding, self).__init__()
        self.args = args
        self.info = info
        self.build()
        self.init()

    def build(self):
        args = self.args
        self.square_dim = args.embed_dim * args.embed_dim
        rank = args.rank

        self.attribute_embedding = self.build_embedding(args.num_attributes, 2 * self.square_dim,
                                                        'attribute', rank=rank)
        self.feature_linear = self.build_mlp(args.feature_dim, rank, 2 * self.square_dim,
                                             'feature', activate=False)
        self.concept_embedding = self.build_embedding(args.max_concepts, 2 * self.square_dim,
                                                      'concept', rank=rank)
        self.embed_init = nn.Parameter(torch.randn(2 * self.square_dim))

        #self.activation = F.leaky_relu
        self.activation = lambda x: torch.clamp(x, -1, 5) + 0.1 * x

    def build_mlp(self, dim_in, dim_hidden, dim_out, name, activate=True):
        if (not activate and dim_hidden >= min(dim_in, dim_out)) or dim_hidden == 0:
            linear = nn.Linear(dim_in, dim_out)
            setattr(self, name+'_linear_layer', linear)
            return lambda x: linear(x)
        linear1 = nn.Linear(dim_in, dim_hidden)
        linear2 = nn.Linear(dim_hidden, dim_out)
        setattr(self, name+'_linear1', linear1)
        setattr(self, name+'_linear2', linear2)
        if activate:
            return lambda x: linear2(self.activation(linear1(x)))
        else:
            return lambda x: linear2(linear1(x))

    def build_embedding(self, n, dim, name, rank=0):
        if rank == 0 or rank >= dim:
            return nn.Embedding(n, dim)
        else:
            embedding = nn.Embedding(n, rank)
            embedding.requires_grad = False
            linear = nn.Linear(rank, dim)
            setattr(self, name+'_hidden_embedding', embedding)
            setattr(self, name+'_linear', linear)
            return lambda x: linear(embedding(x))

    def init(self):
        for param in self.parameters():
            if param.requires_grad:
                init.normal_(param, 0, self.args.init_variance)

    def forward(self, data):
        args = self.args
        info = self.info
        batch_size = data.answer.shape[0]
        def tensor(x):
            if 'float' in x.dtype.name:
                return torch.Tensor(x).to(info.device)
            else:
                return torch.LongTensor(x).to(info.device)

        num_objects = max([data.scene[i].shape[0] for i in range(batch_size)])
        num_nodes = num_objects + args.max_concepts

        def embed_without_bg(embedding, x):
            non_bg = (x != -1).float()
            x = x+1
            non_bg = non_bg
            return (embedding(x) * non_bg[:,:,None]).sum(1)

        objects = self.embed_init[None, None].repeat((batch_size, num_objects, 1))
        for i in range(batch_size):
            num_here = data.scene[i].shape[0]
            if args.task.startswith('toy'):
                objects[i, :num_here] =\
                    embed_without_bg(self.attribute_embedding, tensor(data.scene[i]))
            else:
                objects[i, :num_here] = self.feature_linear(tensor(data.scene[i]))

        all_nodes = torch.cat((self.concept_embedding(torch.arange(args.max_concepts).to(info.device))
                               [None].repeat((batch_size, 1, 1)), objects), 1)
        all_nodes1 = all_nodes[:, :, :self.square_dim]\
            .view((batch_size, num_nodes, args.embed_dim, -1)).contiguous()
        all_nodes2 = all_nodes[:, :, self.square_dim:]\
            .view((batch_size, num_nodes, args.embed_dim, -1)).contiguous()
        all_nodes = torch.matmul(all_nodes1, all_nodes2)
        #init = 0.3 * torch.randn((1, 1, args.embed_dim, args.embed_dim)).repeat((batch_size, num_nodes, 1, 1)).to(info.device)
        #all_nodes = self.matmul(all_nodes1, init, all_nodes2)
        attention = torch.ones((batch_size, num_nodes)).to(info.device)

        max_program_length = data.program.shape[1]
        arguments = self.concept_embedding(tensor(data.program)[:, :, 1])
        arguments1 = arguments[:, :, :self.square_dim]\
            .view((batch_size, max_program_length, args.embed_dim, -1)).contiguous()
        arguments2 = arguments[:, :, self.square_dim:]\
            .view((batch_size, max_program_length, args.embed_dim, -1)).contiguous()
        operations = tensor(data.program)[:, :, 0]
        history = []

        for i in range(max_program_length):
            working_memory = self.matmul(arguments1[:, i, None].transpose(3, 2), all_nodes, arguments2[:, i, None])
            #working_memory = self.matmul(arguments1[:, i, None].transpose(3, 2), all_nodes1, arguments2[:, i, None])
            #working_memory = self.matmul(arguments1[:, i, None], all_nodes, arguments2[:, i, None])

            is_verify = (operations[:, i, None] == info.protocol['operations', 'verify']).float()
            is_transfer = 1-is_verify

            expanded_memory = working_memory.view((batch_size, num_nodes, -1)) * attention[:, :, None]
            expanded_nodes = all_nodes.view((batch_size, num_nodes, -1))
            similar_to_each = torch.matmul(expanded_memory, expanded_nodes.transpose(2, 1))

            if_verify = self.activation(similar_to_each[:, :, info.protocol['concepts', 'yes']] -
                                        similar_to_each[:, :, info.protocol['concepts', 'no']])

            if_transfer = self.activation(similar_to_each.sum(1))

            attention = is_verify * if_verify + is_transfer * if_transfer

            history.append({'attention': attention})

        history = {k: torch.stack([item[k] for item in history]) for k in history[0].keys()}
        output_softmax = F.log_softmax(attention, 1)

        return output_softmax, history

    def matmul(self, *mats):
        output = mats[0]
        for x in mats[1:]:
            output = torch.matmul(output, x)
        return output
