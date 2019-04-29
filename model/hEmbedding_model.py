#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import sys
args = sys.args
info = sys.info
import matplotlib.pyplot as plt
import numpy as np

class HEmbedding(nn.Module):
    def __init__(self):
        super(HEmbedding, self).__init__()
        self.build()
        self.init()

    def build(self):
        self.attribute_embedding = self.build_embedding(args.num_attributes, args.embed_dim,
                                                        'attribute', 0)
        self.feature_mlp = self.build_mlp(args.feature_dim, args.embed_dim,
                                             'feature', args.hidden_dim1)
        self.concept_embedding = self.build_embedding(args.max_concepts, args.embed_dim,
                                                      'concept', args.hidden_dim2)
        self.relation_embedding = self.build_embedding(args.max_concepts, args.embed_dim,
                                                       'relation', 0, matrix=args.model=='h_embedding_mul')

        if args.similarity == 'cosine':
            self.similarity = F.cosine_similarity
            self.true_th = 0.9

        self.max_signal = args.temperature
        self.huge_value = 100
        self.scale = lambda x: self.max_signal * (x-self.true_th) / (1-self.true_th)
        self.train_exist_th = 0
        self.val_exist_th = 0

    def build_mlp(self, dim_in, dim_out, name, dim_hidden):
        if dim_hidden <= 0:
            return nn.Linear(dim_in, dim_out)
        linear1 = nn.Linear(dim_in, dim_hidden)
        linear2 = nn.Linear(dim_hidden, dim_out)
        setattr(self, name+'_linear1', linear1)
        setattr(self, name+'_linear2', linear2)
        #return lambda x: linear2(torch.sigmoid(linear1(x)))
        return lambda x: linear2(linear1(x))

    def build_embedding(self, n, dim, name, dim_hidden, matrix=False):
        if dim_hidden <= 0:
            if not matrix:
                hidden_embedding = nn.Embedding(n, dim)
            else:
                hidden_embedding = nn.Embedding(n, dim * dim)
            embedding = hidden_embedding
            setattr(self, name+'_hidden_embedding', hidden_embedding)
        else:
            hidden_embedding = nn.Embedding(n, dim_hidden)
            if not matrix:
                hidden_linear = nn.Linear(dim_hidden, dim)
            else:
                hidden_linear = nn.Linear(dim_hidden, dim * dim)
            setattr(self, name+'_hidden_embedding', hidden_embedding)
            setattr(self, name+'_hidden_linear', hidden_linear)
            embedding = lambda x: hidden_linear(hidden_embedding(x))

        if not matrix:
            return embedding
        else:
            def matrix_embedding(x):
                oneD_embedding = hidden_embedding(x)
                matrix_shape = oneD_embedding.shape[:-1] + torch.Size((dim, dim))
                return oneD_embedding.view(matrix_shape)
            return matrix_embedding

    def forward(self, data):
        batch_size = data.answer.shape[0]

        def tensor(x):
            if 'float' in x.dtype.name:
                return torch.Tensor(x).to(info.device)
            else:
                return torch.LongTensor(x).to(info.device)

        concept_arguments = self.concept_embedding(tensor(data.program)[:, :, 1])
        relation_arguments = self.relation_embedding(tensor(data.program)[:, :, 1])
        all_concepts = self.concept_embedding(torch.arange(args.max_concepts).to(info.device))
        history = []
        attentions = []

        for i in range(batch_size):
            num_objects = data.scene[i].shape[0]
            num_nodes = num_objects + args.max_concepts
            if args.task.startswith('toy'):
                objects = self.embed_without_bg(tensor(data.scene[i]))
            else:
                objects = self.feature_mlp(tensor(data.scene[i]))
            all_nodes = torch.cat([all_concepts, objects])
            all_nodes = F.normalize(all_nodes, p=2, dim=-1)
            attention = torch.ones(num_nodes).to(info.device) * self.huge_value

            for j, (op, arg) in enumerate(data.program[i]):
                op_s = info.protocol['operations', int(op)]
                arg_s = info.protocol['concepts', int(arg)]
                attention = attention * 1

                if op_s == 'select':
                    if arg_s == 'object_only':
                        attention[:args.max_concepts] = -self.huge_value
                    elif arg_s == 'concept_only':
                        attention[args.max_concepts:] = -self.huge_value
                    else:
                        raise Exception('unsupported select argument')

                elif op_s == 'filter':
                    attention = torch.min(attention, self.scale(self.similarity(
                        all_nodes, concept_arguments[i, j, None])))

                elif op_s == 'verify':
                    attention = torch.min(attention, self.scale(self.similarity(
                        all_nodes, concept_arguments[i, j, None])))
                    attention[torch.arange(num_nodes).long() != arg] = -self.huge_value

                elif op_s == 'choose':
                    attention[:args.max_concepts] = -self.huge_value
                    attention[arg] = self.huge_value

                elif op_s == 'exist':
                    attention[len(info.protocol['concepts']): args.max_concepts] = -self.huge_value
                    s = attention.max()
                    yes = s - self.exist_th
                    no = - yes
                    attention = attention * 0
                    attention[:] = -self.huge_value
                    attention[info.protocol['concepts', 'yes']] = yes
                    attention[info.protocol['concepts', 'no']] = no

                elif op_s == 'transfer':
                    gather = torch.matmul(F.softmax(attention, -1), all_nodes)
                    if args.model == 'h_embedding_mul':
                        transferred = torch.matmul(gather, relation_arguments[i, j])
                        to_compare = torch.matmul(all_nodes, relation_arguments[i, j])
                        attention = self.scale(self.similarity(to_compare, transferred[None])-
                                               self.similarity(all_nodes, transferred[None]))
                    else:
                        transferred = gather + relation_arguments[i, j]
                        to_compare = all_nodes
                        attention = self.scale(self.similarity(to_compare, transferred[None]))

                elif op_s in ['<NULL>', '<START>', '<END>', '<UNKNOWN>']:
                    pass

                else:
                    raise Exception('no such operation %s supported' % op_s)

                history.append({'attention': attention})

            attentions.append(attention[:len(info.protocol['concepts'])])

        attentions = torch.stack(attentions)
        program_length = data.program.shape[1]
        history = {k: [torch.stack([history[i*program_length+j][k]
                                    for j in range(program_length)])
                                   for i in range(batch_size)]
                   for k in history[0].keys()}
        if args.loss == 'cross_entropy':
            output = F.log_softmax(attentions[:, :args.max_concepts], 1)
            target = torch.LongTensor(data.answer).to(info.device)
        elif args.loss in ['mse', 'binary']:
            output = attentions[:, :args.max_concepts]
            target = torch.zeros_like(output).to(info.device)
            target[torch.arange(data.answer.shape[0]).long(), data.answer] = 1
            target.requires_grad = False

        return output, target, history

    def embed_without_bg(self, x):
        if isinstance(x, list):
            x = torch.LongTensor(x).to(info.device)

        x = x+1
        return self.attribute_embedding(x).sum(-2)

    def matmul(self, *mats):
        output = mats[0]
        for x in mats[1:]:
            output = torch.matmul(output, x)
        return output

    @property
    def exist_th(self):
        return self.train_exist_th if self.training else self.val_exist_th

    def visualize_embedding(self, relation_type=None, normalizing=False):
        to_visualize = {}
        to_numpy = info.to_numpy
        normalize = info.normalize if normalizing else\
            lambda x: x

        if relation_type is not None:
            matrix = to_numpy(self.get_embedding(relation_type, True))
            if args.model == 'h_embedding_add':
                to_visualize[relation_type] = matrix

        names = args.names
        for name in names:
            vec = to_numpy(self.get_embedding(name))
            to_visualize[name+'_ori'] = normalize(vec)
            if relation_type is not None:
                if args.model == 'h_embedding_mul':
                    to_visualize[name+'_convert'] = normalize(np.matmul(vec, matrix))
                else:
                    to_visualize[name+'_convert'] = normalize(info.normalize(vec) + matrix)

        to_visualize['zero_point'] = list(to_visualize.values())[0] * 0

        original = np.array([to_visualize[name+'_ori'] for name in names])

        if 'isinstance' in args.subtask:
            original = np.concatenate([original, np.array([to_numpy(normalize(self.get_embedding(cat, False)))
                for cat in info.vocabulary.records])])
            for cat in info.vocabulary.records:
                to_visualize[cat+'_concept'] = to_numpy(normalize(self.get_embedding(cat, False)))

        if 'query' in args.subtask and args.model == 'h_embedding_add':
            for cat in info.vocabulary.records:
                to_visualize[cat+'_operation'] = to_numpy(normalize(self.get_embedding(cat, True)))

        info.vistb(to_visualize, args.visualize_dir)

        if relation_type is not None:
            if args.model == 'h_embedding_mul':
                converted = normalize(np.matmul(original, matrix))
            else:
                converted = normalize(original + matrix[None])
            plt.matshow(np.matmul(converted, (converted-original).transpose()))
            plt.colorbar()
            plt.savefig(os.path.join(args.visualize_dir, 'distance.jpg'))
            plt.cla()

            plt.matshow(np.matmul(converted, converted.transpose()))
            plt.colorbar()
            plt.savefig(os.path.join(args.visualize_dir, 'cosine_converted.jpg'))
            plt.cla()

            plt.matshow(matrix[None] if matrix.ndim < 2
                        else matrix)
            plt.colorbar()
            plt.savefig(os.path.join(args.visualize_dir, relation_type+'_matrix.jpg'))
            plt.cla()
            plt.close()
        else:
            converted = None

        plt.matshow(np.matmul(original, original.transpose()))
        plt.colorbar()
        plt.savefig(os.path.join(args.visualize_dir, 'cosine_ori.jpg'))
        plt.clf()
        return to_visualize, original, converted

    def visualize_query(self, queried):
        to_visualize = {}
        to_numpy = info.to_numpy
        normalize = info.normalize

        obj_base = [info.vocabulary[np.random.choice(info.vocabulary[cat], 1)[0]]
                    for cat in info.vocabulary.records
                    if cat != queried]

        operator = to_numpy(self.get_embedding(queried, True))
        if args.model == 'h_embedding_add':
            to_visualize[queried] = operator

        for at in info.vocabulary[queried]:
            to_visualize['obj_'+at] = to_numpy(normalize(self.embed_without_bg(
                obj_base + [info.vocabulary[at]])))
            to_visualize[at] = to_numpy(normalize(self.get_embedding(at)))
            if args.model == 'h_embedding_add':
                to_visualize[at+'_queried'] = to_visualize['obj_'+at] + operator
            else:
                to_visualize[at+'_queried'] = np.matmul(to_visualize['obj_'+at], operator)

        to_visualize['zero_point'] = list(to_visualize.values())[0] * 0
        to_visualize = dict(sorted(to_visualize.items()))

        vectors = normalize(np.stack(list(to_visualize.values())))
        plt.matshow(np.matmul(vectors, vectors.transpose()))
        plt.colorbar()
        plt.savefig(os.path.join(args.visualize_dir, 'queried.jpg'))
        plt.cla()

        return to_visualize

    def get_embedding(self, name, relational=False):
        embedding = self.concept_embedding if not relational else self.relation_embedding
        return embedding(torch.LongTensor([info.protocol['concepts', name]]).to(info.device))[0]

    def init(self):
        for name, param in self.named_parameters():
            init.normal_(param, 0, args.init_variance)
            # init.orthogonal_(param)
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
