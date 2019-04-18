#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import numpy as np
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class HEmbedding(nn.Module):
    def __init__(self, args, info=None):
        super(HEmbedding, self).__init__()
        self.args = args
        self.info = info
        self.build()
        self.init()

    def build(self):
        args = self.args
        self.square_dim = args.embed_dim * args.embed_dim
        rank = args.rank

        self.attribute_embedding = self.build_embedding(args.num_attributes, args.embed_dim,
                                                        'attribute', rank=rank)
        self.feature_mlp = self.build_mlp(args.feature_dim, 0, args.embed_dim,
                                             'feature')
        self.concept_embedding = self.build_embedding(args.max_concepts, args.embed_dim,
                                                      'concept', rank=rank)
        self.relation_embedding = self.build_embedding(args.max_concepts, self.square_dim,
                                                       'relation', rank=rank, matrix=True)

        #self.activation = F.leaky_relu
        self.activation = torch.sigmoid
        if args.similarity == 'cosine':
            self.similarity = F.cosine_similarity
            self.temperature = 3
            self.train_true_th = 0.8
            self.val_true_th = 0.8
        elif args.similarity == 'square':
            self.similarity = lambda x, y: -(x - y).pow(2).mean(-1)\
                if isinstance(x, torch.Tensor) else self.info.np.power(x-y, 2).mean(-1)
            self.temperature = 0.5
            self.train_true_th = -1
            self.val_true_th = -1

        self.positive = 10
        self.negative = -10

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

    def build_embedding(self, n, dim, name, rank=0, matrix=False):
        if rank == 0 or rank >= dim:
            embedding = nn.Embedding(n, dim)
            setattr(self, name+'_hidden_embedding', embedding)
        else:
            hidden_embedding = nn.Embedding(n, rank)
            #hidden_embedding.requires_grad = False
            linear = nn.Linear(rank, dim)
            setattr(self, name+'_hidden_embedding', hidden_embedding)
            setattr(self, name+'_linear', linear)
            embedding = lambda x: linear(hidden_embedding(x))

        if not matrix:
            return embedding
        else:
            dim_root = np.sqrt(dim)
            if dim_root == int(dim_root):
                dim_root = int(dim_root)
                def matrix_embedding(x):
                    oneD_embedding = embedding(x)
                    matrix_shape = oneD_embedding.shape[:-1] + torch.Size((dim_root, dim_root))
                    return oneD_embedding.view(matrix_shape)
                return matrix_embedding
            else:
                raise Exception('trying to give a matrix embedding for non-square dimension')

    def forward(self, data):
        args = self.args
        info = self.info
        batch_size = data.answer.shape[0]

        def tensor(x):
            if 'float' in x.dtype.name:
                return torch.Tensor(x).to(info.device)
            else:
                return torch.LongTensor(x).to(info.device)

        def embed_without_bg(embedding, x):
            non_bg = (x != -1).float()
            x = x+1
            non_bg = non_bg
            return (embedding(x) * non_bg[:,:,None]).sum(1)

        concept_arguments = self.concept_embedding(tensor(data.program)[:, :, 1])
        relation_arguments = self.relation_embedding(tensor(data.program)[:, :, 1])
        all_concepts = self.concept_embedding(torch.arange(args.max_concepts).to(info.device))
        history = []
        attentions = []

        for i in range(batch_size):
            num_objects = data.scene[i].shape[0]
            num_nodes = num_objects + args.max_concepts
            if args.task.startswith('toy'):
                objects = embed_without_bg(self.attribute_embedding, tensor(data.scene[i]))
            else:
                objects = self.feature_mlp(tensor(data.scene[i]))
            all_nodes = torch.cat([all_concepts, objects])
            attention = torch.ones(num_nodes).to(info.device)

            for j, (op, arg) in enumerate(data.program[i]):
                op_s = info.protocol['operations', int(op)]
                arg_s = info.protocol['concepts', int(arg)]

                if op_s == 'select':
                    if arg_s == 'object_only':
                        attention[:args.max_concepts] = 0
                    elif arg_s == 'concept_only':
                        attention[args.max_concepts:] = 0
                    else:
                        raise Exception('unsupported argument')

                elif op_s == 'verify':
                    attention = torch.min(attention, self.activation((self.similarity(all_nodes, concept_arguments[i, j, None]) - self.true_th)
                                                                   * self.temperature))

                elif op_s == 'choose':
                    attention[:args.max_concepts] = 0
                    attention[arg] = 1

                elif op_s == 'exist':
                    s = attention.max()
                    yes = s * self.positive
                    no = self.positive - yes
                    attention = attention * 1
                    attention[:] = self.negative
                    attention[info.protocol['concepts', 'yes']] = yes
                    attention[info.protocol['concepts', 'no']] = no

                elif op_s == 'transfer':
                    gather = torch.matmul(attention[None], all_nodes)
                    transfered = torch.matmul(gather, relation_arguments[i, j])
                    attention = self.activation((self.similarity(all_nodes, transfered) - self.true_th)
                                              * self.temperature) * attention.max()

                elif op_s in ['<NULL>', '<START>', '<END>', '<UNKNOWN>']:
                    pass

                else:
                    raise Exception('no such operation %s supported' % op_s)

                history.append({'attention': attention})

            attentions.append(attention[:args.max_concepts])

        attentions = torch.stack(attentions)
        program_length = data.program.shape[1]
        history = {k: [torch.stack([history[i*program_length+j][k]
                                    for j in range(program_length)])
                                   for i in range(batch_size)]
                   for k in history[0].keys()}
        output_softmax = F.log_softmax(attentions[:, :args.max_concepts], 1)

        return output_softmax, history

    def matmul(self, *mats):
        output = mats[0]
        for x in mats[1:]:
            output = torch.matmul(output, x)
        return output

    @property
    def true_th(self):
        return self.train_true_th if self.training else self.val_true_th

    def visualize_embedding(self, relation_type=None, normalizing=False):
        info = self.info
        plt = info.plt
        np = info.np
        matrix = self.relation_embedding(torch.LongTensor([info.protocol['concepts', relation_type]]).to(info.device))[0]
        to_visualize = {}
        to_numpy = lambda x: x.cpu().detach().numpy()
        normalize = (lambda x: F.normalize(x, p=2, dim=0)) if normalizing else\
            lambda x: x

        names = self.args.names
        for name in names:
            vec = self.concept_embedding.weight[info.protocol['concepts', name]]
            to_visualize[name] = to_numpy(normalize(vec))
            to_visualize[name+'_convert'] = to_numpy(normalize(torch.matmul(vec, matrix)))
        to_visualize['zero_point'] = to_visualize[names[0]] * 0

        info.vistb(to_visualize, self.args.visualize_dir)
        original = np.array([to_visualize[name] for name in names])
        converted = np.array([to_visualize[name+'_convert'] for name in names])
        plt.matshow(np.array([self.similarity(torch.Tensor(original[i, None]),
                                              torch.Tensor(converted)).numpy()
                              for i in range(len(names))]))
        plt.colorbar()
        plt.savefig(os.path.join(self.args.visualize_dir, 'distance.jpg'))
        plt.clf()
        plt.matshow(np.matmul(original, original.transpose()))
        plt.colorbar()
        plt.savefig(os.path.join(self.args.visualize_dir, 'cosine_ori.jpg'))
        plt.clf()
        plt.matshow(np.matmul(converted, converted.transpose()))
        plt.colorbar()
        plt.savefig(os.path.join(self.args.visualize_dir, 'cosine_converted.jpg'))
        plt.clf()
        return to_visualize, original, converted

    def get_embedding(self, name, relational=False):
        info = self.info
        embedding = self.concept_embedding if not relational else self.relation_embedding
        return embedding(torch.LongTensor([info.protocol['concepts', name]]).to(info.device))

    def init(self):
        for param in self.parameters():
            if param.requires_grad:
                init.normal_(param, 0, self.args.init_variance)
                # init.orthogonal_(param)
        self.new_optimizer()

    def new_optimizer(self):
        info = self.info
        args = self.args
        info.optimizer = optim.Adam(self.parameters(),
                                    lr=args.lr)
        info.scheduler = ReduceLROnPlateau(info.optimizer)

    def save(self, name):
        info = self.info
        torch.save({'model': self.state_dict(),
                    'optimizer': info.optimizer.state_dict(),
                    'scheduler': info.scheduler.state_dict(),
                    'protocol': info.protocol['concepts']},
                   os.path.join(self.args.ckpt_dir, name+'.tar'))

    def load(self, name, retrain=False):
        info = self.info
        args = self.args
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
        info.protocol.reset()
        [info.protocol['operations', o] for o in protocol[0]]
        [info.protocol['concepts', c] for c in protocol[1]]
