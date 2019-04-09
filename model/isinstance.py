#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

class instance_dataset(torch.utils.data.Dataset):
    instance_map = {
        'color': ['red', 'cyan', 'green', 'gray', 'blue', 'purple', 'brown', 'yellow'],
        'shape': ['sphere', 'cube', 'cylinder'],
        'size': ['large', 'small'],
        'material': ['metal', 'rubber']
    }
    instances = [x for k, v in instance_map.items() for x in v]

    def __init__(self, args, info, ckpt):
        self.split = 'train'
        self.args = args
        self.info = info
        model = ckpt['model']
        self.embeddingsIn, self.embeddingsOut, self.embeddingsId =\
            [model['concept_embedding%s.weight' % s].cpu().detach().numpy()
             for s in ['In', 'Out', 'Id']]
        self.protocol = ckpt['protocol']['concepts']
        self.build()

    def index(self, item):
        return [embedding[self.protocol.index(item)]
                for embedding in [self.embeddingsIn,
                                  self.embeddingsOut,
                                  self.embeddingsId]]

    def build(self):
        if self.args.isinstance_mode == 'color_1':
            self.exclude = np.random.choice(self.instance_map['color'], 1)[0]
            self.train_pairs = [(x, y, y in self.instance_map.get(x, []))
                          for x in self.protocol for y in self.protocol
                          if y != self.exclude]
            self.val_pairs = [(x, self.exclude, x == 'color')
                              for x in self.protocol]
        elif self.args.isinstance_mode == 'any_1':
            self.exclude = np.random.choice(self.instances, 1)[0]
            self.train_pairs = [(x, y, y in self.instance_map.get(x, []))
                          for x in self.protocol for y in self.protocol
                          if y != self.exclude]
            self.val_pairs = [(x, self.exclude, self.exclude in self.instance_map.get(x, []))
                              for x in self.protocol]
        elif self.args.isinstance_mode == 'shape_cat':
            self.exclude = self.instance_map['shape']
            self.train_pairs = [(x, y, y in self.instance_map.get(x, []))
                                for x in self.protocol for y in self.protocol
                                if y not in self.exclude and x != 'shape']
            self.val_pairs = [(x, y, y in self.instance_map.get(x, []))
                              for x in self.protocol for y in self.protocol
                              if y in self.exclude or x == 'shape']
        elif self.args.isinstance_mode == 'any_cat':
            self.exclude_cat = np.random.choice(list(self.instance_map), 1)[0]
            self.exclude = self.instance_map[self.exclude_cat]
            self.train_pairs = [(x, y, y in self.instance_map.get(x, []))
                                for x in self.protocol for y in self.protocol
                                if y not in self.exclude and x != self.exclude_cat]
            self.val_pairs = [(x, y, y in self.instance_map.get(x, []))
                              for x in self.protocol for y in self.protocol
                              if y in self.exclude or x == self.exclude_cat]
        else:
            raise Exception('no such task')

        self.build_datasets()

    def build_datasets(self):
        def dataset(pairs):
            output = {'input': [[], [], []], 'output': []}
            for pair in pairs:
                embedding_x = self.index(pair[0])
                embedding_y = self.index(pair[1])
                output['input'][0].append(embedding_y[0])
                output['input'][1].append(embedding_x[1])
                output['input'][2].append(embedding_x[2])
                output['output'].append([1] if pair[2] else [0])

            for k, v in output.items():
                output[k] = np.array(v)
            n = output['output'].shape[0]
            n_true = output['output'].sum()
            n_false = n - n_true
            output['weight'] = 0.5 / (np.ones(n) * n_false +\
                                    output['output'][:, 0] * (n_true - n_false))
            return output

        self.datasets = {'train': dataset(self.train_pairs),
                         'val': dataset(self.val_pairs)}

    def to_split(self, split):
        self.split = split

    def __getitem__(self, index):
        if index != -1:
            return self.datasets[self.split]['input'][:, index],\
                self.datasets[self.split]['output'][index]
        else:
            return self[np.random.choice(len(self), 1)[0]]

    def get_random(self, num):
        datas = [self[i] for i in
                 np.random.choice(np.arange(len(self)), num,
                                  p = self.datasets[self.split]['weight'])]
        x1, x2, x3 = [torch.Tensor([data[0][i] for data in datas])
                      for i in range(3)]
        y = torch.Tensor([data[1] for data in datas])
        return (x1, x2, x3), y

    def __len__(self):
        return self.datasets[self.split]['output'].shape[0]

class InstanceNet(nn.Module):

    def __init__(self, args, info=None):
        super(InstanceNet, self).__init__()
        self.args = args
        self.info = info
        self.build()

    def build(self):
        args = self.args

        def build_mlp(dim_in, dim_hidden, dim_out, name):
            linear1 = nn.Linear(dim_in, dim_hidden)
            linear2 = nn.Linear(dim_hidden, dim_out)
            setattr(self, name+'_linear1', linear1)
            setattr(self, name+'_linear2', linear2)
            return lambda x: linear2(torch.relu(linear1(x)))

        #self.mlp = build_mlp(args.embed_dim + args.identity_dim,
        self.mlp = build_mlp(args.identity_dim,
                             args.isinstance_hidden_dim,
                             1,
                             'isinstance')


    def forward(self, data):
        weight = (data[0][:] * data[1][:]).sum(1)
        weighted_ident = (weight[:, None] * data[2][:])
        #output = self.mlp(torch.cat([data[0][:], weighted_ident], 1))
        output = self.mlp(weighted_ident)
        return output
