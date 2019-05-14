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
from model.utils.resnet import Attribute_Network

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.build()

    def build(self):
        self.feature_mlp = self.build_mlp(args.feature_dim, args.embed_dim,
                                          'feature', args.hidden_dim)
        self.resnet_model = Attribute_Network()

    def build_mlp(self, dim_in, dim_out, name, dim_hidden):
        if dim_hidden <= 0:
            return nn.Linear(dim_in, dim_out)
        linear1 = nn.Linear(dim_in, dim_hidden)
        linear2 = nn.Linear(dim_hidden, dim_out)
        setattr(self, name+'_linear1', linear1)
        setattr(self, name+'_linear2', linear2)
        return lambda x: linear2(torch.sigmoid(linear1(x)))
        #return lambda x: linear2(linear1(x))

    def forward(self, data):
        batch_size = data['answer'].shape[0]

        if info.visual_dataset.mode == 'detected':
            feature, recognized = self.resnet_model(data)

        classification = []
        for i in range(batch_size):
            if info.visual_dataset.mode == 'pretrained':
                this = self.feature_mlp(info.to(data['scene'][i]))
            elif info.visual_dataset.mode == 'detected':
                this = self.feature_mlp(recognized[i][1])
                #this = recognized[i][1][:, :args.embed_dim]
            classification.append(this)
        classification = torch.cat(classification, dim=0)

        output = F.log_softmax(classification, 1)
        target = info.to(data['object_classes'])

        return output, target, None, 0.

    def init(self):
        for name, param in self.named_parameters():
            if not name.startswith('resnet_model'):
                if info.new_torch:
                    init.normal_(param, 0, args.init_variance)
                else:
                    init.normal(param, 0, args.init_variance)
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
