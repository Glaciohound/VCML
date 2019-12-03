#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : resnet.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 20.08.2019
# Last Modified Date: 20.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


# -*- coding: utf-8 -*-
# File   : reasoning_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/06/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch.nn as nn
import jactorch.nn as jacnn
import jactorch.models.vision.resnet as resnet
from .scene_graph import SceneGraph


class ResNetSceneGraph(nn.Module):
    def __init__(self, device, relation=True, dropout_rate=0):
        super().__init__()

        self.device = device
        self.resnet = resnet.resnet34(
            pretrained=True,
            incl_gap=False,
            num_classes=None
        )
        self.resnet.layer4 = jacnn.Identity()
        self.scene_graph = SceneGraph(256,
                                      [None, 512, 512],
                                      16,
                                      relation=relation)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, batch):
        valid_indexes = [
            i for i, length in enumerate(batch['object_length'])
            if length > 0
        ]
        features = self.resnet(batch['image'].to(self.device))
        dropout_features = self.dropout(features)
        if len(valid_indexes) > 0:
            output = self.scene_graph(
                dropout_features[valid_indexes],
                batch['objects'].to(self.device),
                batch['object_length'][valid_indexes].to(self.device)
            )
        else:
            output = []
        output = [
            output[valid_indexes.index(i)]
            if i in valid_indexes
            else (None, None, None)
            for i in range(batch['batch_size'])
        ]
        return features, dropout_features, output
