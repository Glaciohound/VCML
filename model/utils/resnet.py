#! /usr/bin/env python3
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
import sys
info = sys.info

class Attribute_Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet.resnet34(pretrained=True, incl_gap=False, num_classes=None)
        self.resnet.layer4 = jacnn.Identity()
        self.scene_graph = SceneGraph(256, [None, 256, 256], 16)

    def forward(self, batch):
        features = self.resnet(info.to(batch['image']))
        output = self.scene_graph(features,
                                  info.to(batch['objects']),
                                  info.to(batch['object_lengths']))
        return features, output
