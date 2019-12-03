#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : dataset_config.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 05.08.2019
# Last Modified Date: 18.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from dataset.visual_dataset.utils import \
    split_visual, sceneGraph_utils


dataset_config_register = {
    'GQA': {
        'visual_split_fn': split_visual.split_train_val,
        'scene_process': sceneGraph_utils.gqa_post_process,
    },
    'CLEVR': {
        'visual_split_fn': split_visual.split_train_val,
        'scene_process': sceneGraph_utils.clevr_post_process,
    },
    'CUB': {
        'visual_split_fn': split_visual.cub_split,
        'scene_process': sceneGraph_utils.cub_post_process,
    },
}
