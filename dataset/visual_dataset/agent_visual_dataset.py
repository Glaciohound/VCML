#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : agent_visual_dataset.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This class is a dataset for visual images and sceneGraph information

import copy
import torch
from tqdm import tqdm
from utility.common import union


class AgentDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger, base_dataset, indexes=None):
        self.args = args
        self.logger = logger
        self.base_dataset = base_dataset
        if indexes is None:
            self.set_indexes(base_dataset.indexes)
        else:
            self.set_indexes(indexes)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.indexes[index]
        else:
            assert index in self.indexes_set or \
                index == self.args.null_image

        return self.base_dataset[index]

    def match_images(self, image_dir):
        self.base_dataset.match_images(image_dir)

    def copy(self, hard=False):
        if hard:
            return AgentDataset(self.args, self.logger,
                                self.base_dataset.copy(), self.indexes)
        else:
            return AgentDataset(self.args, self.logger,
                                self.base_dataset, self.indexes)

    """ utilities """

    def __len__(self):
        return len(self.indexes)

    @classmethod
    def collate(cls, datas):
        return datas

    def set_indexes(self, indexes):
        indexes = copy.deepcopy(indexes)
        self.indexes = indexes
        self.indexes_set = set(indexes)
        return self

    def add_indexes(self, indexes):
        indexes = copy.deepcopy(indexes)
        self.indexes.extend(indexes)
        self.indexes_set = set(self.indexes)
        return self

    def remove_indexes(self, indexes):
        for ind in indexes:
            self.indexes.remove(ind)
        self.indexes_set = set(self.indexes)
        return self

    def set_inputs(self, inputs):
        self.base_dataset.set_inputs(inputs)

    def __contains__(self, query):
        return query in self.indexes_set

    def filter(self, requirement):
        output = self.copy()
        output.set_indexes([
            index for index in self.indexes
            if requirement(self.sceneGraphs[index])
        ])
        return output

    def keys(self):
        return self.indexes

    @property
    def sceneGraphs(self):
        return self.base_dataset.sceneGraphs

    @property
    def local_sceneGraphs(self):
        return {
            key: self.sceneGraphs[key]
            for key in self.indexes
        }

    def get_classification(self, *arg, **kwarg):
        return self.base_dataset.get_classification(*arg, **kwarg)

    def register_concepts(self, concepts):
        self.base_dataset.register_concepts(concepts)

    def filter_concepts(self, concepts):
        self.base_dataset.filter_concepts(concepts)

    def resplit(self, resplit_by):
        if not isinstance(resplit_by, dict):
            self.logger('Splitting by function')
            resplits = {
                index: resplit_by(self.sceneGraphs[index])
                for index in tqdm(self.indexes)
            }
            resplits = {
                split: [index for index, this_split in resplits.items()
                        if this_split == split]
                for split in ['train', 'val', 'test']
            }

        output = {}
        for split in ['train', 'val', 'test']:
            indexes = resplits[split]
            output[split] = self.copy()
            output[split].set_indexes(indexes)
        return output

    def union(self, another):
        union_indexes = union(self.indexes, another.indexes,
                              as_set=True)
        output = self.__copy__()
        output.set_indexes(union_indexes)
        return output

    def mark_splits(self, split_indexes):
        self.base_dataset.mark_splits(split_indexes)
