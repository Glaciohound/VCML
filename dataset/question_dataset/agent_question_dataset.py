#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : agent_question_dataset.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 16.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file defined a class for agenting the question dataset

import torch
import copy
from utility.common import random_choice, arange
from dataset.dataloader import get_dataloader


class AgentDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, indexes=None):
        self.base_dataset = base_dataset
        self.args = self.base_dataset.args

        if indexes is None:
            self.set_indexes(base_dataset.indexes)
        else:
            self.set_indexes(indexes)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.indexes[index]
        else:
            assert index in self.indexes_dict

        return self.base_dataset[index]

    def __len__(self):
        return len(self.indexes)

    def set_indexes(self, indexes):
        indexes = copy.deepcopy(indexes)
        self.indexes = indexes
        self.indexes_dict = dict(zip(indexes, arange(len(indexes))))

    def resplit(self):
        output = {}
        for split in ['train', 'val', 'test']:
            indexes = [index for index in self.indexes
                       if index[0] == split]
            output[split] = self.copy()
            output[split].set_indexes(indexes)
        return output

    def __copy__(self):
        raise NotImplementedError()

    def copy(self, deep=False):
        if not deep:
            return AgentDataset(self.base_dataset, self.indexes)
        else:
            return AgentDataset(self.base_dataset.copy(), self.indexes)

    def filter(self, requirement, inplace=False):
        if not inplace:
            output = self.copy()
        else:
            output = self

        filtered_indexes = [
            ind for ind in self.indexes
            if requirement(self.base_questions[ind])
        ]
        output.set_indexes(filtered_indexes)
        return output

    def random_choice(self, requirement, num=-1, balance_fn=None):
        if isinstance(num, float):
            num = int(num)

        output = self.copy()

        filtered_indexes = random_choice(
            self.indexes,
            num,
            requirement,
            balance_fn,
            on_value=self.base_dataset.questions,
        )

        output.set_indexes(filtered_indexes)
        return output

    def get_dataloader(self):
        return get_dataloader(
            self, self.base_dataset.collate_fn, self.args)

    def load_indexes(self, indexes):
        if not isinstance(indexes, list):
            indexes = [indexes]
        questions = [
            self[ind] for ind in indexes
        ]
        return self.base_dataset.collate_fn(questions)

    @property
    def questions(self):
        """
        This function returns questions dict according to self.indexes,
        however as some question keys may duplicate, repeated questions
        may only apper once
        """
        return {
            ind: self.base_dataset.questions[ind]
            for ind in self.indexes
        }

    @property
    def question_list(self):
        """
        This function returns questions list according to self.indexes """
        return [
            self.base_dataset.questions[ind]
            for ind in self.indexes
        ]

    @property
    def base_questions(self):
        return self.base_dataset.questions

    @property
    def program_translator(self):
        return self.base_dataset.program_translator

    def union(self, another, inplace=False):
        if self.base_dataset != another.base_dataset:
            new_base = self.base_dataset.union(
                another.base_dataset, inplace=inplace
            )

        union_indexes = self.indexes + another.indexes
        output = new_base.get_agent()
        output.set_indexes(union_indexes)
        return output

    def full(self):
        output = self.copy()
        output.set_indexes(self.base_dataset.indexes)
        return output

    def state_dict(self):
        ckpt = {
            'base_dataset': self.base_dataset.state_dict(),
            'indexes': self.indexes,
        }
        return ckpt

    def load_state_dict(self, ckpt):
        if 'base_dataset' in ckpt:
            self.base_dataset.load_state_dict(ckpt['base_dataset'])
        self.set_indexes(ckpt['indexes'])

    def load_parts(self, *arg, **kwarg):
        self.base_dataset.load_parts(*arg, **kwarg)
