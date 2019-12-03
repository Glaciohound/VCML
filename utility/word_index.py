#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : word_index.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 26.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

import numpy as np
import os
from utility.common \
    import make_dir, load, dump, contains
from copy import deepcopy

special_tokens = [
    '<NULL>', '<START>', '<END>', '<UNK>',
]


class Tools:
    """
    A kit-like class, for storing all Word_Index instances
    """
    def __init__(self, kit_config, logger):
        self.logger = logger
        self.dict = {}
        for name, config in kit_config.items():
            item = Word_Index(name, config['use_special_tokens'],
                              logger)
            setattr(self, name, item)
            self.dict[name] = item

    def save(self, path):
        local_path = os.path.join(path, 'word2index')
        make_dir(local_path)
        for item in self.dict.values():
            item.save(local_path)

    def load(self, path):
        local_path = os.path.join(path, 'word2index')
        for item in self.dict.values():
            item.load(local_path)

    def state_dict(self):
        ckpt = {name: item.state_dict()
                for name, item in self.dict.items()}
        return ckpt

    def load_state_dict(self, ckpt):
        for name, item in self.dict.items():
            item.load_state_dict(ckpt[name])

    def count(self, AinB_config):
        for name, item in self.dict.items():
            num = len(item)
            setattr(self, 'n_'+name, num)
        for A, B in AinB_config:
            AinB = A_in_B_dict(self.dict[A], self.dict[B])
            setattr(self, A+'_in_'+B, AinB)


class Word_Index:
    def __init__(self, name, use_special_tokens, logger):
        self.name = name
        self.use_special_tokens = use_special_tokens
        self.logger = logger
        self.init()

    def init(self):
        if self.use_special_tokens:
            self.records = deepcopy(special_tokens)
        else:
            self.records = []

        self.make_dict()

    def load(self, path):
        filepath = os.path.join(path, self.name+'.pkl')
        ckpt = load(filepath)
        self.load_state_dict(ckpt)
        self.logger(f'loading Word_Index: {self.name}')
        self.logger(f'from file {filepath}, length={len(self)}',
                    resume=True)

    def save(self, path):
        filepath = os.path.join(path, self.name+'.pkl')
        self.logger(f'saving Word_Index: {self.name}, '
                    f'length={len(self)}')
        dump(self.records, filepath)

    def state_dict(self):
        ckpt = {'records': self.records}
        return ckpt

    def load_state_dict(self, ckpt):
        if isinstance(ckpt, dict):
            self.records = ckpt['records']
        else:
            self.records = ckpt
        self.make_dict()

    def reset(self, items):
        self.init()
        for item in items:
            self.register(item)

    def make_dict(self):
        self.dict = {item: i for i, item in enumerate(self.records)}
        self.len = len(self.records)

    def __getitem__(self, query):
        if isinstance(query, int):
            if query < len(self):
                return self.records[query]
            else:
                if self.use_special_tokens:
                    return '<UNK>'
                else:
                    raise Exception('index out of range')

        elif isinstance(query, str):
            if query in self.dict:
                return self.dict[query]
            else:
                if self.use_special_tokens:
                    return self.dict['<UNK>']
                else:
                    raise Exception(f'{query} not in word2index {self.name}')
        elif isinstance(query, list):
            return np.array(self.records)[query].tolist()
        else:
            raise Exception(f'invalid query: {query}')

    def register(self, item, multiple=False):
        if multiple:
            for y in item:
                self.register(y)
        else:
            assert isinstance(item, str)
            if item not in self:
                self.records.append(item)
                self.dict[item] = len(self)
                self.len += 1

    def register_special(self):
        self.register(special_tokens, True)

    def register_related(self, stats):
        for key, values in stats.items():
            if key in self or contains(self, values):
                self.register(key)
                self.register(values, multiple=True)

    def __len__(self):
        return self.len

    def __contains__(self, item):
        return item in self.dict

    def __iter__(self):
        return iter(self.records)

    def indexes(self):
        return list(range(len(self)))

    def filter_out(self, forbidden):
        remained = [
            x for x in self.records if x not in forbidden
        ]
        self.reset(remained)

    def filter_most_frequent(
        self,
        word_count,
        num=-1,
    ):
        if num == -1:
            return
        else:
            remained = set()
            n = 0
            for word, count in sorted(
                word_count.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if word in self:
                    remained.add(word)
                    n += 1
                if n == num:
                    break

            self.reset(remained)

    def get_indexes(self, items):
        return [
            self.dict[item]
            for item in items
        ]


def A_in_B_dict(a, b):
    return np.array([
        b[x] if x in b
        else -1
        for x in a
    ])
