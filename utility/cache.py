#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : cache.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 06.08.2019
# Last Modified Date: 30.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


import os
from utility.common import load, dump, make_parent_dir


class Cache:
    def __init__(self, name, logger, args):
        self.name = name
        self.logger = logger
        self.args = args

        if self.args.clear_cache:
            logger('Clearing cache: {self.filename}')
            self.rm_file(empty_ok=True)

    def exist(self):
        output = os.path.exists(self.filename)
        return output

    def load(self):
        self.logger(f'Using cache: {self.filename}')
        output = load(self.filename)
        return output

    def cache(self, obj):
        self.logger(f'Saving cache: {self.filename}')
        make_parent_dir(self.filename)
        dump(obj, self.filename)
        self.obj = obj

    def rm_file(self, empty_ok=False):
        if empty_ok and not os.path.exists(self.filename):
            return
        os.remove(self.filename)

    @property
    def filename(self):
        output = os.path.join(
            self.args.cache_dir, self.name+'.pkl'
        )
        return output

    def __enter__(self):
        if self.exist():
            self.obj = self.load()
        else:
            self.logger(f'Cache not found: {self.filename}')
            self.obj = None
        return self

    def __exit__(self, _type, _value, _traceback):
        pass
