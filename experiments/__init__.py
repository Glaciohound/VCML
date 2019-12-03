#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : __init__.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


from importlib import import_module

from utility.common import contains


class lazy_import:
    def __init__(self):
        pass

    def __getitem__(self, query):
        import_type, task, name = query

        if import_type == 'build-dataset':
            package = import_module(f'.build_{name}',
                                    'experiments.build_exp')
            return package

        elif import_type == 'run-experiment':
            if task in ['CUB']:
                if contains(name, ['hypernym', 'fewshot']):
                    package = import_module(
                        f'.{name}', 'experiments.cub_hypernym')
                elif 'meronym' in name:
                    package = import_module(
                        f'.{name}', 'experiments.cub_meronym')
                else:
                    raise Exception()
                return package
            if task in ['CLEVR', 'GQA']:
                if contains(name,
                            ['synonym', 'samekind', 'zeroshot',
                             'normal']):
                    package = import_module(
                        f'.{name}', 'experiments.normal')
                    return package
                elif 'debiasing' in name:
                    package = import_module(
                        f'.{name}', 'experiments.debiasing')
                    return package
                else:
                    raise Exception(f'no such module: {name}')
            else:
                raise Exception(f'no such module: {name}')
