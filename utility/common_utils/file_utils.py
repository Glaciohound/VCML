#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : file_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# file system tools


import os
import pickle
import json
import shutil
from shutil import copy2


def make_parent_dir(filename):
    parent_dir = os.path.dirname(filename)
    make_dir(parent_dir)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_empty(path):
    return not os.path.exists(path) or len(os.listdir(path)) == 0


def load(filename):
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            loaded = json.load(f)
    else:
        raise Exception('File not recognized: %s' % filename)
    return loaded


def dump(content, filename):
    if filename.endswith('.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(content, f)
    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(content, f, indent=4)
    else:
        raise Exception('File not recognized: %s' % filename)


def load_knowledge(name, knowledge_type, logger=None, from_source=False):
    filename = os.path.join(
        'knowledge',
        'source' if from_source else '',
        f'{name}_{knowledge_type}.json'
    )
    if os.path.exists(filename):
        knowledge = load(filename)
    else:
        knowledge = None
    if logger is not None:
        if knowledge is not None:
            logger(f'Loading knowledge \"{knowledge_type}\" for {name} '
                   f'length = {len(knowledge)}')
        else:
            logger(f'Loading knowledge \"{knowledge_type}\", but is None')
    return knowledge


def copy_verbose(src, dst, logger=None):
    message = f'copying from {src} to {dst}'
    if logger is not None:
        logger(message)
    else:
        print(message)
    copy2(src, dst)


def copytree_verbose(src, dst, logger=None):
    shutil.copytree(src, dst, copy_function=copy_verbose)
