#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : question_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 05.08.2019
# Last Modified Date: 31.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# question utility functions

import numpy as np


def tokenize(
    s,
    delim=' ',
    add_start_token=True,
    add_end_token=True,
    punct_to_keep=None,
    punct_to_remove=None,
):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, delim)

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def encode_question(question, word2idx, allow_unk=False):
    seq_idx = question2seq(question, allow_unk)
    seq_idx = [word2idx[x] for x in seq_idx]
    return np.array(seq_idx)


def question2seq(question, allow_unk=False):
    seq_tokens = tokenize(
        question,
        punct_to_keep=[';', ',', '?', '.'],
        punct_to_remove=['-', '_']
    )
    seq_idx = []
    for token in seq_tokens:
        seq_idx.append(token)
    return seq_idx
