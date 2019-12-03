#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : program_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 21.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

import numpy as np


def preprocess_operation(cur):
    operation = cur['operation']
    category = None
    if ' ' in operation:
        splits = operation.split(' ')
        if len(splits) > 2:
            operation = splits[0]
            category = '_'.join(splits[1:])
        else:
            operation, category = splits
    if category == 'rel':
        category = 'relation'

    argument = cur['argument']
    if ' (' in argument:
        argument, obj = argument.split(' (')
        obj = obj.split(')')[0].split(',')
    elif argument == 'scene':
        obj = 'scene'
    else:
        obj = None

    argument = argument[:-1] if argument.endswith(' ') else argument
    if argument == '' and operation not in ['and', 'or', 'same', 'different']:
        argument = 'scene'
    elif argument == '?':
        argument = ''

    return operation, category, argument, obj


# semantic translator for hEmbedding_v2
def semantic2program(program_list):
    output = []

    def add_operation(x, y):
        output.append({'operation': x,
                       'argument': y})

    def convert_operation(x):
        output[-1]['operation'] = x

    for op in program_list:
        operation, category, argument, obj = preprocess_operation(op)

        # object-level reasoning
        if operation == 'select':
            add_operation('select_object', '<NULL>')
            add_operation('filter', argument)

        elif operation == 'filter':
            add_operation('filter', argument)

        elif operation == 'query':
            add_operation('unique_object', '<NULL>')
            add_operation('query', argument)

        elif operation == 'exist':
            add_operation('exist', '<NULL>')

        # concept-level reasoning
        elif operation == 'select_concept':
            add_operation('select_concept', argument)

        elif operation in ['synonym', 'hypernym', 'samekind', 'meronym']:
            add_operation(operation, argument)

        elif operation == 'isinstanceof':
            add_operation('isinstanceof', '<NULL>')

        # classification operation
        elif operation == 'classify':
            add_operation('classify', argument)

        else:
            raise Exception('no such operation supported {}'.format(op))

    return output


def encode_program(program, operations, arguments):
    return np.array([
        [
            operations[op['operation']],
            arguments[op['argument']]
        ]
        for op in program
    ])


def decode_program(program, operations, arguments):
    output = [
        {
            'operation': operations[op[0]],
            'argument': arguments[op[1]]
        }
        for op in program
    ]
    return output
