#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : evaluate.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 23.07.2019
# Last Modified Date: 24.09.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# codes for conducting result analysis

import numpy as np


# Accuracy

def accuracy_classification(output, gt_class, confidence, args):
    yes = (gt_class == 1) * confidence
    no = (gt_class == 0) * confidence
    yes_num = yes.sum()
    no_num = no.sum()

    accuracy = (output * gt_class + (1 - output) * (1 - gt_class))
    yes_acc = (accuracy * yes).sum()
    no_acc = (accuracy * no).sum()

    if args.balance_classification:
        if no_num == 0:
            output = yes_acc / yes_num
        elif yes_num == 0:
            output = no_acc / no_num
        else:
            output = (yes_acc / yes_num + no_acc / no_num) / 2
    else:
        output = (yes_acc + no_acc) / (yes_num + no_num)
    return output
    '''
    total = np.ones_like(gt_class).sum()
    correct = ((output >= 0.5) == gt_class).astype(int).sum()
    return correct / total
    '''


def accuracy_plain(output, answer, confidence, args):
    '''
    return output[answer]
    '''
    if output[answer] == max(output.values()):
        return 1
    else:
        return 0


def accuracy_by_type(outputs, data, args):
    types = set(data['type'])
    batch_size = len(data['type'])

    type_index = {
        q_type: [index for index in range(batch_size)
                 if data['type'][index] == q_type]
        for q_type in types
    }

    type_acc = {
        q_type[:3]+'_acc': RW_given_index(
            outputs, data, type_index[q_type], args
        ).mean()
        for q_type in types
    }

    return type_acc


# RW: Right/Wrong annotations


def total_RW(outputs, data, args):
    return RW_given_index(
        outputs, data, list(range(len(data['type']))), args
    )


def RW_given_index(outputs, data, indexes, args):
    right = np.zeros(len(indexes))
    for ind, i in enumerate(indexes):
        if data['type'][i] == 'classification':
            right[ind] = accuracy_classification(
                outputs[i], data['answer'][i], data['confidence'][i], args
            )
        else:
            right[ind] = accuracy_plain(
                outputs[i], data['answer'][i], data['confidence'][i], args
            )

    return right


def total_accuracy(outputs, data, args):
    return total_RW(outputs, data, args).mean()


def eval(outputs, data, args):
    results = {}
    results.update(accuracy_by_type(outputs, data, args))
    return results
