#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : collate_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 01.09.2019
# Last Modified Date: 01.09.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


import numpy as np
from utility.common import to_tensor

'''
'stack':
    stack the batch as an array,
    no matter if they are of the same shape or not.
'pad-stack':
    pad the batch into the same shape, and then stack them
'concat':
    concatenate batch along the specified dimension
'list':
    doing nothing but link the batch as a list, tensorizing them if required

'''


def sanity_check_on_keys(datas, setting):
    keys = datas[0].keys()
    assert set(keys).issubset(set(setting.keys())),\
        'mismatching collate setting and data'
    # for data in datas:
    #     assert set(data.keys()) == set(keys),\
    #         'mismatch among data'


def pad_stack(data, setting):
    ndim = data[0].ndim
    dims = [max([data.shape[i] for data in data])
            for i in range(ndim)]
    result = np.zeros(shape=(len(data),) + tuple(dims),
                      dtype=data[0].dtype)

    if 'pad_value' in setting:
        def fill_fn(*args):
            return setting['pad_value']
    else:
        fill_fn = setting['pad_fn']

    if ndim == 1:
        for i, data in enumerate(data):
            result[i, :data.shape[0]] = data
            for j in range(data.shape[0], dims[0]):
                result[i, j] = fill_fn(j)

    elif ndim == 2:
        for i, data in enumerate(data):
            result[i, :data.shape[0], :data.shape[1]] = data
            for j in range(data.shape[0], dims[0]):
                for k in range(data.shape[1], dims[1]):
                    result[i, j, k] = fill_fn(j, k)

    else:
        raise Exception('n-dimension unsupported: %d' % ndim)

    return result


class collateFn:
    def __init__(self, setting):
        self.setting = setting

    def __call__(self, datas):
        setting = self.setting
        if not isinstance(datas, list):
            datas = [datas]
        batch_size = len(datas)

        keys = datas[0].keys()
        sanity_check_on_keys(datas, setting)
        output = {
            'question_lengths':
            [data['question_encoded'].shape[0] for data in datas]
        }

        datas = {k: [data.get(k, None) for data in datas] for k in keys}
        for k in keys:
            setting_k = setting[k]

            if setting_k['type'] == 'stack':
                result = np.array(datas[k])
                if setting_k['tensor']:
                    result = to_tensor(result)

            elif setting_k['type'] == 'pad-stack':
                result = pad_stack(datas[k], setting_k)
                if setting_k['tensor']:
                    result = to_tensor(result)

            elif setting_k['type'] == 'concat':
                result = np.concatenate(datas[k], axis=setting_k['axis'])
                if setting_k['tensor']:
                    result = to_tensor(result)

            elif setting_k['type'] == 'list':
                if setting_k['tensor']:
                    result = [to_tensor(item) for item in datas[k]]
                else:
                    result = datas[k]

            output[k] = result

        output['batch_size'] = batch_size
        return output
