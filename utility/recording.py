#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : recording.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 03.08.2019
# Last Modified Date: 27.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


import os
import numpy as np
import torch
from abc import abstractmethod

from utility.common import make_parent_dir


class History:
    def __init__(self, name, capacity):
        self.capacity = capacity
        self.log = []
        self.n = 0
        self.log_interval = 1
        self.name = name

    def record(self, value):
        self.n += 1
        if self.n % self.log_interval == 0:
            self.log.append(value)
            if len(self.log) == self.capacity:
                self.log = self.log[::2]
                self.log_interval *= 2

    def plot(self, ax, label):
        axis_time = np.arange(len(self.log))
        axis_time *= self.log_interval
        axis_value = np.array(self.log)
        ax.plot(axis_time, axis_value, label=label)

    def state_dict(self):
        ckpt = {
            'n': self.n,
            'log': self.log,
            'log_interval': self.log_interval,
            'name': self.name,
            'capacity': self.capacity,
        }
        return ckpt

    def load_state_dict(self, ckpt):
        self.__dict__.update(ckpt)


class MatrixHistory:
    def __init__(self, name, dims, capacity):
        self.name = name
        self.capacity = capacity
        self.log = np.zeros(dims + (capacity,))
        self.log_indexes = np.zeros(dims[0], dtype=int)

    def record(self, row_ind, value):
        self.log[row_ind, :, self.log_indexes[row_ind] % self.capacity] = value
        self.log_indexes[row_ind] += 1

    def reset(self):
        self.log *= 0
        self.log_indexes *= 0

    def state_dict(self):
        ckpt = {
            'name': self.name,
            'capacity': self.capacity,
            'log': self.log,
            'log_indexes': self.log_indexes,
        }
        return ckpt

    def load_state_dict(self, ckpt):
        self.__dict__.update(ckpt)


class AverageRecording:
    def __init__(self, name, history_capacity):
        self.name = name
        self.current = 0

        self.history = History(
            f'{name}_history', history_capacity)
        self.value_history = History(
            f'{name}_valueHistory', history_capacity)

    @abstractmethod
    def reset(self):
        pass

    def record(self, new_value):
        assert not isinstance(new_value, torch.Tensor), 'not converted'
        new_value = new_value
        self.history.record(new_value)

        self.update_average(new_value)
        self.value_history.record(self.value)

    def __str__(self):
        return '{0}: {1:.3f}'.format(self.name, self.value)

    def state_dict(self):
        return {
            'name': self.name,
            'current': self.current,
            'history': self.history.state_dict(),
            'value_history': self.value_history.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.name = ckpt['name']
        self.current = ckpt['current']
        self.history.load_state_dict(ckpt['history'])
        self.value_history.load_state_dict(ckpt['value_history'])

    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def update_average(self):
        pass

    def plot(self, ax, label):
        self.value_history.plot(ax, label)
        ax.set_xlabel('time')
        ax.set_ylabel('value')
        ax.set_title(self.name)


class ExponentialAverage(AverageRecording):
    def __init__(self, name, capacity, momentum=0.99):
        super().__init__(name, capacity)
        self.momentum = momentum
        self.weight = 0

    def update_average(self, new_value):
        y = self.momentum
        self.current = y * self.current + (1 - y) * new_value
        self.weight = y * self.weight + (1 - y) * 1

    @property
    def value(self):
        if self.weight == 0:
            return 0
        else:
            return self.current / self.weight

    def reset(self):
        self.weight = 0
        self.current = 0

    def state_dict(self):
        ckpt = super().state_dict()
        ckpt.update({
            'weight': self.weight
        })
        return ckpt

    def load_state_dict(self, ckpt):
        super().load_state_dict(ckpt)
        self.weight = ckpt['weight']


class SimpleAverage(AverageRecording):
    def __init__(self, name, capacity):
        super().__init__(name, capacity)
        self.count = 0

    def update_average(self, new_value):
        self.current += new_value
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return self.current / self.count

    def reset(self):
        self.count = 0
        self.current = 0

    def state_dict(self):
        ckpt = super().state_dict()
        ckpt.update({
            'count': self.count
        })
        return ckpt

    def load_state_dict(self, ckpt):
        super().load_state_dict(ckpt)
        self.count = ckpt['count']


class CoeffMatrixRecording:
    def __init__(self, name, dim, history_capacity):
        self.name = name
        self.dim = dim
        self.history_capacity = history_capacity
        self.history = MatrixHistory(
            name,
            (dim, dim),
            history_capacity,
        )

    def record(self, row_ind, value):
        if value.ndim == 2:
            for i in range(value.shape[0]):
                self.history.record(row_ind, value[i])
        else:
            self.history.record(row_ind, value)

    def reset(self):
        self.history.reset()

    def calculate_coeff(self):
        coeff_matrix = np.zeros((self.dim, self.dim))
        log_mat = self.history.log

        for i in range(self.dim):
            x = log_mat[i, i]
            if x.var() > 0:
                for j in range(self.dim):
                    y = log_mat[i, j]
                    coeff = np.polyfit(x, y, 1)[0]
                    coeff_matrix[i, j] = coeff

        return coeff_matrix

    def state_dict(self):
        ckpt = {
            'name': self.name,
            'dim': self.dim,
            'history_capacity': self.history_capacity,
            'history': self.history.state_dict(),
        }
        return ckpt

    def load_state_dict(self, ckpt):
        self.name = ckpt['name']
        self.dim = ckpt['dim']
        self.history_capacity = ckpt['history_capacity']
        self.history.load_state_dict(ckpt['history'])


class AverageGroup:
    def __init__(self, groupname, mode, history, path):
        self.groupname = groupname
        self.mode = mode
        self.history = history
        self.path = path
        if mode == 'exponential':
            self.recording_cls = ExponentialAverage
        elif mode == 'simple':
            self.recording_cls = SimpleAverage

        self.group = {}

    def record(self, value_dict):
        for key in value_dict.keys():
            if key not in self.group:
                self.setup_new(key)

        for key, recording in self.group.items():
            if key in value_dict and value_dict[key] is not None:
                recording.record(value_dict[key])
            else:
                recording.record(recording.value)

    def setup_new(self, key):
        new_name = f'{self.groupname}_{key}'
        self.group[key] = self.recording_cls(new_name, self.history)

    def reset(self):
        for item in self.group.values():
            item.reset()

    def state_dict(self):
        ckpt = {
            'groupname': self.groupname,
            'mode': self.mode,
            'group': {key: recording.state_dict()
                      for key, recording in self.group.items()}
        }
        return ckpt

    def load_state_dict(self, ckpt):
        self.groupname = ckpt['groupname']
        self.mode = ckpt['mode']
        for key, recording in ckpt['group'].items():
            self.setup_new(key)
            self.group[key].load_state_dict(recording)

    def __str__(self):
        return ' '.join([
            '{0}: {1:.2f}'.format(key, recording.value)
            for key, recording in self.group.items()
        ])

    def __len__(self):
        return len(self.group)

    @property
    def values(self):
        return {
            key: recording.value
            for key, recording in self.group.items()
        }

    @property
    def visualize_filename(self):
        return os.path.join(
            self.path,
            'plots',
            f'{self.groupname}.jpg'
        )

    def plot_on_axes(self, key_dict, ax_array, label):
        for key, recording in self.group.items():
            if key not in key_dict:
                key_dict[key] = len(key_dict)
            ind = key_dict[key]
            recording.plot(ax_array[ind], label)

    def savefig(self, fig):
        fig.suptitle(self.groupname)
        fig.tight_layout()
        make_parent_dir(self.visualize_filename)
        fig.savefig(self.visualize_filename)
