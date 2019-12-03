#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : monitor.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 23.07.2019
# Last Modified Date: 18.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# monitor and visualize data

import matplotlib
import numpy as np

from utility.common import dict_gather, get_axes


def load_plt():
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


class Monitor:

    def __init__(self, args, message, control):
        self.args = args
        self.message = message
        self.control = control
        self.coaches = [
            {
                'message': message[i],
                'control': control[i],
            }
            for i in range(len(message))
        ]
        self.i_epoch = None

    def monitor(self):
        self.plt = load_plt()
        in_epoch = self.args.in_epoch

        for i in range(self.args.epochs):
            self.count_epoch()

            if 'train' in in_epoch:
                self._synchronize()
                self._gather()

            if 'val' in in_epoch:
                self._synchronize()
                self._gather()

            if (self.i_epoch + 1) % self.args.test_every == 0:

                if 'test' in in_epoch:
                    self._synchronize()
                    self._gather()

                if 'test_ref' in in_epoch:
                    self._synchronize()
                    self._gather()

            self._synchronize()

    @property
    def num(self):
        return len(self.coaches)

    def count_epoch(self):
        i_epoch = [
            channel.get()
            for channel in self.message
        ][0]
        self.i_epoch = i_epoch

    def _gather(self):
        recordings = [
            channel.get()
            for channel in self.message
        ]
        value_dicts = [item.values
                       for item in recordings]
        value_gather = dict_gather(value_dicts)
        value_mean = {k: v.mean() for k, v in value_gather.items()}
        value_error = {k: np.std(v, ddof=1) for k, v in value_gather.items()}
        print_command = {
            'type': 'log',
            'content': {
                'mean': value_mean,
                'deviation': value_error,
            }
        }
        self._broadcast(print_command)

        n_group = max([len(item) for item in recordings])
        if n_group > 0 and not self.args.silent:
            fig, ax_array = get_axes(n_group, self.plt)
            for ax in ax_array:
                ax.grid()
            key_dict = dict()
            for i, item in enumerate(recordings):
                item.plot_on_axes(key_dict, ax_array, i)
            for ax in ax_array:
                if ax.has_data():
                    ax.legend()
            recordings[0].savefig(fig)
            self.plt.close('all')

    def _broadcast(self, command, coaches=None):
        if coaches is None:
            coaches = self.coaches

        for coach in coaches:
            coach['control'].put(command)

    def _wait(self):
        for coach in self.coaches:
            coach['message'].get()

    def _synchronize(self):
        self._wait()
        self._broadcast({'type': 'let_go'}, self.coaches)
