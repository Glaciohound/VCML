import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
args = sys.args
info = sys.info

class Recording:
    length_limit = 70
    def __init__(self, name='anonymous', mode='average', momentum=0.99):
        self.args = args
        self.info = info
        self.values = {}
        self.history = {}
        self.log = {}
        self.name = name

        self.previous = self.values
        valid_modes = ['average', 'decaying']
        assert mode in valid_modes, 'invalid mode: %s' % mode
        self.mode = mode
        self.momentum = momentum
        self.weight_total = 0
        self.n = 0

    def update(self, value_dict):
        for key in value_dict:
            if not key in self.values:
                self.values[key] = 0
                self.history[key] = []
                self.log[key] = []
        self.previous = value_dict
        self.previous_float = {k: float(v) for k, v in self.previous.items()}

        self.weight_total = self.weight_total*self.momentum +\
            (1-self.momentum)

        for k, new_value in self.previous_float.items():
            self.log[k].append(new_value)
            v = self.values[k]
            if self.mode == 'average':
                #self.values[k] = (v * self.n + new_value) / (self.n + 1)
                self.values[k] = np.array(self.log[k]).mean()
            else:
                self.values[k] = (v*self.momentum + (1-self.momentum)*new_value)

        self.n += 1

        for k, data_v in self.data.items():
            self.history[k].append(data_v)
        if self.n % self.args.visualize_interval == 0:
            self.visualize()

    def visualize(self):
        for key, values in self.history.items():
            fig, axes = plt.subplots(1)
            axes.plot(values)
            if 'loss' in key:
                axes.set_yscale('log')
            else:
                axes.set_yscale('linear')
            fig.savefig(os.path.join(self.args.visualize_dir,
                                     '{}_{}.jpg'.format(self.name, key)))
            plt.close(fig)

    def clear(self):
        self.weight_total = 0
        self.n = 0
        for key in self.values:
            self.values[key] = 0
            self.log[key] = []

    @property
    def data(self):
        if self.mode == 'decaying':
            if self.weight_total == 0:
                return {k: 0 for k in self.keys()}
            return {k: v / self.weight_total
                    for k, v in self.values.items()}
        else:
            return self.values

    def __str__(self):
        long_str, short_str = self.strings()
        return long_str if len(long_str) <= self.length_limit else short_str

    def strings(self):
        return (', '.join(['%s:%.4f' % (k, v) for k, v in self.data.items()]),
                ', '.join(['%s:%.3f' % (k[0]+k[-1], v) for k, v in self.data.items()]))
