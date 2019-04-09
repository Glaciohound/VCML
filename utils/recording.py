import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Recording:
    def __init__(self, args, info, keys, name='anonymous', mode='average', momentum=0.99):
        self.args = args
        self.info = info
        self.keys = keys
        self.name = name

        self.values = {k: 0 for k in keys}
        info.log[name] = {k: [] for k in keys}
        self.previous = self.values
        valid_modes = ['average', 'decaying']
        assert mode in valid_modes, 'invalid mode: %s' % mode
        self.mode = mode
        self.momentum = momentum
        self.weight_total = 0
        self.n = 0

    def update(self, values):
        info = self.info
        self.previous = {self.keys[i]: v for i, v in enumerate(values)}
        self.previous_ = {k: float(v) for k, v in self.previous.items()}
        self.weight_total = self.weight_total*self.momentum +\
            (1-self.momentum)
        for k, v in self.values.items():
            new_value = self.previous_[k]
            if self.mode == 'average':
                self.values[k] = (v * self.n + new_value) / (self.n + 1)
            else:
                self.values[k] = (v*self.momentum + (1-self.momentum)*new_value)

        self.n += 1

        for key in self.keys:
            data = self.data
            info.log[self.name][key].append(data[key])

        if self.n % self.args.visualize_time == 0:
            self.visualize()

    def visualize(self):
        for key, values in self.info.log[self.name].items():
            fig, axes = plt.subplots(1)
            axes.plot(values)
            fig.savefig(os.path.join(self.args.visualize_dir,
                                     '{}_{}.jpg'.format(self.name, key)))
            plt.close(fig)

    def clear(self):
        self.weight_total = 0
        self.n = 0
        for key in self.keys:
            self.values[key] = 0
            self.previous[key] = 0

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
        return ', '.join(['%s:%.4f' % (k, v) for k, v in self.data.items()])
