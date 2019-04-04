class Recording:
    def __init__(self, args, info, keys, mode='average', momentum=0.97):
        self.keys = keys
        self.values = {k: 0 for k in keys}
        self.previous = self.values
        valid_modes = ['average', 'decaying']
        assert mode in valid_modes, 'invalid mode: %s' % mode
        self.mode = mode
        self.momentum = momentum
        self.weight_total = 0
        self.n = 0
        self.history = []

    def update(self, values):
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
        self.history.append((self.previous_, str(self)))

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
