import numpy as np
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, args, info=None):
        self.info = info
        self.args = args
        getattr(self, 'build_'+args.game)()

    def init_one(self):
        args = self.args
        self.init_state = np.random.randint(args.size, size=[args.size_dataset])
        self.init = np.zeros((args.size_dataset, args.size))
        self.init[np.arange(args.size_dataset), self.init_state] = 1

    def build_rotate(self):
        args = self.args

        self.init_one()
        actions = np.zeros((args.size_dataset, args.length))
        self.input = np.column_stack((self.init, actions))

        self.output = np.zeros((args.size_dataset, args.length, args.size), dtype=np.int)
        for i in range(args.size_dataset):
            for j in range(args.length):
                self.output[i , j, (self.init_state[i]+j+1)%args.size] = 1

    def build_oscillate(self):
        args = self.args

        self.init_one()
        actions = np.zeros((args.size_dataset, args.length))

        self.output = np.zeros((args.size_dataset, args.length, args.size), dtype=np.int)
        for i in range(args.size_dataset):
            state = self.init_state[i]
            mode = 'forward'
            for j in range(args.length):
                if mode == 'forward' and state == args.size - 1:
                    mode = 'backward'
                elif mode == 'backward' and state == 0:
                    mode = 'forward'
                state += 1 if mode == 'forward' else -1
                self.output[i , j, state] = 1
                actions[i, j] = 1 if mode == 'forward' else 0
        self.input = np.column_stack((self.init, actions))

    def build_freely(self):
        args = self.args

        self.init_one()
        actions = np.random.randint(args.num_action, size=(args.size_dataset, args.length))
        self.input = np.column_stack((self.init, actions))

        self.output = np.zeros((args.size_dataset, args.length, args.size), dtype=np.int)
        move_fn = lambda x, action: (action + x) % args.size
        for i in range(args.size_dataset):
            state = self.init_state[i]
            for j in range(args.length):
                state = move_fn(state, actions[i, j])
                self.output[i , j, state] = 1

    def build_locked(self):
        args = self.args

        self.init_one()
        actions = np.zeros((args.size_dataset, args.length))
        self.input = np.column_stack((self.init, actions))

        self.output = np.zeros((args.size_dataset, args.length, args.size), dtype=np.int)
        for i in range(args.size_dataset):
            for j in range(args.length):
                self.output[i, j, self.init_state[i]] = 1

    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def __len__(self):
        return self.output.shape[0]
