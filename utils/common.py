import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm as tqdm_proto

def get_imageId(filename):
    return filename.rstrip('.jpg').rstrip('.png').split('/')[-1]


def union(x, y):
    return list(set(x).union(set(y)))


def pick_one(x):
    return list(x)[0]


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif x.dtype.char in ['d', 'f']:
        return torch.Tensor(x)
    elif x.dtype.char in ['l', 'b']:
        return torch.LongTensor(x)
    else:
        return x

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, torch.autograd.Variable):
        return x.data.cpu().numpy()

def to_normalized(x):
    if isinstance(x, torch.Tensor):
        return F.normalize(x, dim=-1)
    elif isinstance(x, np.ndarray):
        return to_normalized(torch.Tensor(x)).numpy()
    else:
        raise Exception('unsupported type: %s' % str(type(x)))


class tqdm:

    def __init__(self, *arg, pbar_list=None, **kwarg):
        self.pbar = tqdm_proto(*arg, **kwarg)
        self.pbar_list = pbar_list
        pbar_list.append(self)

    def update(self, *arg):
        self.pbar.update(*arg)

    def set_description(self, *arg):
        self.pbar.set_description(*arg)

    def write(self, *arg):
        self.pbar.write(*arg)

    def __iter__(self):
        self.it = iter(self.pbar)
        return self

    def __next__(self):
        item = next(self.it)
        return item

    def monitor(self):
        if self.pbar.n == self.pbar.total:
            self.pbar.close()
            self.pop_self()

    def pop_self(self):
        if self in self.pbar_list:
            self.pbar_list.remove(self)
