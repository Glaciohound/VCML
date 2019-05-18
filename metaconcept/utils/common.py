import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm as tqdm_proto

from metaconcept import info
from metaconcept.utils.embedding.visualize_tb import visualize_word_embedding_tb as vistb_original


def get_imageId(filename):
    return filename.rstrip('.jpg').rstrip('.png').split('/')[-1]


def union(x, y):
    return list(set(x).union(set(y)))


def pick_one(x, requirement=None, on_value=False):
    if requirement:
        if not on_value:
            x = [y for y in x if requirement(y)]
        else:
            x = [k for k in x if requirement(x[k])]
    return list(x)[0]


def random_one(x, requirement=None, on_value=False, num=1, **kwarg):
    if requirement:
        if not on_value:
            x = [y for y in x if requirement(y)]
        else:
            x = [k for k in x if requirement(x[k])]
    if num == 1:
        return np.random.choice(list(x), 1, **kwarg)[0]
    else:
        return np.random.choice(list(x), num, **kwarg)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        if isinstance(x[0], float):
            return torch.Tensor(x)
        elif isinstance(x[0], int):
            return torch.LongTensor(x)
        else:
            return x
    elif isinstance(x, np.ndarray):
        if x.dtype.char in ['d', 'f']:
            return torch.Tensor(x)
        elif x.dtype.char in ['l', 'b']:
            return torch.LongTensor(x)
        else:
            return x
    elif isinstance(x, int) or isinstance(x, float) \
            or np.isscalar(x):
        return torch.tensor(x)
    else:
        return x


def matmul(*mats):
    output = mats[0]
    for x in mats[1:]:
        if isinstance(output, torch.Tensor):
            output = torch.matmul(output, x)
        else:
            output = np.matmul(output, x)
    return output


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


def min_fn(*xs):
    output = xs[0]
    for x in xs[1:]:
        if info.new_torch:
            output = torch.min(output, x)
        else:
            output = output.min(x)
    return output


def max_fn(x, y):
    if info.new_torch:
        return torch.max(x, y)
    else:
        return y.max(x)


class tqdm:
    def __init__(self, *arg, **kwarg):
        self.pbar = tqdm_proto(*arg, **kwarg)
        if not hasattr(info, 'pbars'):
            info.pbars = []
        info.pbars.append(self)
        for pbar in info.pbars:
            pbar.monitor()

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
        if self in info.pbars:
            info.pbars.remove(self)


def contains(x, elements):
    for e in elements:
        if e in x:
            return True
    return False


def equal_ratio(x, y):
    match = equal_items(x, y)
    return match.mean()


def equal_items(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        match = (np.array(x) == np.array(y)).astype(float)
    else:
        match = (x == y).float()
    return match


def recall(x, y):
    match = to_numpy(x) * to_numpy(y)
    return match.sum() / y.sum()


def arange(*arg):
    return list(range(*arg))


def vistb(dicts, visualize_dir, dim=None):
    if dim:
        dicts = {k: v[dim] for k, v in dicts.items()}
    vistb_original(dicts, visualize_dir)


def logit_and(x, y):
    max_ = max_fn(x, y)
    min_ = min_fn(x, y)
    residue_true = -min_fn(max_, max_ - min_)

    return min_ - F.softplus(residue_true)


def logit_exist(x, y):
    return x + F.softplus(-y)


def log_and(x, y):
    return log(x) + log(y)


def log_or(x, y):
    return -logit_and(-x, -y) + log_and(-x, -y)
    # return log(max_fn(x, y))


def log(x):
    return -F.softplus(-x)


def log_xor(x, y):
    return F.softplus(y - x) - F.softplus(-x) - F.softplus(y)


def log_xand(x, y):
    return log_xor(-x, y)


def logit_xor(x, y):
    return x + F.softplus(y - x) - F.softplus(y + x)


def logit_xand(x, y):
    return -logit_xor(x, y)

