import torch
import numpy as np
import pickle

def init_seed(n=0):
    torch.manual_seed(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(n)

def save_log(log_file, *objs):
    with open(log_file, 'wb') as f:
        pickle.dump(objs, f)
