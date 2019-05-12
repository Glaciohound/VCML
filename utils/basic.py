import torch
import numpy as np
import os
import pickle

def init_seed():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

def save_log(log_file, *objs):
    with open(log_file, 'wb') as f:
        pickle.dump(objs, f)
