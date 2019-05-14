from glob import glob
import os
import sys
import pickle
import numpy as np

results = {}

for filename in glob(os.path.join('../../data/log/',
                                  sys.argv[1], '*')):
    with open(filename, 'rb') as f:
        history, args = pickle.load(f)
    if 'isinstance_right' in history:
        right = np.array(history['isinstance_right'])
    else:
        right = np.array(history['accuracy'])
    results[args['name']] =\
        {'cur': right[-1], 'mean': right.mean()}


def filter(*pieces):
    output = {}
    for name, item in results.items():
        feasible = True
        for p in pieces:
            if p not in name:
                feasible = False
        if feasible:
            output[name] = item
    return dict(sorted(output.items()))

from IPython import embed
embed()
