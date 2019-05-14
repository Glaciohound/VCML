from glob import glob
import os
import sys
import pickle
import numpy as np

results = []

for filename in glob(os.path.join('../../data/log/',
                                  sys.argv[1], '*')):
    with open(filename, 'rb') as f:
        history, args = pickle.load(f)
    if 'isinstance_right' in history:
        right = np.array(history['isinstance_right'])
    else:
        right = np.array(history['accuracy'])
    results.append((args['name'],
                    {'cur': right[-1], 'mean': right.mean()}
                    ))


def filter(*pieces):
    output = []
    for r in results:
        feasible = True
        for p in pieces:
            if p not in r[0]:
                feasible = False
        if feasible:
            output.append(r)
    return output

from IPython import embed
embed()
