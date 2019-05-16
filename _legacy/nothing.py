import json
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob

def refurbish(name):
    print(name)
    with open(name, 'r') as f:
        old_scene = json.load(f)

    new_scene = {'info': old_scene['info'], 'scenes':[]}
    for scene in tqdm(old_scene['scenes']):
        new = {}
        for k in ['image_index', 'image_filename']:
            new[k] = scene[k]
        new['objects'] = []
        for obj in scene['objects']:
            new_obj = {}
            for k, v in obj.items():
                new_obj[k] = v if isinstance(v, str) else np.array(v)
            new['objects'].append(new_obj)
        new_scene['scenes'].append(new)

    with open(name.split('.')[1][1:] + '.pkl', 'wb') as f:
        pickle.dump(new_scene, f)

filenames = glob('./*json')
for name in reversed(filenames):
    refurbish(name)
