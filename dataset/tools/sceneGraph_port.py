import os
import sys
args = sys.args
info = sys.info
import json
import pickle
from glob import glob
from functools import reduce
from utils.common import union, pick_one, get_imageId


def load_sceneGraphs(filename):
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
            if 'scenes' in loaded:
                loaded = loaded['scenes']
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            loaded = json.load(f)
            if 'scenes' in loaded:
                loaded = loaded['scenes']
    sceneGraphs = {}
    for scene in loaded:
        image_id, default = default_scene(scene['image_filename'])
        scene.update(default)
        for k, v in scene.items():
            if k != 'objects':
                scene[k] = v
            else:
                scene['objects'] =\
                    {str(i): obj for i, obj in enumerate(scene['objects'])}
                for obj in scene['objects'].values():
                    for cat, attr in obj.items():
                        if isinstance(attr, str):
                            info.vocabulary[cat, attr]
        sceneGraphs[image_id] = scene

    return sceneGraphs


def merge_sceneGraphs(x, y):
    sceneGraphs = {}
    for image_id in union(x.keys(), y.keys()):
        if image_id not in x:
            sceneGraphs[image_id] = y[image_id]
        elif image_id not in y:
            sceneGraphs[image_id] = x[image_id]
        else:
            scene = {}
            scene_x = x[image_id]
            scene_y = y[image_id]
            for k in union(scene_x.keys(), scene_y.keys()):
                if k != 'objects':
                    scene[k] = scene_x[k] if k in scene_x else scene_y[k]
                else:
                    scene[k] = max(scene_x[k], scene_y[k],
                                   key=lambda z: len(pick_one(z)))
            sceneGraphs[image_id] = scene
    return sceneGraphs


def load_multiple_sceneGraphs(path):
    all_files = union(glob(os.path.join(path, '*.pkl')),
                      glob(os.path.join(path, '*.json')))
    return reduce(merge_sceneGraphs,
                  [load_sceneGraphs(filename) for filename in all_files])


def default_scene(filename):
    image_id = get_imageId(filename)
    scene = {'image_id': image_id, 'image_filename': filename}
    if 'split' not in scene:
        for split in ['train', 'test', 'val']:
            if split in image_id:
                scene['split'] = split
    return (image_id, scene)
