import os
import sys
args = sys.args
info = sys.info
import json
import pickle
from glob import glob
from functools import reduce
from utils.common import union, pick_one, get_imageId
from collections import Counter


def load_sceneGraphs(filename):
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            loaded = json.load(f)

    if 'scenes' in loaded:
        loaded = loaded['scenes']
    if isinstance(loaded, dict):
        loaded = list(loaded.values())

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
        sceneGraphs[image_id] = scene

    return sceneGraphs


def register_vocabulary(sceneGraphs):
    for scene in sceneGraphs.values():
        if 'objects' in scene:
            for obj in scene['objects'].values():
                for cat, attr in obj.items():
                    if isinstance(attr, str):
                        info.vocabulary[cat, attr]

def register_classes(sceneGraphs):
    for scene in sceneGraphs.values():
        if 'objects' in scene:
            object_classes = ['-'.join([obj[cat]
                                        for cat in args.classification])
                            for obj in scene['objects'].values()]
            [info.protocol['classes', obj_class]
             for obj_class in object_classes]


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
                scene[k] = scene_x[k] if k in scene_x else scene_y[k]
            sceneGraphs[image_id] = scene
    return sceneGraphs


def load_multiple_sceneGraphs(path):
    all_files = union(glob(os.path.join(path, '*.pkl')),
                      glob(os.path.join(path, '*.json')))
    return reduce(merge_sceneGraphs,
                  [load_sceneGraphs(filename) for filename in all_files])


def default_scene(filename):
    image_id = get_imageId(filename)
    scene = {'image_id': image_id,
             'image_filename': os.path.join(args.root_dir, filename)}
    if 'split' not in scene:
        for split in ['train', 'test', 'val']:
            if split in image_id:
                scene['split'] = split
    return (image_id, scene)

def filter_sceneGraphs(sceneGraphs, filter_fn, inplace=False):

    infeasible_split = [s['split'] for s in sceneGraphs.values()
                        if not filter_fn(s)]
    print('filtered scene graphs: ', Counter(infeasible_split))
    if not inplace:
        return {k: s for k, s in sceneGraphs.items()
                if filter_fn(s)}
    else:
        keys = list(sceneGraphs.keys())
        for k in keys:
            s = sceneGraphs[k]
            if not filter_fn(s):
                sceneGraphs.pop(k)
        return sceneGraphs


'''
config format:
    {<Attr_main>: [Attr_sub0, Attr_sub1], ...}
'''

def customize_filterFn(config, val_reverse=False):

    def output_fn(scene):

        if not 'objects' in scene:
            return False
        val = scene['split'] != 'train'
        reverse = val and val_reverse

        for obj in scene['objects'].values():
            attrs = set([at for at in obj.values()
                             if isinstance(at, str)])
            for main, subs in config.items():
                if main in attrs:
                    if not reverse:
                        feasible_item = False
                        for sub in subs:
                            if sub in attrs:
                                feasible_item = True
                        if not feasible_item:
                            return False
                    else:
                        for sub in subs:
                            if sub in attrs:
                                return False
        return True

    return output_fn
