import os
import json
import pickle
from glob import glob
from functools import reduce
from metaconcept import info, args
from metaconcept.utils.common import union, get_imageId
from collections import Counter
from tqdm import tqdm


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
        for image_id, scene in loaded.items():
            scene['image_id'] = image_id
        loaded = list(loaded.values())

    sceneGraphs = {}
    for scene in loaded:
        default_scene(scene, filename)
        if isinstance(scene['objects'], list):
            scene['objects'] =\
                {str(i): obj for i, obj in enumerate(scene['objects'])}
        image_id = scene['image_id']

        if args.group == 'gqa':
            scene_obj = {
                'x': 0, 'y': 0,
                'w': scene['width'], 'h': scene['height'],
            }
            if 'location' in scene:
                scene_obj['name'] = scene['location']
            scene['objects']['scene_{image_id}'] = scene_obj

        sceneGraphs[image_id] = scene

    return sceneGraphs


def register_vocabulary(sceneGraphs):
    concepts = set()
    for scene in tqdm(sceneGraphs.values()):
        if 'objects' in scene:
            for obj in scene['objects'].values():
                for cat, attr in obj.items():
                    if isinstance(attr, str):
                        info.vocabulary[cat, attr]
                        concepts.add(attr)
                    if cat == 'attributes':
                        for at in attr:
                            info.vocabulary[cat, at]
                            concepts.add(at)
    for _concept in concepts:
        info.protocol['concepts', _concept]


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


def default_scene(scene, sceneGraph_file=''):
    if 'image_id' not in scene:
        image_id = get_imageId(scene['image_filename'])
        scene['image_id'] = image_id
    else:
        image_id = scene['image_id']

    if 'split' not in scene:
        for split in ['train', 'test', 'val']:
            if split in image_id or split in sceneGraph_file:
                scene['split'] = split


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
