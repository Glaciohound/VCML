import os
import json
import numpy as np


def invert_dict(d):
    return {v: k for k, v in d.items()}


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)

def get_feat_vec_gqa(obj, args, dicts):
    vec = np.zeros((args.dim_object+args.dim_attribute,), dtype='i1')
    vec[dicts['label_to_idx'][obj['name']] -1] = 1
    for attr in obj['attributes']:
        vec[dicts['attribute_to_idx'][attr] + args.dim_object -1] = 1
    return list(vec)

def get_pred_vec_gqa(objects, ground_truth_objects, dicts, max_num_objects):
    object_index = {obj['object_id']: i for i, obj in enumerate(objects)}

    for i, obj in ground_truth_objects.items():
        if i in object_index:
            for relation in obj['relations']:
                j = relation['object']
                if j in object_index:
                    objects[object_index[i]]['relation_vector'][object_index[j]][dicts['predicate_to_idx'][relation['name']]-1] = 1

def get_attrs_clevr(feat_vec):
    shapes = ['sphere', 'cube', 'cylinder']
    sizes = ['large', 'small']
    materials = ['metal', 'rubber']
    colors = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
    obj = {
        'shape': shapes[np.argmax(feat_vec[0:3])],
        'size': sizes[np.argmax(feat_vec[3:5])],
        'material': materials[np.argmax(feat_vec[5:7])],
        'color': colors[np.argmax(feat_vec[7:15])],
        'position': feat_vec[15:18].tolist(),
    }
    return obj


def load_clevr_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    scenes = []
    for s in scenes_dict:
        objs = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'], s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']
            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            item['mask'] = o['mask']
            objs.append(item)
        scenes.append({
            'objects': objs,
        })
    return scenes


def iou(m1, m2):
    intersect = m1 * m2
    union = 1 - (1 - m1) * (1 - m2)
    return intersect.sum() / union.sum()

def iou_box(b1, b2):
    intersect = (min(b1[2], b2[2])-max(b1[0], b2[0])) * (min(b1[3], b2[3]) - max(b1[1], b2[1]))
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - intersect
    return intersect/union

def iomin(m1, m2):
    if m1.sum() == 0 or m2.sum() == 0:
        return 1.0
    intersect = m1 * m2
    return intersect.sum() / min(m1.sum(), m2.sum())

def iomin_box(b1, b2):
    if b1[0]==b1[2] or b1[1]==b1[3] or b2[0]==b2[2] or b2[1]==b2[3]:
        return 1.0
    intersect = (min(b1[2], b2[2])-max(b1[0], b2[0])) * (min(b1[3], b2[3]) - max(b1[1], b2[1]))
    return intersect / min((b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1]))
