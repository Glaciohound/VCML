import json
import math
from tqdm import tqdm
from config import Config
from math import floor
import h5py as h5
import numpy as np

def encode_box(region, org_h, org_w, img_long_size):
    x = region['x']
    y = region['y']
    w = region['w']
    h = region['h']
    scale = float(img_long_size) / max(org_h, org_w)
    image_size = img_long_size
    x, y = math.floor(scale*(region['x']-1)), math.floor(scale*(region['y']-1))
    w, h = math.ceil(scale*region['w']), math.ceil(scale*region['h'])

    # clamp to image
    if x < 0: x = 0
    if y < 0: y = 0

    # box should be at least 2 by 2
    if x > image_size - 2:
        x = image_size - 2
    if y > image_size - 2:
        y = image_size - 2
    if x + w >= image_size:
        w = image_size - x
    if y + h >= image_size:
        h = image_size - y

    # also convert to center-coord oriented
    box = np.asarray([x+floor(w/2), y+floor(h/2), w, h], dtype=np.int32)
    return box

def encode_objects(graph_data, vocabulary, img_long_sizes):
    max_nNames = max([len(obj['names']) for scene in graph_data for obj in scene['objects'].values()])
    max_nAttrs = max([len(obj['attributes']) for scene in graph_data for obj in scene['objects'].values()])
    n_objs = sum([len(scene['objects']) for scene in graph_data])

    encoded_labels = -np.ones((n_objs, max_nNames), dtype=np.int32)
    encoded_boxes  = {size: [] for size in img_long_sizes}
    encoded_attributes = -np.ones((n_objs, max_nAttrs), dtype=np.int32)

    obj_ranges = np.zeros((len(graph_data), 2), dtype=np.int32)

    obj_counter = 0
    pbar = tqdm(total=len(graph_data), desc='encoding objects')
    for i, scene in enumerate(graph_data):
        pbar.update()
        obj_ranges[i, 0] = obj_counter
        scene['id_to_idx'] = {}  # object id to region idx

        for obj in scene['objects'].values():
            for size in img_long_sizes:
                encoded_boxes[size].append(encode_box(obj, scene['height'], scene['width'], size))

            labels = np.array([vocabulary['names'].index(n) for n in obj['names']])
            attributes = np.array([vocabulary['attributes'].index(a) for a in obj['attributes']])
            encoded_labels[obj_counter, :labels.shape[0]] = labels
            encoded_attributes[obj_counter, :attributes.shape[0]] = attributes

            obj_counter += 1
        obj_ranges[i, 1] = obj_counter
    pbar.close()

    for k, boxes in encoded_boxes.items():
        encoded_boxes[k] = np.vstack(boxes)
    return encoded_labels, encoded_attributes, encoded_boxes, obj_ranges

def encode_relations(graph_data, vocabulary):
    encoded_rels = []  # encoded relationship tuple
    rel_ranges = np.zeros((len(graph_data), 2), dtype=np.int32)
    rel_idx_counter = 0

    pbar = tqdm(total=len(graph_data), desc='encoding relations')
    for i, scene in enumerate(graph_data):
        pbar.update()
        rel_ranges[i, 0] = rel_idx_counter
        objects = scene['objects']

        for relation in scene['relations']:
            subj = relation['subject']
            obj = relation['object']
            predicate = relation['predicate']
            encoded_rels.append((objects[subj]['idx'], objects[obj]['idx'],
                                vocabulary['relations'].index(predicate)))
            rel_idx_counter += 1

        rel_ranges[i, 1] = rel_idx_counter
    pbar.close()

    return np.array(encoded_rels), rel_ranges


def load_sceneGraph(input_file):
    obj_set, pred_set, attr_set = set(), set(), set()
    graph_data = []

    for idx, graph in input_file.items():
        graph.update({'image_id': idx, 'relations': []})

        for idx, (obj_id, obj) in enumerate(graph['objects'].items()):
            obj['idx'] = idx
            if 'name' in obj:
                obj['names'] = [obj['name']]

            obj_set.add(obj['name'])
            for attr in obj['attributes']:
                attr_set.add(attr)

        for obj_id, obj in graph['objects'].items():
            for relation in obj['relations']:
                pred_set.add(relation['name'])
                graph['relations'].append({'predicate': relation['name'],
                                           'object': relation['object'],
                                           'subject': obj_id})
        graph_data.append(graph)
    return graph_data

def main(args):
    pbar = tqdm(total=12, desc='creating SG_h5', postfix='reading file')

    # reading sceneGraphs and vocabulary
    with open(args.sceneGraph_json, 'r') as f:
        sceneGraphs = json.load(f)
    pbar.update(); pbar.set_postfix_str('loading graphs')
    graph_data = load_sceneGraph(sceneGraphs)
    with open(args.vocabulary_file, 'r') as f:
        vocabulary = json.load(f)

    img_long_sizes = [512, 1024]

    # encode object
    pbar.update(); pbar.set_postfix_str('encoding objects')
    encoded_labels, encoded_attributes, encoded_boxes, obj_ranges=\
        encode_objects(graph_data, vocabulary, img_long_sizes)

    # encode relations
    pbar.update(); pbar.set_postfix_str('encoding relations')
    encoded_rels, rel_ranges = encode_relations(graph_data, vocabulary)

    # write the h5 file
    pbar.update(); pbar.set_postfix_str('creating h5file')
    f = h5.File(args.sceneGraph_h5, 'w')

    f.create_dataset('labels', data=encoded_labels)
    pbar.update()
    f.create_dataset('attributes', data=encoded_attributes)
    pbar.update()
    f.create_dataset('relations', data=encoded_rels)
    pbar.update()
    f.create_dataset('obj_ranges', data=obj_ranges)
    pbar.update()
    f.create_dataset('rel_ranges', data=rel_ranges)
    pbar.update()
    for k, boxes in encoded_boxes.items():
        f.create_dataset('boxes_%i' % k, data=boxes)

    split_encode = {'train': 0, 'val': 1, 'test': 2}
    split = np.array([split_encode[scene['split']] for scene in graph_data])
    f.create_dataset('split', data=split) # 0: train, 1: val, 2: test
    pbar.update()
    img_ids = np.array(list(sceneGraphs.keys()), dtype='S')
    f.create_dataset('img_ids', data=img_ids)
    pbar.update()
    f.close()
    pbar.update()
    pbar.close()

if __name__ == '__main__':
    args = Config()
    main(args)
