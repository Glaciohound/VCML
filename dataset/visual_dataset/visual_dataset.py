#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visual_dataset.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This class is a dataset for visual images and sceneGraph information


import os
import numpy as np
import torch
from torchvision.transforms import Resize, Compose
from PIL import Image
from tqdm import tqdm
import h5py as h5

from .utils.image_transforms import SquarePad
from .utils import image_utils
from .agent_visual_dataset import AgentDataset
from .utils.sceneGraph_loader import match_images


class Dataset(torch.utils.data.Dataset):
    """
    Base visual dataset.
    Default setting: inputs=['plain'], obj_prior=True,
    """
    valid_inputs = [
        'plain',
        'detection',
        'image',
        'ground-truth',
    ]

    def __init__(self, args, logger, sceneGraphs, index, image_dir=''):
        self.args = args
        self.logger = logger
        self.transform_pipeline = get_transform_pipeline(args.image_scale)
        self.load_sceneGraphs(sceneGraphs)
        self.index = index
        self.image_dir = image_dir

        self.set_inputs(['plain'])
        self.set_obj_prior(True)

    def match_images(self, image_dir):
        match_images(self.sceneGraphs, image_dir, self.logger)

    def __getitem__(self, index):
        if index == self.args.null_image:
            scene = self.default_scene
        else:
            scene = self.sceneGraphs[index]
        filename = scene['image_filename']

        output = {'image_id': index}

        if 'plain' in self.inputs:
            output['plain'] = scene

        if 'ground-truth' in self.inputs:
            output['object_length'] = len(scene['objects'])

        if 'image' in self.inputs:
            if filename == self.args.null_image+'.jpg':
                scale = self.args.image_scale
                image_transformed = np.zeros((3, scale, scale))
                ori_shape = (scale, scale)
            else:
                image_transformed, ori_shape = self.read_image(filename)
            output['image'] = image_transformed

            if 'detection' in self.inputs:
                if filename == self.args.null_image+'.jpg':
                    bboxes = np.zeros([0, 4])
                else:
                    bboxes = image_utils.annotate_objects(
                        scene, ori_shape,
                        (self.args.image_scale, self.args.image_scale),
                        self.obj_prior
                    )
                output.update({'objects': bboxes,
                               'object_length': bboxes.shape[0],
                               })

        return output

    def init_cache(self, cache_dir, main, logger):
        self.cache_filename = os.path.join(
            cache_dir, f'{self.args.task}_processed_images_{self.index}.h5'
        )

        if not os.path.exists(self.cache_filename):
            self.image_ids = list(set([
                scene['image_id'] for scene in
                self.sceneGraphs.values()]))
            num = len(self.image_ids)
            scale = self.args.image_scale

            with h5.File(self.cache_filename, 'w') as cache_file:
                cache_file.create_dataset(
                    'images',
                    shape=(len(self.sceneGraphs), 3, scale, scale),
                    dtype=float,
                )
                cache_file.create_dataset(
                    'image_ids',
                    data=np.array(self.image_ids, dtype='S'),
                )
                cache_file.create_dataset(
                    'image_sizes',
                    data=np.zeros(shape=(num, 2), dtype=int)
                )
                cache_file.create_dataset(
                    'processed',
                    data=np.zeros(num, dtype=int)
                )

        self.cache_file = h5.File(
            self.cache_filename, 'r+')
        self.image_ids = self.cache_file['image_ids'][:].astype('U')
        self.cache_image_index = dict(zip(
            self.image_ids,
            range(len(self.image_ids))
        ))

    def read_cache(self, image_id):
        index = self.cache_image_index[image_id]
        processed = self.cache_file['processed'][index]
        if processed == 1:
            image = self.cache_file['images'][index]
            size = self.cache_file['image_sizes'][index]
        else:
            image = None
            size = None
        return image, size, processed

    def save_cache(self, image_id, image, size):
        index = self.cache_image_index[image_id]
        self.cache_file['processed'][index] = 1
        self.cache_file['images'][index] = image
        self.cache_file['image_sizes'][index] = size

    def read_image(self, filename):
        if not os.path.exists(filename):
            filename = os.path.join(self.image_dir, filename)
        image = Image.open(filename).convert('RGB')
        shape = image.size
        image = np.array(self.transform_pipeline(image)).transpose(2, 0, 1)

        # normalizing
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(3, 1, 1)
        image = (image / 255.0 - mean) * std

        return image, shape

    def load_sceneGraphs(self, sceneGraphs):
        '''
        if isinstance(sceneGraphs, list):
            sceneGraphs = {id_from_fileName(scene['image_filename']):
                           scene for scene in sceneGraphs}
        '''
        self.sceneGraphs = sceneGraphs

    def register_concepts(self, concepts):
        for scene in tqdm(self.sceneGraphs.values(), leave=False):
            for obj in scene['objects'].values():
                for attr in obj['concepts_contained']:
                    concepts.register(attr)

    def keys(self):
        return self.sceneGraphs.keys()

    def get_agent(self):
        return AgentDataset(self.args, self.logger, self)

    def set_obj_prior(self, value=True):
        self.obj_prior = value

    def copy(self):
        return Dataset(self, self.args, self.logger, self.sceneGraphs,
                       self.index)

    def __len__(self):
        return len(self.sceneGraphs)

    def set_inputs(self, inputs):
        assert set(inputs).issubset(self.valid_inputs),\
            'required inputs should all be valid'
        self.inputs = inputs

    @property
    def indexes(self):
        return list(self.sceneGraphs)

    @property
    def default_scene(self):
        null_image = self.args.null_image
        return {
            'image_id': null_image,
            'image_filename': f'{null_image}.jpg',
            'concepts_contained': [],
            'objects': {},
            'height': 0,
            'width': 0,
        }

    def mark_splits(self, split_indexes):
        for split, indexes in split_indexes.items():
            for ind in indexes:
                self.sceneGraphs[ind]['split'] = split


def get_transform_pipeline(image_scale):
    tform = [
        SquarePad(),
        Resize(image_scale),
    ]
    transform_pipeline = Compose(tform)
    return transform_pipeline
