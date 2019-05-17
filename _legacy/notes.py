import os
from copy import deepcopy
from PIL import Image

import h5py as h5
import numpy as np
import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from metaconcept import info, args
from metaconcept.dataset.tools.image_transforms import SquarePad
from metaconcept.dataset.tools import sceneGraph_port, image_utils
from metaconcept.dataset.tools import protocol
from metaconcept.dataset.toy import teddy_dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        cls = Dataset
        self.mode = 'sceneGraph'
        self.valid_modes = ['sceneGraph', 'encoded_sceneGraph', 'pretrained', 'detected']
        tform = [
            SquarePad(),
            Resize(args.image_scale),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

        if not hasattr(cls, 'main_sceneGraphs'):
            print('Loading sceneGraphs ... ', end='', flush=True)
            cls.main_sceneGraphs = cls.load_graphs()
            print('DONE')
            print('Registering vocabulary ... ', end='', flush=True)
            sceneGraph_port.register_vocabulary(cls.main_sceneGraphs)
            print('DONE')

        self.sceneGraphs = deepcopy(cls.main_sceneGraphs)
        if args.visual_bias and config == 'full':
            print('Filtering sceneGraphs ...', end='', flush=True)
            self.filter_fn = \
                sceneGraph_port.customize_filterFn(args.visual_bias,
                                                   val_reverse=True,
                                                   )
            sceneGraph_port.filter_sceneGraphs(
                self.sceneGraphs,
                self.filter_fn,
                inplace=True,
            )
            print('DONE')

        self.split()

    def __getitem__(self, index):
        try:
            return self.__getitem_inner__(index)
        except Exception as exc:
            print(exc)
            raise exc

    def __getitem_inner__(self, index):
        if not isinstance(index, str):
            index = self.index[index]

        output = {}
        scene = self.sceneGraphs[index]

        if 'objects' in scene:
            output['object_lengths'], output['object_classes']\
                = self.analyze_objects(scene, output)

        if self.mode == 'sceneGraph':
            output.update({'scene_plain': scene})

        elif self.mode == 'encoded_sceneGraph':
            output.update({'scene': self.encode_sceneGraphs(scene)})

        elif self.mode == 'pretrained':
            output.update({'scene': self.get_features(scene)})

        elif self.mode == 'detected':
            image = Image.open(scene['image_filename']).convert('RGB')
            shape = image.size
            image_transformed = self.transform_pipeline(image).numpy()
            bboxes = image_utils.annotate_objects(
                scene, shape,
                (args.image_scale, args.image_scale))['objects']
            output.update({'image': image_transformed,
                            'objects': bboxes,
                            'object_lengths': bboxes.shape[0]})

        return output

    def analyze_objects(self, scene, output):

        object_lengths = len(scene['objects'])

        all_concepts = info.vocabulary.concepts
        object_classes = np.zeros((len(scene['objects']), len(all_concepts)),
                                    dtype=float)

        for i, obj in enumerate(scene['objects'].values()):
            for cat, attr in obj.items():
                if isinstance(attr, str) and attr in all_concepts:
                    object_classes[i, all_concepts.index(attr)] = 1

                if cat == 'attributes':
                    for at in attr:
                        object_classes[i, all_concepts.index(at)] = 1

        return object_lengths, object_classes

    @classmethod
    def load_graphs(cls):
        info.vocabulary = protocol.Protocol(args.allow_output_protocol,
                                            '', # args.vocabulary_file,
                                            gather=True,
                                            use_special_tokens=False)

        if args.group == 'gqa':
            sceneGraphs = sceneGraph_port.load_multiple_sceneGraphs(args.sceneGraph_dir)

        elif args.group == 'clevr':
            sceneGraphs = sceneGraph_port.load_multiple_sceneGraphs(args.sceneGraph_dir)

            if args.task.endswith('pt'):
                sceneGraphs = sceneGraph_port.merge_sceneGraphs(
                    sceneGraph_port.load_multiple_sceneGraphs(args.feature_sceneGraph_dir),
                    sceneGraphs,
                )

        elif args.task == 'toy':
            sceneGraphs = teddy_dataset.ToyDataset.build_visual_dataset()

        else:
            raise Exception('No such task supported: %s' % args.task)

        if args.group == 'gqa' or args.task.endswith('dt'):
            all_imageNames = image_utils.get_imageNames(args.image_dir)
            for imageName in all_imageNames:
                default_scene = {'image_filename': imageName}
                sceneGraph_port.default_scene(default_scene)
                image_id = default_scene['image_id']
                if not image_id in sceneGraphs:
                    sceneGraphs[image_id] = default_scene
                else:
                    sceneGraphs[image_id].update(default_scene)

        return sceneGraphs

    def split(self):
        self.split_indexes = {key: [k for k, s in self.sceneGraphs.items()
                                    if s.get('split', 'train') == key]
                              for key in ['train', 'test', 'val']}
        self.index = list(self.sceneGraphs.keys())

    def to_split(self, split):
        self.split = split
        self.index = self.split_indexes[split]
        return self

    def to_mode(self, mode):
        if mode in self.valid_modes:
            self.mode = mode
        else:
            raise Exception('invalid mode: %s' % mode)
        if mode in ['detected']:
            for image_id, scene in self.sceneGraphs.items():
                image_utils.match_objects(scene, inplace=True)

    def __len__(self):
        return min(len(self.index), args.max_sizeDataset)

    def items(self):
        return self.sceneGraphs.items()

    def keys(self):
        return self.sceneGraphs.keys()

    @classmethod
    def collate(cls, datas):
        return datas

    def get_features(self, scene):
        return scene['features']

    def encode_sceneGraphs(self, scene):
        features = []
        for obj in scene['objects'].values():
            features.append([
                info.vocabulary[at] for cat, at in obj.items()
                if isinstance(at, str)
            ])
        dim_features = max([len(x) for x in features])
        for x in features:
            x += [-1 for i in range(dim_features - len(x))]
        return np.array(features)

