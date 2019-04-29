import json, pickle
import os
import h5py as h5
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from copy import deepcopy
from dataset.tools.image_transforms import SquarePad
from argparse import Namespace
from tqdm import tqdm
from dataset.tools import protocol
from glob import glob
from dataset.toy import teddy_dataset
import sys
args = sys.args
info = sys.info

class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='sceneGraphs'):
        self.mode = mode
        self.valid_modes = ['sceneGraphs', 'encoded_sceneGraphs', 'pretrained', 'recognized']

        print('loading sceneGraphs ... ')
        self.load_graphs()

        tform = [
            SquarePad(),
            Resize(args.image_scale),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    def __getitem__(self, index):
        if not isinstance(index, str):
            index = self.index[index]

        if args.task == 'gqa':
            image_unpadded = Image.open(self.filenames[index]).convert('RGB')

            w, h = image_unpadded.size
            img_scale_factor = args.image_scale / max(w, h)
            if h > w:
                im_size = (args.image_scale, int(w * img_scale_factor), img_scale_factor)
            elif h < w:
                im_size = (int(h * img_scale_factor), args.image_scale, img_scale_factor)
            else:
                im_size = (args.image_scale, args.image_scale, img_scale_factor)

            gt_boxes, gt_classes, gt_attributes =\
                [e[self.obj_ranges[index, 0] : self.obj_ranges[index, 1]]
                for e in [self.gt_boxes, self.gt_classes, self.gt_attributes]]
            gt_rels = self.gt_relations\
                [self.rel_ranges[index, 0]  : self.rel_ranges[index, 1]]

            entry = Namespace()
            entry.__dict__.update({
                'index': index,
                'img_size': im_size,
                'img': self.transform_pipeline(image_unpadded),

                'gt_boxes': gt_boxes,
                'gt_classes': gt_classes,
                'gt_attributes': gt_attributes,
                'gt_relations': gt_rels,

                'scale': args.image_scale / args.box_scale,
            })

            if hasattr(self, 'rpn_rois'):
                entry['proposals'] = self.rpn_rois[index]

            self.assertion_checks(entry)
            return entry

        else:
            if self.mode == 'sceneGraphs':
                return self.sceneGraphs[index]
            elif self.mode == 'encoded_sceneGraphs':
                return self.encode_sceneGraphs(self.sceneGraphs[index])
            elif self.mode == 'pretrained':
                return self.get_features(self.sceneGraphs[index])
            elif self.mode == 'recognized':
                raise Exception('not implemented yet')


    def load_graphs(self):
        if args.group in ['gqa', 'clevr']:
            info.vocabulary = protocol.Protocol(args.allow_output_protocol,
                                                args.vocabulary_file,
                                                gather=True,
                                                use_special_tokens=False)

        if args.group == 'gqa':
            pbar = tqdm(total=5, desc='Loading SceneGraphs')

            SG_h5 = h5.File(args.sceneGraph_h5, 'r')
            pbar.update()
            self.splits = SG_h5['split'][:]
            self.image_ids = SG_h5['img_ids'][:].astype('U').tolist()
            self.filenames = [os.path.join(args.image_dir, filename+'.jpg')
                            for filename in self.image_ids]
            self.index = np.arange(self.splits.shape[0])
            self.split_indexes = {
                k: np.array([i for i in range(self.splits.shape[0])
                            if self.splits[i] == v])
                for k, v in {'train': 0, 'val': 1, 'test': 2}.items()
            }

            pbar.update()
            self.obj_ranges = SG_h5['obj_ranges'][:]
            self.rel_ranges = SG_h5['rel_ranges'][:]

            # loading box information
            pbar.update()
            self.gt_classes = SG_h5['labels'][:, 0]
            self.gt_boxes = SG_h5['boxes_{}'.format(args.box_scale)][:].astype(np.float32)  # will index later
            self.gt_attributes = SG_h5['attributes'][:]
            # convert from xc, yc, w, h to x1, y1, x2, y2
            self.gt_boxes[:, :2] = self.gt_boxes[:, :2] - self.gt_boxes[:, 2:] / 2
            self.gt_boxes[:, 2:] = self.gt_boxes[:, :2] + self.gt_boxes[:, 2:]

            # load relation labels
            pbar.update()
            self.gt_relations = SG_h5['relations'][:]

            pbar.update()
            pbar.close()
            SG_h5.close()

        elif args.group == 'clevr':
            self.filenames = glob(os.path.join(args.image_dir, '*/*'))
            self.image_ids = [self.get_imageId(x) for x in self.filenames]
            self.sceneGraphs = {image_id: {'split': image_id.split('_')[1], 'image_id': image_id}
                                for image_id in self.image_ids}

            for key in ['train', 'val', 'test']:
                if args.task.endswith('pt'):
                    sceneGraph_pkl = os.path.join(args.pt_sceneGraph_dir, 'clevr_%s_scenes_parsed_pretrained.pkl' % key)
                    if os.path.exists(sceneGraph_pkl):
                        with open(sceneGraph_pkl, 'rb') as f:
                            loaded = pickle.load(f)['scenes']
                        for scene in loaded:
                            image_id = self.get_imageId(scene['image_filename'])
                            self.sceneGraphs[image_id]['features'] =\
                                [obj['feature'] for obj in scene['objects']]

                sceneGraph_json = os.path.join(args.sceneGraph_dir, 'CLEVR_%s_scenes.json' % key)
                if os.path.exists(sceneGraph_json):
                    with open(sceneGraph_json, 'r') as f:
                        loaded = json.load(f)['scenes']
                    for scene in loaded:
                        image_id = self.get_imageId(scene['image_filename'])
                        for k, v in scene.items():
                            if k != 'objects':
                                self.sceneGraphs[image_id][k] = v
                            else:
                                self.sceneGraphs[image_id]['objects'] =\
                                    {str(i): obj for i, obj in enumerate(scene['objects'])}
                                for obj in scene['objects']:
                                    for cat, attr in obj.items():
                                        if isinstance(attr, str):
                                            info.vocabulary[cat, attr]

        elif args.task == 'toy':
            teddy_dataset.ToyDataset.build_visual_dataset()
            self.sceneGraphs = teddy_dataset.ToyVisualDataset()

        else:
            raise Exception('No such task supported: %s' % args.task)

        self.split_indexes = {key: [s for k, s in self.sceneGraphs.items()
                                    if s['split'] == key]
                              for key in ['train', 'test', 'val']}

    def get_imageId(self, filename):
        return filename.rstrip('.jpg').rstrip('.png').split('/')[-1]

    def to_split(self, split):
        self.split = split
        self.index = self.split_indexes[split]
        return self

    def to_mode(self, mode):
        if mode in self.valid_modes:
            self.mode = mode
        else:
            raise Exception('invalid mode: %s' % mode)

    @classmethod
    def get_datasets(cls):
        base_dataset = cls()
        train, val, test = [deepcopy(base_dataset).to_split(s)
                            for s in ['train', 'val', 'test']]
        return train, val, test

    def __len__(self):
        return min(len(self.index), args.max_sizeDataset)

    def items(self):
        return self.sceneGraphs.items()

    def keys(self):
        return self.sceneGraphs.keys()

    def assertion_checks(self, entry):
        pass

    @classmethod
    def collate(cls, data):
        return data

    def get_features(self, scene):
        return np.array(scene['features'])

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
