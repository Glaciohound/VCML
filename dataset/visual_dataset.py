import json
import os
import h5py as h5
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from copy import deepcopy
from dataset.image_transforms import SquarePad
from argparse import Namespace
from tqdm import tqdm
from dataset.tools.question_utils import build_tokenMap


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, info=None):
        self.info = info
        self.args = args

        pbar = tqdm(total=2, desc='Creating Dataset')
        self.load_graphs()
        pbar.update()
        self.load_vocabulary()
        self.info.vocabulary = self.vocabulary
        pbar.update()
        pbar.close()

        tform = [
            SquarePad(),
            Resize(args.image_scale),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    def __getitem__(self, index_):
        args = self.args
        if isinstance(index_, str):
            index_ = self.image_ids.index(index_)
        index = self.index[index_]
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

    def load_graphs(self):
        args = self.args
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

        if args.max_nImages > -1:
            self.valid_num = args.max_nImages

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

    def load_vocabulary(self):
        args = self.args
        with open(args.vocabulary_file, 'r') as f:
            self.vocabulary = json.load(f)

        build_tokenMap(self, self.vocabulary)

    def to_split(self, split):
        self.split = split
        self.index = self.split_indexes[split]
        return self

    @classmethod
    def get_datasets(cls, args):
        base_dataset = cls(args)
        train, val, test = [deepcopy(base_dataset).to_split(s)
                            for s in ['train', 'val', 'test']]
        return train, val, test

    def __len__(self):
        return len(self.index)

    def assertion_checks(self, entry):
        pass

    @classmethod
    def collate(cls, data):
        return data
