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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, command):
        self.args = args
        self.command = command

        self.load_graphs()
        self.load_vocabulary()

        tform = [
            SquarePad(),
            Resize(args.image_scale),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    def __getitem__(self, index_):
        if isinstance(index_, str):
            index_ = self.image_ids.index(index_)
        args = self.args
        index = self.split_index[index_]
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
            [e[self.box_ranges[index, 0] : self.box_ranges[index, 1]]
             for e in [self.gt_boxes, self.gt_classes, self.gt_attributes]]
        gt_rels = self.gt_relations\
            [self.relation_ranges[index, 0]  : self.relation_ranges[index, 1]]

        entry = Namespace()
        entry.__dict__.update({
            'index': index,
            'img_size': im_size,
            'img': self.transform_pipeline(image_unpadded),

            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'gt_attributes': gt_attributes,
            'gt_relations': gt_rels,

            'scale': args.image_scale / args.box_scale,  # Multiply the boxes by this.
        })

        if hasattr(self, 'rpn_rois'):
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        return entry

    def load_graphs(self):
        args = self.args
        pbar = tqdm(total=5, desc='Loading SceneGraphs', postfix='file')

        SG_h5 = h5.File(args.sceneGraph_h5, 'r')
        pbar.update(); pbar.set_postfix_str('infos')
        self.splits = SG_h5['split'][:]
        self.image_ids = SG_h5['img_ids'][:].astype('U').tolist()
        self.filenames = [os.path.join(args.image_dir, filename+'.jpg')
                          for filename in self.image_ids]
        self.split_indexes = {
            k: np.array([i for i in range(self.splits.shape[0]) if self.splits[i] == v])
            for k, v in {'train': 0, 'val': 1, 'test': 2}.items()
        }

        if args.max_nImages > -1:
            self.valid_num = args.max_nImages

        pbar.update(); pbar.set_postfix_str('boxes')
        self.box_ranges = SG_h5['box_ranges'][:]
        self.relation_ranges = SG_h5['relation_ranges'][:]

        # loading box information
        pbar.update(); pbar.set_postfix_str('objects')
        self.gt_classes = SG_h5['labels'][:, 0]
        self.gt_boxes = SG_h5['boxes_{}'.format(args.box_scale)][:].astype(np.float32)  # will index later
        self.gt_attributes = SG_h5['attributes'][:]
        assert np.all(self.gt_boxes[:, :2] >= 0)  # sanity check
        assert np.all(self.gt_boxes[:, 2:] > 0)  # no empty box
        # convert from xc, yc, w, h to x1, y1, x2, y2
        self.gt_boxes[:, :2] = self.gt_boxes[:, :2] - self.gt_boxes[:, 2:] / 2
        self.gt_boxes[:, 2:] = self.gt_boxes[:, :2] + self.gt_boxes[:, 2:]

        # load relation labels
        pbar.update(); pbar.set_postfix_str('relations')
        self.gt_relations = SG_h5['relations'][:]

        pbar.update()
        pbar.close()
        SG_h5.close()

    def load_vocabulary(self):
        args = self.args
        with open(args.vocabulary_file, 'r') as f:
            self.vocabulary = json.load(f)
        self.vocabulary['label_to_idx']['__background__'] = 0
        self.vocabulary['predicate_to_idx']['__background__'] = 0

        ((self.label_to_idx, self.idx_to_labels),
         (self.attribute_to_idx, self.idx_to_attributes),
         (self.predicate_to_idx, self.idx_to_predicates)) =\
            [(x, sorted(x, key=lambda k: x[k]))
             for x in [self.vocabulary[c]
                       for c in ('label_to_idx', 'attribute_to_idx', 'predicate_to_idx')]]

    def to_mode(self, mode):
        self.mode = mode
        self.split_index = self.split_indexes[mode]
        return self

    @classmethod
    def get_datasets(cls, args):
        base_dataset = cls(args, command=None)
        train, val, test = [deepcopy(base_dataset).to_mode(m)
                            for m in ['train', 'val', 'test']]
        return train, val, test

    def __len__(self):
        return len(self.split_index)

def assertion_checks(entry):
    pass

def cn_collate(data):
    return data

class DataLoader(torch.utils.data.DataLoader):
    @classmethod
    def get_dataloaders(cls, args):
        train_dataset, val_dataset, test_dataset = Dataset.get_datasets(args)
        train_loader = cls(
            dataset=train_dataset,
            batch_size=args.train_batch_size * args.num_gpus,
            shuffle=args.train_shuffle,
            num_workers=args.num_workers,
            collate_fn=cn_collate,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = cls(
            dataset=val_dataset,
            batch_size=args.val_batch_size * args.num_gpus,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=cn_collate,
            drop_last=True,
            pin_memory=True,
        )
        test_loader = cls(
            dataset=test_dataset,
            batch_size=args.val_batch_size * args.num_gpus,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=cn_collate,
            drop_last=True,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader
