import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from copy import deepcopy
from dataset.tools.image_transforms import SquarePad
from dataset.tools import sceneGraph_port, image_utils
import h5py as h5
import os
from dataset.tools import protocol
from dataset.toy import teddy_dataset
import sys
args = sys.args
info = sys.info

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
            print('loading sceneGraphs ... ')
            cls.main_sceneGraphs = cls.load_graphs()
            sceneGraph_port.register_vocabulary(cls.main_sceneGraphs)

        self.sceneGraphs = deepcopy(cls.main_sceneGraphs)
        if args.visual_bias and args.train_config and config == 'full':
            self.filter_fn =\
                sceneGraph_port.customize_filterFn(args.train_config,
                                                   val_reverse=True,
                                                   )
            sceneGraph_port.filter_sceneGraphs(
                self.sceneGraphs,
                self.filter_fn,
                inplace=True,
            )
        self.split()


    def __getitem__(self, index):
        if not isinstance(index, str):
            index = self.index[index]

        if args.group == 'gqa':
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

            entry = {
                'index': index,
                'img_size': im_size,
                'img': self.transform_pipeline(image_unpadded),

                'gt_boxes': gt_boxes,
                'gt_classes': gt_classes,
                'gt_attributes': gt_attributes,
                'gt_relations': gt_rels,

                'scale': args.image_scale / args.box_scale,
            }

            return entry

        elif args.group in ['clevr', 'toy']:
            output = {}
            scene = self.sceneGraphs[index]

            if 'objects' in scene:
                obj_inds = sorted(list(scene['objects'].keys()))
                object_classes = ['-'.join([scene['objects'][ind][cat]
                                            for cat in args.classification])
                                for ind in obj_inds]
                object_classes = np.array([info.vocabulary['classes', obj_class]
                                           for obj_class in object_classes])
                output['object_classes'] = object_classes

            if self.mode == 'sceneGraph':
                output.update({'scene_plain': scene})

            elif self.mode == 'encoded_sceneGraph':
                output.update({'scene': self.encode_sceneGraphs(scene)})

            elif self.mode == 'pretrained':
                output.update({'scene': self.get_features(scene),
                               'object_lengths': len(scene['objects'])})

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


    @classmethod
    def load_graphs(cls):
        info.vocabulary = protocol.Protocol(args.allow_output_protocol,
                                            args.vocabulary_file,
                                            gather=True,
                                            use_special_tokens=False)

        if args.group == 'gqa':

            SG_h5 = h5.File(args.sceneGraph_h5, 'r')
            splits = SG_h5['split'][:]
            image_ids = SG_h5['img_ids'][:].astype('U').tolist()
            filenames = [os.path.join(args.image_dir, filename+'.jpg')
                            for filename in image_ids]
            index = np.arange(splits.shape[0])
            split_indexes = {
                k: np.array([i for i in range(splits.shape[0])
                            if splits[i] == v])
                for k, v in {'train': 0, 'val': 1, 'test': 2}.items()
            }

            obj_ranges = SG_h5['obj_ranges'][:]
            rel_ranges = SG_h5['rel_ranges'][:]

            # loading box information
            gt_classes = SG_h5['labels'][:, 0]
            gt_boxes = SG_h5['boxes_{}'.format(args.box_scale)][:].astype(np.float32)  # will index later
            gt_attributes = SG_h5['attributes'][:]
            # convert from xc, yc, w, h to x1, y1, x2, y2
            gt_boxes[:, :2] = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2
            gt_boxes[:, 2:] = gt_boxes[:, :2] + gt_boxes[:, 2:]

            # load relation labels
            gt_relations = SG_h5['relations'][:]

            SG_h5.close()
            sceneGraphs = {
                'splits': splits,
                'image_ids': image_ids,
                'filenames': filenames,
                'split_indexes': split_indexes,
                'obj_ranges': obj_ranges,
                'rel_ranges': rel_ranges,
                'gt_classes': gt_classes,
                'gt_boxes': gt_boxes,
                'gt_attributes': gt_attributes,
                'gt_relations': gt_relations,
            }

        if args.group == 'clevr':
            sceneGraphs = sceneGraph_port.load_multiple_sceneGraphs(args.sceneGraph_dir)

            if args.task.endswith('pt'):
                sceneGraphs = sceneGraph_port.merge_sceneGraphs(
                    sceneGraph_port.load_multiple_sceneGraphs(args.feature_sceneGraph_dir),
                    sceneGraphs,
                )

            if args.task.endswith('dt'):
                all_imageNames = image_utils.get_imageNames(args.image_dir)
                for imageName in all_imageNames:
                    image_id, default_scene = sceneGraph_port.default_scene(imageName)
                    if not image_id in sceneGraphs:
                        sceneGraphs[image_id] = default_scene
                    else:
                        sceneGraphs[image_id].update(default_scene)


        elif args.task == 'toy':
            sceneGraphs = teddy_dataset.ToyDataset.build_visual_dataset()

        else:
            raise Exception('No such task supported: %s' % args.task)

        sceneGraph_port.register_classes(sceneGraphs)
        return sceneGraphs

    def split(self):
        self.split_indexes = {key: [k for k, s in self.sceneGraphs.items()
                                    if s['split'] == key]
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
