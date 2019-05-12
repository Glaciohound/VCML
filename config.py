from argparse import ArgumentParser
import torch
import os
import shutil
import pprint
import sys
import numpy as np

class Info():
    def __init__(self):
        setattr(sys, 'info', self)
        self.new_torch = torch.__version__.startswith('1')
        args = sys.args
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')\
            if self.new_torch else\
            'cuda' if args.use_cuda else 'cpu'
        self.to = lambda x: (x.to(self.device) if self.new_torch else
                             x.cuda() if self.device == 'cuda' else
                             x.cpu())

class Config:
    conceptual_tokens = ['synonym', 'antonym', 'isinstance']
    def __init__(self):
        setattr(sys, 'args', self)
        args = self.parse_args()
        self.__dict__.update(vars(args))
        self.post_process()

    def parse_args(self):
        parser = ArgumentParser()
        self.dir_args = {}
        group = 'gqa'
        def dir_add_argument(arg_name, **kwargs):
            if not group in self.dir_args:
                self.dir_args[group] = []
            arg_name = '--' + group + '_' + arg_name[2:]
            self.dir_args[group].append(arg_name.strip('-'))
            parser.add_argument(arg_name, **kwargs)

        parser.add_argument('--task', default='toy',
                            choices=['gqa', 'toy', 'clevr_pt', 'clevr_dt'])
        parser.add_argument('--model', default='h_embedding_add2',
                            choices=['relation_model',
                                     'u_embedding',
                                     'h_embedding_mul',
                                     'h_embedding_add',
                                     'h_embedding_add2'])
        parser.add_argument('--similarity', default='cosine',
                            choices=['cosine', 'square'])

        group = 'gqa'
        parser.add_argument('--gqa_data_dir', default='../../data/gqa')
        dir_add_argument('--image_dir', default='raw/allImages/images')
        dir_add_argument('--sceneGraph_h5', default='processed/SG.h5')
        dir_add_argument('--sceneGraph_json', default='raw/sceneGraphs/all_sceneGraphs.json')
        dir_add_argument('--vocabulary_file', default='processed/gqa_vocabulary.json')
        dir_add_argument('--protocol_file', default='processed/gqa_protocol.json')
        dir_add_argument('--questions_h5', default='processed/questions')
        dir_add_argument('--questions_json', default='raw/questions/all_balanced_questions.json')
        group = 'clevr'
        parser.add_argument('--clevr_data_dir', default='../../data/clevr')
        dir_add_argument('--image_dir', default='raw/CLEVR_v1.0/images')
        #dir_add_argument('--sceneGraph_dir', default='raw/CLEVR_v1.0/scenes')
        dir_add_argument('--sceneGraph_dir', default='detections')
        dir_add_argument('--feature_sceneGraph_dir', default='attr_net/results')
        dir_add_argument('--protocol_file', default='processed/clevr_protocol.json')
        dir_add_argument('--vocabulary_file', default='processed/clevr_vocabulary.json')
        group = 'toy'
        parser.add_argument('--toy_data_dir', default='../../data/gqa')
        dir_add_argument('--protocol_file', default='processed/toy_protocol.json')
        dir_add_argument('--vocabulary_file', default='processed/toy_vocabulary.json')

        parser.add_argument('--allow_output_protocol', action='store_true')
        parser.add_argument('--toy_objects', type=int, default=3)
        parser.add_argument('--toy_attributes', type=int, default=16)
        parser.add_argument('--toy_attributesPobject', type=int,
                            default=4)
        parser.add_argument('--toy_categories', type=int, default=4)

        parser.add_argument('--subtask', default='exist',
                            choices=['exist', 'filter', 'query',
                                     'exist_synonym', 'filter_synonym',
                                     'query_antonym', 'query_isinstance',
                                     'query_isinstance_rev',
                                     'visual_bias',
                                     'classification',
                                     'filter_isinstance'])
        parser.add_argument('--no_aid', action='store_true')
        parser.add_argument('--classification', nargs='+', required=False,
                            choices=['color', 'shape', 'material', 'size'])
        parser.add_argument('--questionsPimage', type=int, default=1)
        parser.add_argument('--train_config', nargs='+', required=False,
                            help='in the form of \'Attr_0:Attr_1,Attr2 ...\'')
        parser.add_argument('--incremental_training', nargs='+', required=False,
                            choices=['full', 'partial', 'replaced'],
                            default=['full'])
        parser.add_argument('--val_concepts', nargs='+', required=False)

        parser.add_argument('--max_sizeDataset', type=int, default=5000)
        parser.add_argument('--box_scale', type=int, default=1024)
        parser.add_argument('--image_scale', type=int, default=256)
        parser.add_argument('--ipython', action='store_true')

        parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--epochs', type=int, default=50, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
        parser.add_argument('--init_variance', type=float, default=0.01)
        parser.add_argument('--num_workers', default=1)
        parser.add_argument('--no_train_shuffle', action='store_false')
        parser.add_argument('--perfect_th', type=float, default=0.99)
        parser.add_argument('--visualize_dir', type=str, nargs='*',
                            default='../../data/visualize')
        parser.add_argument('--log_dir', type=str,
                            default='../../data/log')
        parser.add_argument('--ckpt_dir', type=str,
                            default='../../data/gqa/checkpoints')
        parser.add_argument('--visualize_time', type=int, default=500)

        parser.add_argument('--true_th', type=float, default=0.9)
        parser.add_argument('--temperature_init', type=float, default=2)
        parser.add_argument('--non_bool_weight', type=float, default=0.1)
        parser.add_argument('--penalty', type=float, default=0)

        parser.add_argument('--no_validation', action='store_true')
        parser.add_argument('--no_random', action='store_true')
        parser.add_argument('--ckpt', type=str)
        parser.add_argument('--name', type=str, default='trial')

        parser.add_argument('--max_relations', type=int, default=100)
        parser.add_argument('--max_concepts', type=int, default=50)
        parser.add_argument('--num_attributes', type=int, default=3000)
        parser.add_argument('--size', type=int, default=4)
        parser.add_argument('--num_action', type=int, default=4)
        parser.add_argument('--size_dataset', type=int, default=5000)

        parser.add_argument('--question_filter', default='None',
                            choices=['None', 'existance'])

        parser.add_argument('--embed_dim', type=int, default=60)
        parser.add_argument('--identity_dim', type=int, default=50)
        parser.add_argument('--hidden_dim1', type=int, default=0)
        parser.add_argument('--hidden_dim2', type=int, default=0)
        parser.add_argument('--attention_dim', type=int, default=5)
        parser.add_argument('--operation_dim', type=int, default=3)
        parser.add_argument('--feature_dim', type=int, default=512)
        parser.add_argument('--size_attention', type=int, default=30)
        parser.add_argument('--identity_only', action='store_true')

        parser.add_argument('--generalization_ratio', type=float, default=0.25)
        parser.add_argument('--conceptual_question_ratio', type=float, default=0.2)

        return parser.parse_args()

    def post_process(self):
        dicts = self.__dict__
        self.root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

        if self.visualize_dir:
            self.visualize_dir = os.path.join(self.visualize_dir, self.model, self.name)
            if os.path.exists(self.visualize_dir):
                shutil.rmtree(self.visualize_dir)
            os.makedirs(self.visualize_dir)

        self.log_dir = os.path.join(self.log_dir, self.model)
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.model)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.num_gpus = torch.cuda.device_count()
        self.use_cuda = self.num_gpus > 0
        self.toy_categories = min(self.toy_categories, self.toy_attributes)
        self.toy_attributesPobject = min(self.toy_attributesPobject, self.toy_categories)
        self.group = self.task.split('_')[0]
        for group, arguments in self.dir_args.items():
            for arg in arguments:
                dicts[arg] = os.path.join(dicts[group+'_data_dir'],
                                          dicts[arg])
                if group == self.group:
                    dicts[arg.replace(group+'_', '')] = dicts[arg]

        if self.task.endswith('dt'):
            self.feature_dim = 256

        self.conceptual = False
        for k in self.conceptual_tokens:
            if k in self.subtask:
                self.conceptual = True
        if self.subtask == 'visual_bias':
            self.conceptual = True

        if self.no_validation:
            self.generalization_ratio = 0
        self.task_concepts = {}

        if self.train_config:
            if ':' in self.train_config[0]:
                train_config = {}
                for item in self.train_config:
                    main, attrs = item.split(':')
                    attrs = attrs.split(',')
                    train_config[main] = attrs
            self.train_config = train_config

    def print(self):
        pprint.pprint('Arguments: ------------------------')
        pprint.pprint(self.__dict__)
        pprint.pprint('-----------------------------------')
