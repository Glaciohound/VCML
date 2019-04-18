from argparse import ArgumentParser
import torch
import os
import shutil
import pprint

class Config:
    conceptual_tokens = {'synonym': '_syn', 'antonym': '_anto'}

    def __init__(self, info):
        self.info = info
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
                            choices=['gqa', 'toy', 'clevr_pt', 'clevr_rc'])
        parser.add_argument('--model', default='relation_model',
                            choices=['relation_model', 'u_embedding', 'h_embedding'])
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
        dir_add_argument('--sceneGraph_dir', default='raw/CLEVR_v1.0/scenes')
        dir_add_argument('--pt_sceneGraph_dir', default='attr_net/results')
        dir_add_argument('--protocol_file', default='processed/clevr_protocol.json')
        dir_add_argument('--vocabulary_file', default='processed/clevr_vocabulary.json')
        group = 'toy'
        parser.add_argument('--toy_data_dir', default='../../data/gqa')
        dir_add_argument('--protocol_file', default='processed/toy_protocol.json')

        parser.add_argument('--allow_output_protocol', action='store_true')
        parser.add_argument('--toy_objects', type=int, default=3)
        parser.add_argument('--toy_attributes', type=int, default=16)
        parser.add_argument('--toy_attributesPobject', type=int,
                            default=4)
        parser.add_argument('--toy_categories', type=int, default=4)

        parser.add_argument('--subtask', default='exist',
                            choices=['exist', 'filter', 'query',
                                     'exist_synonym', 'filter_synonym'])
        parser.add_argument('--questionsPimage', type=int, default=1)

        parser.add_argument('--max_sizeDataset', type=int, default=20000)
        parser.add_argument('--box_scale', type=int, default=1024)
        parser.add_argument('--image_scale', type=int, default=592)

        parser.add_argument('--num_workers', default=1)
        parser.add_argument('--no_train_shuffle', action='store_false')
        parser.add_argument('--mode', default='concept-net',
                            choices=['concept-net'])
        parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--epochs', type=int, default=30, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
        parser.add_argument('--grad_clip', type=float, default=1)
        parser.add_argument('--init_variance', type=float, default=0.01)
        parser.add_argument('--loss', type=str, default='mse',
                            choices=['mse', 'weighted', 'first', 'last'])
        parser.add_argument('--curriculum_learning', action='store_true')
        parser.add_argument('--perfect_th', type=float, default=0.05)
        parser.add_argument('--visualize_dir', type=str,
                            default='../../data/visualize')
        parser.add_argument('--ckpt_dir', type=str,
                            default='../../data/gqa/checkpoints')
        parser.add_argument('--visualize_time', type=int, default=500)
        parser.add_argument('--no_validation', action='store_true')
        parser.add_argument('--ipython', action='store_true')

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

        parser.add_argument('--embed_dim', type=int, default=50)
        parser.add_argument('--rank', type=int, default=0)
        parser.add_argument('--identity_dim', type=int, default=50)
        parser.add_argument('--hidden_dim', type=int, default=100)
        parser.add_argument('--attention_dim', type=int, default=5)
        parser.add_argument('--operation_dim', type=int, default=3)
        parser.add_argument('--feature_dim', type=int, default=512)
        parser.add_argument('--size_attention', type=int, default=30)
        parser.add_argument('--identity_only', action='store_true')

        parser.add_argument('--isinstance_mode', type=str, default='color_1',
                            choices=['color_1', 'any_1', 'shape_cat', 'any_cat'])
        parser.add_argument('--isinstance_hidden_dim', type=int, default=3)
        parser.add_argument('--isinstance_size', type=int, default=10)
        parser.add_argument('--isinstance_length_epoch', type=int, default=10000)
        parser.add_argument('--isinstance_epochs', type=int, default=100)

        parser.add_argument('--generalization_ratio', type=float, default=0.25)
        parser.add_argument('--conceptual_question_ratio', type=float, default=0.2)

        return parser.parse_args()

    def post_process(self):
        dicts = self.__dict__
        self.visualize_dir = os.path.join(self.visualize_dir, self.model, self.name)
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.model)
        if os.path.exists(self.visualize_dir):
            shutil.rmtree(self.visualize_dir)
        os.makedirs(self.visualize_dir)
        self.num_gpus = torch.cuda.device_count()
        self.use_cuda = self.num_gpus > 0
        self.info.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.toy_categories = min(self.toy_categories, self.toy_attributes)
        self.load_by = 'question' if self.mode in ['concept-net']\
            else 'image'
        self.group = self.task.split('_')[0]
        for group, arguments in self.dir_args.items():
            for arg in arguments:
                dicts[arg] = os.path.join(dicts[group+'_data_dir'],
                                          dicts[arg])
                if group == self.group:
                    dicts[arg.replace(group+'_', '')] = dicts[arg]

        self.conceptual = False
        for k in self.conceptual_tokens:
            if k in self.subtask:
                self.conceptual = True
        if self.no_validation:
            self.generalization_ratio = 0

    def print(self):
        pprint.pprint('Arguments: ------------------------')
        pprint.pprint(self.__dict__)
        pprint.pprint('-----------------------------------')
