#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : config.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 26.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


import argparse
import jacinle
import torch
import os
import shutil
from pprint import pprint

from utility.common import \
    make_dir, is_empty, yes_or_no, copytree_verbose
from .dataset_config import dataset_config_register


class Args:
    def __init__(self):
        raw_args = self.parse_args()
        raw_args = self.post_process(raw_args)
        self.__dict__.update(vars(raw_args))

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        dir_args = {}
        group = None

        def dir_add_argument(arg_name, **kwargs):
            assert group is not None
            if group not in dir_args:
                dir_args[group] = []
            arg_name = '--' + group + '_' + arg_name[2:]
            dir_args[group].append(arg_name.strip('-'))
            parser.add_argument(arg_name, **kwargs)

        # options determining what experiment to run
        parser.add_argument('--mode', type=str, default='run-experiment',
                            required=True,
                            choices=['run-experiment',
                                     'build-dataset',
                                     'output-dataset'],
                            )
        parser.add_argument('--task', required=True,
                            choices=['GQA', 'CLEVR', 'CUB'])
        parser.add_argument('--dataset', type=str, default='')
        parser.add_argument('--visual_inputs', type=str, nargs='*',
                            choices=['plain', 'image', 'detection'],
                            default=['image', 'detection'],
                            )
        parser.add_argument('--experiment', type=str, required=True,
                            help='controls the dataset processing')
        parser.add_argument('--num_parallel', type=int, default=1)
        parser.add_argument('--ipython', action='store_true',
                            help='ipython embedding before running')
        parser.add_argument('--in_epoch', nargs='+', type=str,
                            default=['train', 'test', 'val'],
                            choices=['train', 'test', 'val',
                                     'test_ref'])
        parser.add_argument('--test_every', type=int, default=1)
        parser.add_argument('--old', action='store_true',
                            help='run in an old way')
        parser.add_argument('--webpage',
                            default='http://vcml.csail.mit.edu/data')

        # settings for training parameters
        parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
        parser.add_argument('--init_variance', type=float, default=1e-4,
                            help='parameter initialization variance')
        parser.add_argument('--random_seed', type=int, default=-1,
                            help='-1 if no manual random seed is set')
        parser.add_argument('--curriculum_training', type=int, default=0)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--balance_classification', action='store_true')

        # model-related options
        parser.add_argument('--model_similarity', default='cosine',
                            choices=['cosine', 'tree_cosine'])
        parser.add_argument('--model', default='VCML',
                            choices=[
                                'VCML',
                                'NSCL',
                                'GRUCNN',
                                'BERTvariant',
                                'GRU',
                                'BERT',
                                # 'MAC',
                            ])

        # arguments determining methods for building & composing dataset
        parser.add_argument('--size', type=int, default=600000)
        parser.add_argument('--questionsPimage', type=int, default=3)
        parser.add_argument('--with_checktable', action='store_true')
        parser.add_argument('--filter_out_original_test', action='store_true')
        parser.add_argument('--confidence_th', type=int, default=4,
                            help='when dataset')
        parser.add_argument('--fewshot_ratio', type=float, default=0.1)

        parser.add_argument('--most_frequent', type=int, default=-1,
                            help='use most frequent concepts')
        parser.add_argument('--generalization_ratio', type=float, default=0.25,
                            help='ratio of concepts for generalization')
        parser.add_argument('--conceptual_question_ratio', type=float,
                            default=0.1,
                            help='ratio of conceptual questions')
        parser.add_argument('--split_ratio', type=float, nargs='+',
                            default=[0.7, 0.2, 0.1])
        parser.add_argument('--debiasing_leak', type=int, default=200)
        parser.add_argument('--visual_mix_ratio', type=float, default=0.01)
        parser.add_argument('--output_visual', action='store_true')

        # adding dataset-specific relative directories

        parser.add_argument('--data_dir', default='../data/vcml_data/')
        # GQA dataset
        group = 'GQA'
        parser.add_argument('--GQA_root_dir',
                            default='GQA')
        dir_add_argument('--image_dir', default='raw/images')
        dir_add_argument('--sceneGraph_dir', default='raw/sceneGraphs/')
        dir_add_argument('--dataset_dir',
                         default='augmentation')

        # CLEVR dataset
        group = 'CLEVR'
        parser.add_argument('--CLEVR_root_dir',
                            default='CLEVR')
        dir_add_argument('--image_dir', default='raw/images')

        dir_add_argument('--sceneGraph_dir', default='detections')
        dir_add_argument('--dataset_dir',
                         default='augmentation')
        dir_add_argument('--ref_scene_json', type=str,
                         default='kexin_scenes/scenes.json')
        dir_add_argument('--ref_image_dir', type=str,
                         default='kexin_scenes/images')

        # CUB dataset
        group = 'CUB'
        parser.add_argument('--CUB_root_dir',
                            default='CUB')
        dir_add_argument('--image_dir', default='raw/images')

        dir_add_argument('--sceneGraph_dir',
                         default='augmentation/sceneGraphs')
        dir_add_argument('--dataset_dir',
                         default='augmentation')

        # image setting
        parser.add_argument('--box_scale', type=int, default=1024,
                            help='bounding box size')
        parser.add_argument('--image_scale', type=int, default=224)
        parser.add_argument('--null_image', default='NULLIMAGE')

        # visualization & checkpointing
        parser.add_argument('--log_dir', type=str,
                            default='../data/log')
        parser.add_argument('--glove_pretrained_dir',
                            default='../data/log/checkpoints')
        parser.add_argument('--history_length', type=int, default=10000)
        parser.add_argument('--silent', action='store_true',
                            help='turning off logging and visualizing')
        parser.add_argument('--visualize_relation', type=str,
                            help='metaconcept for visualization')
        parser.add_argument('--name', type=str, default='')
        parser.add_argument('--pretrained', action='store_true')
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--clear_cache', action='store_true')
        parser.add_argument('--cache_dir', default='cache/')
        parser.add_argument('--not_build_reasoning', action='store_true')
        parser.add_argument('--temp_dir', type=str,
                            default='/tmp/torch_extensions_hc')

        # model parameters
        parser.add_argument('--true_th', type=float, default=0.7)
        parser.add_argument('--temperature', type=float, default=10)
        parser.add_argument('--conceptual_weight', type=float, default=1,
                            help='weight for conceptual questions')
        parser.add_argument('--embed_dim', type=int, default=50)
        parser.add_argument('--feature_dim', type=int, default=512)
        parser.add_argument('--metaconcept_hidden_dim', type=int, default=10)
        parser.add_argument('--hidden_dim', type=int, default=100)
        parser.add_argument('--fix_resnet', action='store_true')
        parser.add_argument('--other_offset', default=0., type=float,
                            help='offset for logits of other concepts')
        parser.add_argument('--sample_size', default=100000, type=int,
                            help='num of sample points for intergral')
        parser.add_argument('--detach_in_rel', action='store_true',
                            help='detach concept in relation calculation')
        parser.add_argument('--pretrained_embedding', action='store_true')
        parser.add_argument('--finetune_embedding', action='store_true')
        parser.add_argument('--force_off_lm', action='store_true')
        parser.add_argument('--force_on_lm', action='store_true')
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--penalty', type=float, default=0)
        parser.add_argument('--length_penalty', type=float, default=0)

        # temporary configurations for debugging
        parser.add_argument('--test_only', action='store_true')
        parser.add_argument('--concept_filter', type=int, nargs='*',
                            default=[])
        parser.add_argument('--use_gt_program', action='store_true')

        args = parser.parse_args()
        args.dir_args = dir_args
        return args

    def post_process(self, args):
        if args.name == '':
            args.name = args.experiment
        args.dataset_config = dataset_config_register
        args.root_dir = os.path.abspath('.')
        args.cache_dir = os.path.join(args.data_dir, args.cache_dir)

        # pytorch config initialization
        args.num_gpus = torch.cuda.device_count()
        args.use_cuda = args.num_gpus > 0
        if args.use_cuda:
            args.cudas = [
                torch.device(f'cuda: {i}')
                for i in range(args.num_gpus)
            ]

        # getting group-related paths
        group = args.task
        args.group_dir = os.path.join(
            args.data_dir, getattr(args, group+'_root_dir'))
        for arg_name in args.dir_args[group]:
            path_arg = getattr(args, arg_name)
            setattr(args, arg_name.replace(group+'_', ''),
                    os.path.join(args.group_dir, path_arg))

        # building directories
        if args.mode == 'run-experiment':
            self.makedir_runExperiment(args)
        elif args.mode == 'build-dataset':
            self.makedir_buildDataset(args)
        elif args.mode == 'output-dataset':
            self.makedir_outputDataset(args)

        # dataset split
        args.split_ratio = dict(zip(
            ['train', 'val', 'test'],
            args.split_ratio
        ))

        return args

    def update(self, args):
        self.__dict__.update(args)

    @property
    def str(self):
        dicts = {k: str(v) for k, v in self.__dict__.items()}
        formatted = jacinle.kvformat(dicts)
        return formatted

    def makedir_runExperiment(self, args):
        args.local_log_dir = os.path.join(
            args.log_dir, args.model, args.task, args.name)
        args.cache_log_dir = os.path.join(args.log_dir, 'cache')

        if not args.silent and not is_empty(args.local_log_dir):
            cache_or_not(args.local_log_dir, args.cache_log_dir,
                         '.'.join([args.model, args.task]),
                         not args.pretrained)
        if not args.silent:
            make_dir(args.local_log_dir)

        if args.resume:
            if not args.silent:
                load_or_not(args.cache_log_dir, args.local_log_dir,
                            '.'.join([args.model, args.task]))
            args.ckpt_version = input(
                'Which version of ckpt you want to load? (default: `best`): '
            ) or 'best'
            if args.num_parallel == 1:
                args.ckpt_index = input(
                    'There is only one trial in your current run, what index '
                    'would you like to load? (default: 0): '
                ) or '0'
            else:
                args.ckpt_index = None

    def makedir_outputDataset(self, args):
        args.output_dir = os.path.join(
            args.group_dir, 'augmentation/questions',
            args.name)
        shutil.rmtree(args.output_dir, True)

    def makedir_buildDataset(self, args):
        pass


def cache_or_not(path, log_dir, tag, remove):
    if not yes_or_no(f'non-empty {path} detected, cache it?'):
        if remove:
            assert yes_or_no('The path will be removed, sure about it?')
            print('Removing the path')
            shutil.rmtree(path)
    else:
        print('Caching the folder')
        make_dir(log_dir)
        cache_dir = os.path.join(log_dir, tag)
        suffix = input('please specify the name for this cache:'
                       f' {cache_dir}.')
        if suffix != '':
            cache_dir = f'{cache_dir}.{suffix}'

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        if remove:
            shutil.move(path, cache_dir)
        else:
            copytree_verbose(path, cache_dir)


def load_or_not(cache_dir, target_dir, tag):
    if yes_or_no('Load previous cache as ckpt?'):
        shutil.rmtree(target_dir)
        sub_folders = os.listdir(cache_dir)
        contain_tag = list(filter(lambda x: tag in x, sub_folders))
        pprint(dict(enumerate(contain_tag)))
        index = int(input('Which folder would you choose? '))
        folder = os.path.join(cache_dir, contain_tag[index])
        copytree_verbose(folder, target_dir)
