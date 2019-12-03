#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 03.12.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

import os

from dataset.visual_dataset.visual_dataset import Dataset
from dataset.question_dataset.question_dataset import Dataset as QDataset
from dataset.visual_dataset.utils.sceneGraph_loader import \
    load_multiple_sceneGraphs
from utility.common import load, make_dir
from utility.cache import Cache
from reason.models.parser import Seq2seqParser
from . import register


def load_visual_dataset(args, logger, process_fn):
    with Cache(args.task+'_sceneGraphs', logger, args) as cache:
        if not cache.exist():
            cache.cache(load_multiple_sceneGraphs(
                args.sceneGraph_dir, args, logger, process_fn))
        sceneGraphs = cache.obj
    visual_dataset = Dataset(args, logger, sceneGraphs, 0).get_agent()
    logger(f'SceneGraphs size: {len(visual_dataset)}')

    return visual_dataset


def split_visual_dataset(dataset_dir, visual_dataset, config, logger):
    visual_split_config = load(
        os.path.join(dataset_dir, config.visual_split_dir))
    visual_splits = visual_dataset.resplit(visual_split_config)
    return visual_splits


def print_args(args, logger):
    logger.showtime()
    logger.split_line()
    logger('Printing Arguments')
    logger(args.str)
    logger.split_line()
    logger.line_break()


def load_training_visual_dataset(args, filename, logger, index):
    # filename = os.path.join(path, dataset, 'sceneGraphs.pkl')
    sceneGraphs = load(filename)
    logger(f'Loaded sceneGraphs from: {filename}')
    logger(f'SceneGraphs size: {len(sceneGraphs)}', resume=True)
    visual_dataset = Dataset(args, logger, sceneGraphs, index).get_agent()
    return visual_dataset


def load_model(args, tools, device, logger):

    if args.model in ('VCML', 'NSCL', 'BERTvariant'):
        from models.model.vcml_model import VCML_Model
        model = VCML_Model(args, tools, device, logger)

    elif args.model == 'BERT':
        from models.model.bert_model import BERTModel
        model = BERTModel(args, tools, device, logger)

    elif args.model.startswith('GRU'):
        use_vision = args.model == 'GRUCNN'
        use_pretrained = args.pretrained_embedding
        finetune = args.finetune_embedding
        assert not (args.force_off_lm and args.force_on_lm), \
            'force-on / off can not be both true'
        use_lm = (not use_pretrained and not args.force_off_lm) or \
            args.force_on_lm
        from models.model.gru_model import GRUModel
        model = GRUModel(args, tools, device, logger,
                         use_vision=use_vision,
                         fix_vision=args.fix_resnet,
                         use_pretrained=use_pretrained,
                         finetune=finetune,
                         use_lm=use_lm)

    '''
    elif args.model == 'MAC':
        from models.model.mac_model import MAC_agent
        model = MAC_agent(args, tools, device)
    '''

    return model


def load_for_schedule(schedule, visual_dataset, tools):
    for stage in schedule:
        for dataset in stage['question_splits'].values():
            dataset.load_parts(visual_dataset, tools)


def questions_directly(path, args, logger=None):
    if logger is not None:
        logger(f'Loading questions from {path}')
    suite = {split: QDataset(load(os.path.join(
        path, f'{split}_questions.json'))['questions'],
                            args).get_agent()
             for split in ['train', 'test', 'val']}

    schedule = [
        {
            'length': args.epochs,
            'question_splits': suite,
            'test_concepts': None,
        }
    ]
    if logger is not None:
        for split in ('train', 'val', 'test'):
            logger(f'{split} questions size = {len(suite[split])}',
                   resume=True)
    return schedule


def load_ref_dataset(args, logger, index, process_fn):
    filename = os.path.join(args.ref_scene_json)
    logger(f'Loading referential-expression dataset from {filename}')
    sceneGraphs = load(filename)
    processed = process_fn(sceneGraphs, '', logger, args)
    visual_dataset = Dataset(args, logger, processed, index,
                             image_dir=args.ref_image_dir
                             ).get_agent()
    logger(f'SceneGraphs size: {len(visual_dataset)}', resume=True)
    return visual_dataset


def get_parser(args, device, logger, index, is_main):
    class fixed_opt:
        def __init__(self, **kwarg):
            self.__dict__.update(kwarg)

    if args.task in ['CLEVR', 'GQA']:
        ckpt_name_dir = args.task + '_reason'
    elif 'meronym' in args.name:
        ckpt_name_dir = 'CUB_meronym_reason'
    else:
        ckpt_name_dir = 'CUB_hypernym_reason'
    ckpt_name = ckpt_name_dir + '.tgz'
    temp_dir = os.path.join(args.temp_dir, 'vcml_reason', str(index))
    make_dir(temp_dir)

    ckpt_link = os.path.join(args.webpage, 'ckpt', ckpt_name)
    ckpt_file = os.path.join(temp_dir, ckpt_name)
    ckpt_dir = os.path.join(temp_dir, ckpt_name_dir)

    logger(f'Loading question parser from {ckpt_link}')
    os.system(f'rm -f {ckpt_file}')
    if is_main:
        os.system(f'wget {ckpt_link} -P {temp_dir}')
    else:
        os.system(f'wget -q {ckpt_link} -P {temp_dir}')
    os.system(f'mkdir {ckpt_dir} && tar xf {ckpt_file} -C {ckpt_dir}')

    opt = fixed_opt(
        load_checkpoint_path=os.path.join(
            temp_dir, ckpt_name_dir, 'checkpoint.pt'),
        gpu_ids=[0],
        fix_embedding=False
    )

    with logger.levelup():
        tools = register.init_word2index(logger)
        tools.load(ckpt_dir)
        tools.operations.register_special()
        parser = Seq2seqParser(opt, tools, device)

    os.system(f'rm -r {ckpt_dir} {ckpt_file}')
    return parser
