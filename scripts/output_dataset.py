#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : output_dataset.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 10.09.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


import os
import sys
import torch
import torchvision
import torch.multiprocessing as mp
from IPython import embed
from IPython.core import ultratb
import json
import h5py
import numpy as np
from cv2 import imread
from PIL import Image
import copy

from experiments import lazy_import
from utility.common import init_seed, make_dir
from utility.logging import Logger

from scripts.utils import register
from scripts.utils.prepare import\
    load_training_visual_dataset, \
    print_args, load_for_schedule


sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)


def one_dataset(args, index, logger):
    init_seed(args.random_seed, index)

    config = lazy_import()['run-experiment',
                           args.task,
                           args.experiment].config
    if args.dataset == '':
        dataset = config['dataset']
    else:
        dataset = args.dataset

    logger('Loading sceneGraphs')
    with logger.levelup():
        filename = os.path.join(args.dataset_dir, 'sceneGraphs.pkl')
        visual_dataset = load_training_visual_dataset(
            args, filename, logger, index)
        visual_dataset.set_inputs(args.visual_inputs)

    # word2indexes
    logger.line_break()
    logger('Loading Word_Index instances')
    with logger.levelup():
        tools = register.init_word2index(logger)
        tools.load(os.path.join(args.dataset_dir, 'my_dataset', dataset))
        tools.count([('arguments', 'concepts'), ('concepts', 'arguments')])
        logger.line_break()

    # question dataset
    logger('Loading question dataset')
    with logger.levelup():
        schedule = config['training_schedule'](
            tools.concepts, args, dataset, logger)
        load_for_schedule(schedule, visual_dataset, tools)

    assert len(schedule) == 1
    return visual_dataset, schedule, tools


def output_features(images, image_filename_dict, feature_h5, raw_h5,
                    args, device, logger):
    # Copyright 2017-present, Facebook, Inc.
    # All rights reserved.
    #
    # This source code is licensed under the license found in the
    # MAC_LICENSE file in the same directory of this file.

    num = len(images)
    channels = 1024
    image_height = 224
    image_width = 224
    dim_x = 14
    dim_y = 14
    batch_size = args.batch_size
    model_stage = 3
    img_size = (image_height, image_width)

    null_image = np.zeros((image_height, image_width, 3), dtype=float)

    def build_model():
        cnn = torchvision.models.resnet101(pretrained=True)
        layers = [
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
        ]
        for i in range(model_stage):
            name = 'layer%d' % (i + 1)
            layers.append(getattr(cnn, name))
            model = torch.nn.Sequential(*layers)
            model.to(device)
            model.eval()
        return model

    def run_batch(cur_batch, model):
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

        image_batch = (cur_batch / 255.0 - mean) / std
        image_batch = torch.FloatTensor(image_batch).to(device)
        image_batch = torch.autograd.Variable(image_batch, volatile=True)

        feats = model(image_batch)
        feats = feats.data.cpu().clone().numpy()
        return feats

    model = build_model()

    logger('Output features to file')
    f_features = h5py.File(feature_h5, 'w')
    f_raw = h5py.File(raw_h5, 'w')
    feat_dset = f_features.create_dataset(
        'features', (num, channels, dim_x, dim_y), dtype=np.float32)
    raw_dset = f_raw.create_dataset(
        'features', (num, 3, image_height, image_width), dtype=np.float32)
    i0 = 0
    cur_batch = []
    pbar = logger.tqdm(images)
    for i, image_id in enumerate(pbar):
        if image_id in image_filename_dict:
            filename = image_filename_dict[image_id]
            img = imread(filename)
            img = np.array(Image.fromarray(img).resize(img_size))
        elif image_id == args.null_image:
            img = null_image
        else:
            raise Exception(f'no such image found: {filename}')
        img = img.transpose(2, 0, 1)[None]
        cur_batch.append(img)
        if len(cur_batch) == batch_size or i == len(images) - 1:
            batch_np = np.concatenate(cur_batch, 0).astype(np.float32)
            feats = run_batch(batch_np, model)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = feats
            raw_dset[i0:i1] = batch_np
            i0 = i1
            cur_batch.clear()


def output_questions(question_dataset, image_index, question_json,
                     args, logger):
    output_questions = []
    pbar = logger.tqdm(question_dataset.question_list)
    for i, question in enumerate(pbar):
        new_question = copy.deepcopy(question)
        new_question['question_index'] = i
        new_question['image_index'] = image_index[question['image_id']]
        output_questions.append(new_question)
    output = {
        'info': f'This is a augmentated question dataset for {args.task} '
                f'{args.experiment} experiment',
        'questions': output_questions
    }
    logger('Output to file')
    with open(question_json, 'w') as f:
        json.dump(output, f)


def output_one_split(visual_dataset, question_dataset,
                     local_output_dir, split, args, device, logger):
    images = list(set(q['image_id'] for q in question_dataset.question_list))
    image_filename_dict = {
        scene['image_id']: scene['image_filename'] for scene in
        visual_dataset.sceneGraphs.values()
    }
    image_index = dict(zip(images, range(len(images))))

    question_json = os.path.join(
        local_output_dir, f'{split}_questions.json')
    logger(f'Outputting question file to {question_json}')
    logger(f'{len(question_dataset)} questions in total', resume=True)
    with logger.levelup():
        output_questions(question_dataset, image_index, question_json,
                         args, logger)

    if args.output_visual:
        feature_h5 = os.path.join(local_output_dir, split+'.h5')
        raw_h5 = os.path.join(local_output_dir, split+'_raw.h5')
        logger(f'Outputting feature file to {feature_h5}, '
               'raw_images to {raw_h5}')
        logger(f'{len(images)} images in total', resume=True)
        with logger.levelup():
            output_features(images, image_filename_dict, feature_h5, raw_h5,
                            args, device, logger)


def output_one(args, index, device, is_main):
    # Initialization
    local_output_dir = os.path.join(args.output_dir, str(index))
    make_dir(local_output_dir)

    logger = Logger(local_output_dir, is_main, False)
    logger(' '.join(sys.argv))
    print_args(args, logger)

    logger(f'Building dataset indexed {index} to {local_output_dir}')
    with logger.levelup():
        visual_dataset, schedule, tools = one_dataset(args, index, logger)
        for split in ['test', 'train', 'val']:
            output_one_split(visual_dataset,
                             schedule[0]['question_splits'][split],
                             local_output_dir, split, args, device, logger)
        tools.save(local_output_dir)

    if is_main:
        embed()


def output_dataset(args):
    ctx = mp.get_context('spawn')
    processes = []

    for i in range(0, args.num_parallel):
        device = args.cudas[i * args.num_gpus // args.num_parallel]
        p = ctx.Process(target=output_one,
                        args=(args, i, device, i == 0))
        processes.append(p)

    for i, p in enumerate(processes[1:]):
        p.start()
    processes[0]._target(*processes[0]._args)

    for p in processes[1:]:
        p.join()
