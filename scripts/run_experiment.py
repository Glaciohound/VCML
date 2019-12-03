#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : run_experiment.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 04.08.2019
# Last Modified Date: 03.12.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license


import os
import sys
import torch.multiprocessing as mp
from IPython import embed
from IPython.core import ultratb
import torch

from experiments import lazy_import
from scripts.control.monitor import Monitor
from scripts.control.coach import Coach
from utility.common import init_seed, plt
from utility.logging import Logger
from utility.recording import AverageGroup
from utility.load_ckpt import download_ckpt

from scripts.utils import register
from scripts.utils.prepare import\
    load_training_visual_dataset, \
    print_args, load_model, load_for_schedule, load_ref_dataset, \
    questions_directly, get_parser
from scripts.utils.train import train


def ready_go(args, index, message, control, device, is_main):
    # Initialization
    sys.excepthook = ultratb.FormattedTB(
        mode='Plain', color_scheme='Linux', call_pdb=is_main)
    torch.cuda.set_device(device)
    local_dir = os.path.join(args.local_log_dir, str(index))

    init_seed(args.random_seed, index)
    logger = Logger(local_dir, is_main, args.silent)
    logger(' '.join(sys.argv))
    print_args(args, logger)

    config = lazy_import()['run-experiment',
                           args.task,
                           args.name].config
    dataset_config = args.dataset_config[args.task]
    if args.dataset == '':
        dataset = config['dataset']
    else:
        dataset = args.dataset

    # sceneGraphs
    logger('Loading sceneGraphs')
    with logger.levelup():
        filename = os.path.join(args.dataset_dir, 'sceneGraphs.pkl')
        visual_dataset = load_training_visual_dataset(
            args, filename, logger, index)
        visual_dataset.set_inputs(args.visual_inputs)
        visual_dataset.match_images(args.image_dir)
    if 'test_ref' in args.in_epoch:
        with logger.levelup():
            ref_dataset = load_ref_dataset(
                args, logger, index,
                dataset_config['scene_process'])
            ref_dataset.set_inputs(args.visual_inputs)

    # word2indexes
    logger.line_break()
    logger('Loading Word_Index instances')
    with logger.levelup():
        tools = register.init_word2index(logger)
        if args.old:
            tools.load(os.path.join(args.dataset_dir,
                                    'my_dataset', dataset))
        else:
            tools.load(os.path.join(args.dataset_dir, 'questions', args.name))
        tools.count([('arguments', 'concepts'), ('concepts', 'arguments')])
        logger.line_break()

    # question dataset
    logger('Loading question dataset')
    with logger.levelup():
        if args.old:
            schedule = config['training_schedule'](
                tools.concepts, args, dataset, logger)
        else:
            aug_questions_dir = os.path.join(
                args.dataset_dir, 'questions', args.experiment,
                str(index))
            schedule = questions_directly(aug_questions_dir, args, logger)
        load_for_schedule(schedule, visual_dataset, tools)

    # question parser
    logger.line_break()
    logger('Loading question parser')
    with logger.levelup():
        question_parser = get_parser(args, device, logger, index, is_main)

    # building model
    logger.line_break()
    logger('Building model')
    with logger.levelup():
        model = load_model(args, tools, device, logger)

    # recording
    logger('Building recordings')
    group_arg = (args.history_length, args.local_log_dir)
    recordings = {
        'train_recording': AverageGroup('train', 'exponential', *group_arg),
        'val_recording': AverageGroup('val', 'simple', *group_arg),
        'test_recording': AverageGroup('test', 'simple', *group_arg),
    }
    if 'test_ref' in args.in_epoch:
        recordings['ref_recording'] = AverageGroup(
            'test_ref', 'simple', *group_arg)

    logger('Building Coach')
    coach = Coach(
        args, index, schedule, question_parser, model,
        tools, recordings, logger,
        local_dir, message, control, device,
        plt, is_main,
    )
    if 'test_ref' in args.in_epoch:
        coach.add_ref_dataset(ref_dataset)

    coach.training_init()
    coach.schedule_dataset()

    # Loading checkpoints if necessary
    if args.resume:
        if args.ckpt_index is not None:
            coach.set_index(args.ckpt_index)
        logger(f'Loading checkpoint: {coach.index}: {args.ckpt_version}')
        coach.load(args.ckpt_version)
    if args.pretrained:
        logger(f'Loading checkpoint ...')
        ckpt = download_ckpt(args, args.task, args.name, index, is_main)
        coach.load_partial(ckpt)
    coach.tools.operations.register_special()

    # go baby go
    if args.ipython and is_main:
        embed()
    train(coach, args)
    if is_main:
        embed()


def run(args):
    ctx = mp.get_context('spawn')

    message = [ctx.Queue() for i in range(args.num_parallel)]
    control = [ctx.Queue() for i in range(args.num_parallel)]
    processes = []

    for i in range(0, args.num_parallel):
        device = args.cudas[i * args.num_gpus // args.num_parallel]
        p = ctx.Process(target=ready_go,
                        args=(args, i, message[i], control[i],
                              device, i == 0))
        processes.append(p)

    monitor = Monitor(args, message, control)
    p = ctx.Process(target=monitor.monitor, args=())
    processes.append(p)

    for i, p in enumerate(processes[1:]):
        p.start()
    processes[0]._target(*processes[0]._args)

    for p in processes:
        p.join()
