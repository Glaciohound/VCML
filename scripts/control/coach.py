#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : coach.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# for building training dataset and scheduling training

import os
import torch
from shutil import copyfile

from utility.common import make_dir, init_seed


class Coach:
    def __init__(self,
                 args, index, schedule, question_parser, model,
                 tools, recordings, logger, local_dir,
                 message, control, device, plt, is_main):
        self.args = args
        self.index = index
        self.schedule = schedule
        self.question_parser = question_parser
        self.model = model
        self.tools = tools
        self.logger = logger
        self.local_dir = local_dir
        self.message = message
        self.control = control
        self.device = device
        self.plt = plt
        self.is_main = is_main

        self.__dict__.update(recordings)
        model.set_coach(self)
        self.init_value()

    def init_value(self):
        self.stage_ind = -1
        self.epoch = -1

    def add_ref_dataset(self, ref_dataset):
        self.ref_dataset = ref_dataset

    def step(self, i_epoch):
        ind = 0
        for i, item in enumerate(self.stages):
            if item['start_time'] <= i_epoch:
                ind = i

        self.stage_ind = ind
        self.setup_dataset(ind)

    def schedule_dataset(self):
        self.stages = []
        offset = 0
        for item in self.schedule:
            new_dataset = {
                'start_time': offset,
                'end_time': offset + item['length'],
                'question_splits': item['question_splits'],
                'test_concepts': item['test_concepts'],
            }
            self.stages.append(new_dataset)
            offset = new_dataset['end_time']

    def setup_dataset(self, ind):
        dataset_suite = self.stages[ind]['question_splits']
        self.train = dataset_suite['train'].get_dataloader()
        self.val = dataset_suite['val'].get_dataloader()
        self.test = dataset_suite['test'].get_dataloader()

    def torch_scheduler_step(self):
        val_loss = self.val_recording.group['loss']
        self.step(val_loss.value)

    def state_dict(self):
        stages_ckpt = [
            {
                'start_time': stage['start_time'],
                'end_time': stage['end_time'],
                'question_splits': {
                    split: dataset.state_dict()
                    for split, dataset
                    in stage['question_splits'].items()
                },
                'test_concepts': stage['test_concepts'],
            } for stage in self.stages
        ]
        ckpt = {
            'model': self.model.state_dict(),
            'tools': self.tools.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'logger': self.logger.state_dict(),
            'train_recording': self.train_recording.state_dict(),
            'val_recording': self.val_recording.state_dict(),
            'test_recording': self.test_recording.state_dict(),
            'stage_ind': self.stage_ind,
            'stages': stages_ckpt,
            'epoch': self.epoch,
        }
        if hasattr(self, 'ref_recording'):
            ckpt['ref_recording'] = self.ref_recording.state_dict()
        return ckpt

    def partial_state_dict(self):
        ckpt = {
            'model': self.model.state_dict(),
            'tools': self.tools.state_dict(),
        }
        return ckpt

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt['model'])
        self.tools.load_state_dict(ckpt['tools'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

        self.logger.load_state_dict(ckpt['logger'])
        self.train_recording.load_state_dict(ckpt['train_recording'])
        self.val_recording.load_state_dict(ckpt['val_recording'])
        self.test_recording.load_state_dict(ckpt['test_recording'])
        if 'ref_recording' in ckpt and hasattr(self, 'ref_recording'):
            ref_ckpt = ckpt['ref_recording']
            if not isinstance(ref_ckpt, dict):
                ref_ckpt = ref_ckpt.state_dict()
            self.ref_recording.load_state_dict(ref_ckpt)
        self.stage_ind = ckpt['stage_ind']
        self.epoch = ckpt['epoch']

        stages_ckpt = ckpt['stages']
        for i, stage in enumerate(self.stages):
            stage['start_time'] = stages_ckpt[i]['start_time']
            stage['end_time'] = stages_ckpt[i]['end_time']
            for split, dataset in stage['question_splits'].items():
                dataset.load_state_dict(
                    stages_ckpt[i]['question_splits'][split])
            stage['test_concepts'] = stages_ckpt[i]['test_concepts']

    def load_partial(self, ckpt):
        # load only what necessary for the pretrained model
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.tools.load_state_dict(ckpt['tools'])

    @property
    def ckpt_dir(self):
        ckpt_dir = os.path.join(self.args.local_log_dir, 'ckpt')
        return ckpt_dir

    def ckpt_filename(self, suffix, index):
        filename = os.path.join(
            self.ckpt_dir,
            f'Coach{index}_{suffix}.pth'
        )
        return filename

    def save(self, suffix, index=None):
        if index is None:
            index = self.index
        ckpt = self.state_dict()
        self.make_dir()
        torch.save(ckpt, self.ckpt_filename(suffix, index))

    def save_partial(self, suffix, index=None):
        if index is None:
            index = self.index
        ckpt = self.partial_state_dict()
        self.make_dir()
        torch.save(ckpt, self.ckpt_filename(str(suffix) + '_partial', index))

    def copy_ckpt(self, from_suffix, to_suffix, index=None):
        if index is None:
            index = self.index
        from_name = self.ckpt_filename(from_suffix, index)
        to_name = self.ckpt_filename(to_suffix, index)
        copyfile(from_name, to_name)

    def load(self, suffix, index=None):
        if index is None:
            index = self.index
        ckpt = torch.load(
            self.ckpt_filename(suffix, index),
            map_location=self.device
        )
        self.load_state_dict(ckpt)

    def make_dir(self):
        make_dir(self.ckpt_dir)

    def wait(self):
        let_go = False
        while not let_go:
            command = self.get()
            if command['type'] == 'let_go':
                let_go = True
            else:
                self.process(command)

    def ready(self):
        self.send('ready')

    def process(self, command):
        if command['type'] == 'log':
            self.logger(command['content'], pretty=True)
        else:
            raise Exception(f'unrecognized command: {command}')

    def send(self, item):
        self.message.put(item)

    def get(self):
        command = self.control.get()
        return command

    def training_init(self):
        init_seed(self.args.random_seed)
        self.model.to(self.device)
        self.optimizer, self.scheduler = self.model.init()

    def synchronize(self):
        self.ready()
        self.wait()

    def set_index(self, index):
        self.index = index
