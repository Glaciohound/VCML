#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : question_dataset.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

import torch
import torch.utils.data
import copy

from .utils import collate_utils, program_utils, question_utils
from .agent_question_dataset import AgentDataset


class Dataset(torch.utils.data.Dataset):

    def __init__(self, questions, args):
        self.args = args
        self.load_questions(questions)
        self.program_translator = program_utils.semantic2program

    def load_parts(self, visual_dataset, tools):
        self.tools = tools
        self.visual_dataset = visual_dataset

    def __getitem__(self, index):

        question = self.questions[index]
        program = self.program_translator(question['semantic'])
        answer = question['answer']

        data = {
            'index': index,
            'type': question['type'],
            'category': question['category'],
            'confidence': question['confidence'],
            'question': question['question'],
            'program': program,
            'answer': answer,
        }
        data.update(self.encode(data))

        image_id = question['image_id']
        if self.visual_dataset is not None:
            scene = self.visual_dataset.base_dataset[image_id]
            data.update(scene)

        return data

    def encode(self, data):
        tools = self.tools
        encoded = {
            'question_encoded': question_utils.encode_question(
                data['question'], tools.words),
            'program_encoded': program_utils.encode_program(
                data['program'],
                tools.operations,
                tools.arguments,
            ),
        }
        if isinstance(data['answer'], str):
            encoded['answer_encoded'] = tools.answers[data['answer']]
        else:
            encoded['answer_encoded'] = data['answer']

        return encoded

    def load_questions(self, questions):
        if isinstance(questions, list):
            questions = {q['question_index']: q for q in questions}
        self.questions = questions
        self.indexes = list(self.questions.keys())
        for index, q in self.questions.items():
            q['index'] = index
            if q['category'] == 'conceptual':
                q['image_id'] = self.args.null_image

    @property
    def collate_fn(self):

        collate_setting = {
            'index': {'type': 'list', 'tensor': False},
            'image_id': {'type': 'list', 'tensor': False},
            'type': {'type': 'list', 'tensor': False},
            'category': {'type': 'list', 'tensor': False},

            'question': {'type': 'list', 'tensor': False},
            'program': {'type': 'list', 'tensor': False},
            'answer': {'type': 'list', 'tensor': False},
            'question_encoded': {
                'type': 'pad-stack',
                'pad_value': self.tools.words['<NULL>'],
                'tensor': True
            },
            'program_encoded': {'type': 'list', 'tensor': True},
            'answer_encoded': {'type': 'list', 'tensor': True},
            'confidence': {'type': 'list', 'tensor': False},

            'plain': {'type': 'list', 'tensor': False},
            'feature': {'type': 'concat', 'axis': 0, 'tensor': True},
            'image': {'type': 'stack', 'tensor': True},
            'objects': {'type': 'concat', 'axis': 0, 'tensor': True},
            'object_length': {'type': 'stack', 'tensor': True},
        }

        fn = collate_utils.collateFn(collate_setting)
        return fn

    def __len__(self):
        return len(self.questions)

    def load_indexes(self, indexes):
        if not isinstance(indexes, list):
            indexes = [indexes]
        questions = [
            self[ind] for ind in indexes
        ]
        return self.collate_fn(questions)

    def union(self, another, inplace=False):
        if inplace:
            self.questions.update(another.questions)
            self.indexes = list(self.questions.keys())
            return self
        else:
            union_questions = copy.copy(self.questions)
            union_questions.update(another.questions)
            output = Dataset(union_questions, self.args)
            return output

    def get_agent(self):
        return AgentDataset(self)

    def __copy__(self):
        raise NotImplementedError()

    def copy(self):
        questions = copy.deepcopy(self.questions)
        output = Dataset(questions, self.args)
        return output

    def state_dict(self):
        ckpt = {
            'questions': self.questions
        }
        return ckpt

    def load_state_dict(self, ckpt):
        self.load_questions(ckpt['questions'])
