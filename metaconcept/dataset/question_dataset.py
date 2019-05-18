import copy
import json

import numpy as np
import torch

from metaconcept import args, info
from metaconcept.dataset.tools import program_utils, question_utils
from metaconcept.dataset.tools import collate_utils
from metaconcept.dataset.toy import teddy_dataset


class Dataset(torch.utils.data.Dataset):
    inited = False

    def __init__(self, visual_dataset, config):
        self.valid_modes = ['encoded', 'plain']
        self.mode = 'plain'
        self.split = 'total'
        self.init()
        self.load_questions(visual_dataset, config)

    @classmethod
    def init(cls):
        if cls.inited:
            return
        cls.inited = True
        cls.program_utils = program_utils

    def __getitem__(self, index):
        try:
            return self.__getitem_inner__(index)
        except Exception as exc:
            print(exc)
            raise exc

    def __getitem_inner__(self, index_):
        if isinstance(index_, int):
            index = self.index[index_]

        question = self.questions[index]
        program_translator = program_utils.semantic2program_r if args.model == 'relation_model'\
            else program_utils.semantic2program_u if args.model == 'u_embedding'\
            else program_utils.semantic2program_h
        program = program_translator(question['semantic']) if 'semantic' in question else []
        question_encoded = question_utils.encode_question(question['question'], info.protocol)
        program_encoded = np.array(
            [[info.protocol['operations', op['operation']],
            info.protocol['concepts', op['argument']]]
            for op in program])
        scene = info.visual_dataset[question['image_id']] if hasattr(info, 'visual_dataset') else None
        answer = question['answer']
        answer_encoded = info.protocol['concepts', answer]

        entry = {
            'index': index,
            'type': question['type'],
            'question': question['question'] if self.mode == 'plain' else question_encoded,
            'program': program if self.mode == 'plain' else program_encoded,
            'answer': question['answer'] if self.mode == 'plain' else answer_encoded,
        }
        entry.update(scene)
        return entry

    def load_questions(self, visual_dataset, config):
        if args.subtasks == ['original']:
            print('Loading questions ... ', end='', flush=True)
            with open(args.questions_json, 'r') as f:
                questions = json.load(f)
            for q_id, question in questions.items():
                questions['id'] = q_id
            print('DONE')
        else:
            self.questions = teddy_dataset.ToyDataset.build_question_dataset(visual_dataset, config)

        self.split_indexes = {key: [q_id for q_id, q in self.questions.items()
                                    if q['split'] == key]
                                for key in ['train', 'test', 'val']}
        self.split_indexes['total'] = list(self.questions.keys())
        self.types = set(q['type'] for q in self.questions.values()
                        if 'type' in q)
        self.answers = set(q['answer'] for q in self.questions.values())

        print('Registering concepts ... ', end='', flush=True)
        question_utils.register_concepts(self.questions)
        print('DONE')

    def to_split(self, split):
        new_dataset = copy.copy(self)
        new_dataset.split = split
        return new_dataset

    def to_mode(self, mode):
        if mode in self.valid_modes:
            self.mode = mode
        else:
            raise Exception('invalid mode: %s' % mode)

    @classmethod
    def get_splits(cls, main_dataset):
        train, val, test = [main_dataset.to_split(s)
                            for s in ['train', 'val', 'test']]
        return train, val, test

    def __len__(self):
        return len(self.index)

    @property
    def index(self):
        return self.split_indexes[self.split]

    def assertion_checks(self, entry):
        pass

    def collate(self, data):
        if self.mode == 'plain':
            collate_setting = {
                'index': {'type': 'stack', 'tensor': False},
                'type': {'type': 'stack', 'tensor': False},
                'question': {'type': 'stack', 'tensor': False},
                'program': {'type': 'stack', 'tensor': False},
                'answer': {'type': 'stack', 'tensor': False},
                'scene_plain': {'type': 'stack', 'tensor': False},

                'scene': {'type': 'list', 'tensor': True},
                'image': {'type': 'stack', 'tensor': True},
                'objects': {'type': 'concat', 'axis': 0, 'tensor': True},
                'object_lengths': {'type': 'stack', 'tensor': True},
                'object_classes': {'type': 'list', 'tensor': True},
                'filter_fn': ('or', (
                    'classification' not in args.subtasks,
                    ('equal_in_length',
                     ('object_lengths', 'scene_plain', 'scene', 'object_classes')),
                ))
            }
            collate_fn = collate_utils.get_collateFn(collate_setting)
        else:
            collate_setting = {
                'index': {'type': 'stack', 'tensor': False},
                'type': {'type': 'stack', 'tensor': False},
                'question': {'type': 'pad-stack', 'pad_value': info.protocol['words', '<NULL>'],
                             'tensor': True},
                'program': {'type': 'pad-stack',
                            'pad_fn': lambda x, y: (info.protocol['operations', '<NULL>'] if y==0
                                                    else info.protocol['concepts', '<NULL>']),
                            'tensor': True},
                'answer': {'type': 'stack', 'tensor': False},
                'scene_plain': {'type': 'stack', 'tensor': False},
                'scene': {'type': 'list', 'tensor': True},
                'image': {'type': 'stack', 'tensor': True},
                'objects': {'type': 'concat', 'axis': 0, 'tensor': True},
                'object_lengths': {'type': 'stack', 'tensor': True},
                'object_classes': {'type': 'list', 'tensor': True},
                'filter_fn': ('or', (
                    'classification' not in args.subtasks,
                    ('equal_in_length',
                     ('object_lengths', 'scene_plain', 'scene', 'object_classes')),
                ))
            }
            collate_fn = collate_utils.get_collateFn(collate_setting)

        if not isinstance(data, list):
            data = [data]
        return collate_fn(data)

