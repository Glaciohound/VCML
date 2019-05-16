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
        self.register_protocol()

    @classmethod
    def init(cls):
        if cls.inited:
            return
        cls.inited = True
        cls.program_utils = program_utils

        cls.collate_setting = {
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
        cls.collate_fn = collate_utils.get_collateFn(cls.collate_setting)

    def __getitem__(self, index):
        try:
            return self.__getitem_inner__(index)
        except Exception as exc:
            print(exc)
            raise exc

    def __getitem_inner__(self, index_):
        if isinstance(index_, str):
            index_ = self.index.index(index_)
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
        if args.group in ['toy', 'clevr']:
            self.questions = teddy_dataset.ToyDataset.build_question_dataset(visual_dataset, config)
        elif args.group == 'gqa':
            with open(args.questions_json, 'r') as f:
                questions = json.load(f)
            filtered = []
            for k, q in questions.items():
                if question_utils.filter_questions(q, args.question_filter):
                    filtered.append(k)
            questions = {k: questions[k] for k in filtered}
            for q_id, question in questions.items():
                question['id'] = q_id
            self.questions = list(questions.values())

        self.split_indexes = {key: [q_id for q_id, q in self.questions.items()
                                    if q['split']==key]
                                for key in ['train', 'test', 'val']}
        self.split_indexes['total'] = list(self.questions.keys())
        self.types = set(q['type'] for q in self.questions.values()
                        if 'type' in q)
        self.answers = set(q['answer'] for q in self.questions.values())

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

    def register_protocol(self):
        mode = self.mode
        self.mode == 'encoded'
        for i in range(len(self)):
            self[i]
        self.mode = mode

    def __len__(self):
        return len(self.index)

    @property
    def index(self):
        return self.split_indexes[self.split]

    def assertion_checks(self, entry):
        pass

    @classmethod
    def collate(cls, data):
        if not isinstance(data, list):
            data = [data]

        return cls.collate_fn(data)
