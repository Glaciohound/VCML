import torch
import json
import numpy as np
from dataset.tools import program_utils, question_utils, protocol
from dataset.toy import teddy_dataset
import sys
args = sys.args
info = sys.info
from dataset.tools import collate_utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    @classmethod
    def init(cls):
        if not hasattr(info, 'compact_data'):
            info.compact_data = False
        cls.program_utils = program_utils
        info.protocol = protocol.Protocol(args.allow_output_protocol, args.protocol_file)
        cls.load_questions()
        info.question_dataset = cls()

        collate_setting = {
            'index': {'type': 'stack', 'tensor': False},
            'type': {'type': 'stack', 'tensor': False},
            'question': {'type': 'pad-stack', 'pad_value': info.protocol['words', '<NULL>'],
                         'tensor': True},
            'program': {'type': 'pad-stack',
                        'pad_fn': lambda x, y: (protocol['operations', '<NULL>'] if y==0
                                                else protocol['concepts', '<NULL>']),
                        'tensor': True},
            'answer': {'type': 'stack', 'tensor': False},
            'scene': {'type': 'stack', 'tensor': True},
            'image': {'type': 'stack', 'tensor': True},
            'objects': {'type': 'concat', 'axis': 0, 'tensor': True},
            'objects_length': {'type': 'stack', 'tensor': True}
        }
        cls.collate_fn = collate_utils.get_collateFn(collate_setting)

    def __getitem__(self, index_):
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
            'question': question['question'] if not info.compact_data else question_encoded,
            'program': program if not info.compact_data else program_encoded,
            'answer': question['answer'] if not info.compact_data else answer_encoded,
        }
        entry.update(scene)
        return entry

    @classmethod
    def load_questions(cls):

        if args.group in ['toy', 'clevr']:
            if args.task.startswith('clevr'):
                teddy_dataset.ToyDataset.load_visual_dataset(info.visual_dataset)
            teddy_dataset.ToyDataset.build_question_dataset()
            cls.questions = teddy_dataset.ToyQuestionDataset()

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
            cls.questions = list(questions.values())

        cls.split_indexes = {key: [q_id for q_id, q in cls.questions.items()
                                    if q['split']==key]
                                for key in ['train', 'test', 'val']}
        cls.types = set(q['type'] for q in cls.questions.values()
                        if 'type' in q)

    def to_split(self, split):
        self.split = split
        self.index = self.split_indexes[split]
        return self

    @classmethod
    def get_datasets(cls):
        cls.init()
        train, val, test = [cls().to_split(s)
                            for s in ['train', 'val', 'test']]
        for dataset in [train, val, test]:
            for i in range(len(dataset)):
                dataset[i]
        return train, val, test

    def __len__(self):
        return min(len(self.index), args.max_sizeDataset)

    def assertion_checks(self, entry):
        pass

    @classmethod
    def collate(cls, data):
        if not isinstance(data, list):
            data = [data]
        if info.compact_data:
            return cls.collate_fn(data)
        else:
            return data

    def get_names(self):
        names = []
        for cat, attributes in info.vocabulary.records_.items():
            if cat != 'total':
                for y in attributes:
                    for x in info.protocol['concepts']:
                        if y in x and x not in names:
                            names.append(x)
        return names

    def get_data(self, index):
        return self.collate(self[index])
