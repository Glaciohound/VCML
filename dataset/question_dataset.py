import torch
import json
import numpy as np
from argparse import Namespace
from dataset.tools import program_utils, question_utils, protocol
from dataset.toy import teddy_dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, info=None):
        if not hasattr(info, 'compact_data'):
            info.compact_data = False
        Dataset.args = args
        Dataset.info = info
        self.program_utils = program_utils
        if not hasattr(Dataset, 'questions'):
            Dataset.load_questions()
            Dataset.protocol = protocol.Protocol(args, info)
        info.question_dataset = self

    def __getitem__(self, index_):
        info = self.info
        args = self.args
        if isinstance(index_, str):
            index_ = self.index.index(index_)
        index = self.index[index_]

        question = self.questions[index]
        program = program_utils.semantic2list(question['semantic']) if 'semantic' in question else []
        question_encoded = question_utils.encode_question(question['question'], self.protocol, length=args.max_question_length)
        if info.compact_data:
            program += [{'operation': '<NULL>',
                         'argument': '<NULL>'}
                        for i in range(args.max_program_length-len(program))]
        program_encoded = np.array(
            [[self.protocol['operations', op['operation']],
              self.protocol['concepts', op['argument']]]
             for i, op in enumerate(program)
             if i < args.max_program_length]
        )
        scene = info.visual_dataset[question['imageId']] if hasattr(info, 'visual_dataset') else None
        answer = question['answer']
        answer_encoded = self.protocol['concepts', answer]

        entry = Namespace()
        entry.__dict__.update({
            'index': index,
            'question': question['question'] if not info.compact_data else question_encoded,
            'scene': scene,
            'program': program if not info.compact_data else program_encoded,
            'answer': question['answer'] if not info.compact_data else answer_encoded,
        })
        return entry

    @classmethod
    def load_questions(cls):
        args = cls.args
        info = cls.info
        if args.toy:
            cls.questions =\
                teddy_dataset.ToyQuestionDataset(args, info)
            info.visual_dataset =\
                teddy_dataset.ToyVisualDataset(args, info)
            all_indexes = np.arange(args.size_toy)
            split_train = int(args.size_toy * 0.7)
            split_val = int(args.size_toy * 0.9)
            cls.split_indexes = {'train': all_indexes[:split_train],
                                 'val': all_indexes[split_train: split_val],
                                 'test': all_indexes[split_val:]}

        else:
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
            cls.split_indexes = {split: [] for split in
                                 ['train', 'test', 'val', 'challenge']}
            for i, q in enumerate(cls.questions):
                split = q.get('split', 'train')
                cls.split_indexes[split].append(i)

    def to_split(self, split):
        self.split = split
        self.index = self.split_indexes[split]
        return self

    @classmethod
    def get_datasets(cls, args, info):
        train, val, test = [cls(args, info).to_split(s)
                            for s in ['train', 'val', 'test']]
        return train, val, test

    def __len__(self):
        return len(self.index)

    def assertion_checks(self, entry):
        pass

    @classmethod
    def collate(cls, data):
        output = Namespace()
        if cls.info.compact_data:
            output_dict = {
                k: np.array([getattr(x, k) for x in data])
                for k in ('index', 'answer', 'program', 'question', 'scene')
            }
            output.__dict__.update(output_dict)
            return output
        else:
            return data
