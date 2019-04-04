import torch
import json
import numpy as np
from argparse import Namespace
from dataset.tools import program_utils, question_utils, protocol
from dataset.toy import teddy_dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    @classmethod
    def init(cls, args, info):
        if not hasattr(info, 'compact_data'):
            info.compact_data = False
        cls.args = args
        cls.info = info
        cls.program_utils = program_utils
        info.protocol = protocol.Protocol(args.allow_output_protocol, args.protocol_file)
        cls.load_questions()
        info.question_dataset = cls()

    def __getitem__(self, index_):
        if isinstance(index_, str):
            index_ = self.index.index(index_)
        index = self.index[index_]

        question = self.questions[index]
        program = program_utils.semantic2list(question['semantic']) if 'semantic' in question else []
        question_encoded = question_utils.encode_question(question['question'], self.info.protocol)
        program_encoded = np.array(
            [[self.info.protocol['operations', op['operation']],
              self.info.protocol['concepts', op['argument']]]
             for op in program])
        scene = self.info.visual_dataset[question['image_id']] if hasattr(self.info, 'visual_dataset') else None
        answer = question['answer']
        answer_encoded = self.info.protocol['concepts', answer]

        entry = {
            'index': index,
            'question': question['question'] if not self.info.compact_data else question_encoded,
            'scene': scene,
            'program': program if not self.info.compact_data else program_encoded,
            'answer': question['answer'] if not self.info.compact_data else answer_encoded,
        }
        return entry

    @classmethod
    def load_questions(cls):
        args = cls.args
        info = cls.info
        teddy_dataset.ToyDataset.init(args, info)

        if args.task in ['toy', 'clevr_pt']:
            if args.task == 'clevr_pt':
                teddy_dataset.ToyDataset.load_visual_dataset(info.visual_dataset)
            teddy_dataset.ToyDataset.build_question_dataset()
            cls.questions = teddy_dataset.ToyQuestionDataset()

        elif args.task == 'gqa':
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

    def to_split(self, split):
        self.split = split
        self.index = self.split_indexes[split]
        return self

    @classmethod
    def get_datasets(cls, args, info):
        cls.init(args, info)
        train, val, test = [cls().to_split(s)
                            for s in ['train', 'val', 'test']]
        for dataset in [train, val, test]:
            for i in range(len(dataset)):
                dataset[i]
        return train, val, test

    def __len__(self):
        return min(len(self.index), self.args.max_sizeDataset)

    def assertion_checks(self, entry):
        pass

    @classmethod
    def collate(cls, data):
        output = Namespace()
        protocol = cls.info.protocol
        if cls.info.compact_data:
            max_quesiton_length = max([s['question'].shape[0] for s in data])
            max_program_length = max([s['program'].shape[0] for s in data])
            question_template = np.array([protocol['words', '<NULL>']
                                          for i in range(max_quesiton_length)])
            program_template = np.array([[protocol['operations', '<NULL>'],
                                          protocol['concepts', '<NULL>']]
                                         for i in range(max_program_length)])

            for s in data:
                def fill(template, this):
                    copy = template.copy()
                    copy[:this.shape[0]] = this
                    return copy
                s['question'] = fill(question_template, s['question'])
                s['program'] = fill(program_template, s['program'])

            output_dict = {
                k: np.array([x[k] for x in data])
                for k in data[0].keys()
            }
            output.__dict__.update(output_dict)
            return output
        else:
            return data

    @classmethod
    def show_attended(cls, output, num=5):
        def get_concept(x):
            x = int(x)
            if x < len(cls.info.protocol['concepts']):
                return cls.info.protocol['concepts', x]
            elif x < cls.args.max_concepts:
                return 'unknown concept %d' % x

        def get_object(x):
            return 'object_%d' % x

        if output.dim() == 3:
            return [cls.show_attended(output[i])
                    for i in range(output.shape[0])]
        else:
            concept_sort = output[:, :cls.args.max_concepts].argsort(-1, True)
            object_sort = output[:, cls.args.max_concepts:].argsort(-1, True)
            return [{'concepts':
                     {get_concept(concept_sort[i, j]):
                      float(output[i, concept_sort[i, j]])
                      for j in range(num)},
                     'objects':
                     {get_object(object_sort[i, j]):
                      float(output[i, object_sort[i, j]+cls.args.max_concepts])
                      for j in range(min(num, object_sort.shape[1]))}}
                     for i in range(output.shape[0])]
