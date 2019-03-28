import numpy as np
from argparse import Namespace
from torch.utils.data import Dataset

class ToyDataset:
    built = False

    def __init__(self, args, info=None):
        cls = ToyDataset
        cls.info = info
        cls.args = args
        if not cls.built:
            cls.build_dataset()

    @classmethod
    def build_dataset(cls):
        args = cls.args
        cls.names = ['obj_{}'.format(i)
                      for i in range(args.toy_names)]
        cls.attributes = ['attr_{}'.format(i)
                           for i in range(args.toy_attributes)]
        cls.sceneGraphs = {}
        cls.questions = []
        cls.entries = []

        for i in range(args.size_toy):
            name = str(i)
            scene = Namespace()
            scene.gt_classes =\
                np.random.choice(np.arange(args.toy_names),
                                 size=args.toy_objects,
                                 replace=False)
            scene.gt_attributes =\
                np.random.randint(args.toy_attributes,
                                  size=(args.toy_objects, args.toy_attributesPobject))
            scene.id = name

            if args.toy_mode == 'exist':
                question = cls.exist_question(scene)
            else:
                question = cls.exist_question(scene)
            question['id'] = i

            cls.sceneGraphs[name] = scene
            cls.questions.append(question)
            cls.entries.append(name)

        cls.built = True

    @classmethod
    def exist_question(cls, scene):
        args = cls.args
        obj = np.random.randint(args.toy_names)
        answer = 'yes' if obj in scene.gt_classes.tolist() else 'no'
        which = str(scene.gt_classes.tolist().index(obj))\
            if answer == 'yes' else '-'
        name = cls.names[obj]
        question = {
            'question': 'Are there any %s in the image?' % name,
            'semantic': [
                {'operation': 'select', 'argument': '{0} ({1})'.format(name, which),
                 'dependencies': []},
                {'operation': 'exist', 'argument': '?',
                 'dependencies': [0]}
            ],
            'answer': answer,
            'imageId': scene.id
        }
        return question

class ToyVisualDataset(Dataset):
    def __init__(self, args, info=None):
        self.args = args
        self.info = info
        self.dataset = ToyDataset(args, info)

    def __getitem__(self, index_):
        if isinstance(index_, int):
            index = self.dataset.entries[index_]
        else:
            index = index_
        return self.dataset.sceneGraphs[index]

    def __len__(self):
        return self.args.size_toy

class ToyQuestionDataset(Dataset):
    def __init__(self, args, info=None):
        self.args = args
        self.info = info
        self.dataset = ToyDataset(args, info)

    def __getitem__(self, index_):
        if isinstance(index_, str):
            index = self.dataset.entries.index(index_)
        else:
            index = index_
        return self.dataset.questions[index]

    def __len__(self):
        return self.args.size_toy
