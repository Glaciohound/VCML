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
        args.toy_categories = min(args.toy_categories, args.toy_attributes)
        cls.classes = ['obj_{}'.format(i)
                      for i in range(args.toy_names)]
        cls.attributes = ['attr_{}'.format(i)
                          for i in range(args.toy_attributes)]
        cls.categories = {'cate_{}'.format(i): []
                          for i in range(args.toy_categories)}
        attributes_ids = list(range(args.toy_attributes))
        for cat in cls.categories:
            entries = np.random.choice(attributes_ids, args.toy_attributes//args.toy_categories, replace=False)
            cls.categories[cat] = [cls.attributes[x] for x in entries]
            for x in entries:
                attributes_ids.remove(x)
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
            scene.gt_attributes = -np.ones((args.toy_objects, args.toy_attributesPobject), dtype=int)
            for j in range(args.toy_objects):
                categories = np.random.choice(list(cls.categories), args.toy_attributesPobject,
                                              replace=False)
                for k in range(args.toy_attributesPobject):
                    scene.gt_attributes[j, k] = cls.attributes.index(np.random.choice(cls.categories[categories[k]], 1)[0])
            scene.id = name

            if args.toy_mode == 'exist':
                question = cls.exist_question(scene)
            elif args.toy_mode == 'filter':
                question = cls.filter_question(scene)
            elif args.toy_mode == 'query':
                question = cls.query_question(scene)
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
        if np.random.randint(2) == 1 or\
                args.toy_names < 3*args.toy_objects:
            obj = np.random.randint(args.toy_names)
        else:
            obj = scene.gt_classes[np.random.randint(args.toy_objects)]
        answer = 'yes' if obj in scene.gt_classes.tolist() else 'no'
        which = str(scene.gt_classes.tolist().index(obj))\
            if answer == 'yes' else '-'
        cls_name = cls.classes[obj]
        question = {
            'question': 'Are there any %s in the image?' % cls_name,
            'semantic': [
                {'operation': 'select', 'argument': '{0} ({1})'.format(cls_name, which),
                 'dependencies': []},
                {'operation': 'exist', 'argument': '?',
                 'dependencies': [0]}
            ],
            'answer': answer,
            'imageId': scene.id
        }
        return question

    @classmethod
    def filter_question(cls, scene):
        args = cls.args
        obj = np.random.randint(args.toy_names)
        attr = np.random.randint(args.toy_attributes)
        answer = 'no'
        which = '-'
        for i in range(scene.gt_classes.shape[0]):
            if scene.gt_classes[i] == obj:
                which = i
                if attr in scene.gt_attributes[i]:
                    answer = 'yes'
        cls_name = cls.classes[obj]
        attr_name = cls.attributes[attr]
        question = {
            'question': 'Are there any %s %s in the image?' % (attr_name, cls_name),
            'semantic': [
                {'operation': 'select', 'argument': '{0} ({1})'.format(cls_name, which),
                 'dependencies': []},
                {'operation': 'filter', 'argument': attr_name,
                 'dependencies': [0]},
                {'operation': 'exist', 'argument': '?',
                 'dependencies': [1]}
            ],
            'answer': answer,
            'imageId': scene.id
        }
        return question

    @classmethod
    def query_question(cls, scene):
        args = cls.args
        which = np.random.randint(args.toy_objects)
        cls_name = cls.classes[scene.gt_classes[which]]
        for category, items in cls.categories.items():
            answer = set(items).intersection(set(map(lambda x: cls.attributes[x], scene.gt_attributes[which].tolist())))
            if len(answer) > 1:
                raise Exception('more than one attribute in category')
            elif len(answer) == 1:
                answer = answer.pop()
                break
        if len(answer) == 0:
            raise Exception('no attributes')
        question = {
            'question': 'What is the %s of %s in the image' % (category, cls_name),
            'semantic': [
                {'operation': 'select', 'argument': '{0} ({1})'.format(cls_name, which),
                 'dependencies': []},
                {'operation': 'query', 'argument': category,
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
