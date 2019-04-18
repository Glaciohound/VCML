from dataset.tools import protocol
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class ToyDataset:
    inited = False

    def __init__(self, args, info):
        cls = ToyDataset
        cls.info = info
        cls.args = args
        cls.inited = True

    @classmethod
    def init(cls, args, info):
        if not cls.inited:
            cls.__init__(cls, args, info)

    @classmethod
    def build_visual_dataset(cls):
        args = cls.args
        info = cls.info
        vocabulary = protocol.Protocol(False, '', gather=True, use_special_tokens=False)
        info.vocabulary = vocabulary
        for i in range(args.toy_attributes):
            cat = min((i * args.toy_categories) // (args.toy_attributes),
                      args.toy_categories)
            attr_name = 'attr_%.2d' % i
            vocabulary['cat_{}'.format(cat), attr_name]

        cls.sceneGraphs = {}
        print('building teddy sceneGraphs')
        for i in tqdm(range(args.max_sizeDataset)):
            split =\
                'train' if i < 0.8 * args.max_sizeDataset or args.no_validation else\
                'val' if i < 1 * args.max_sizeDataset else\
                'test'
            scene = {'objects': {}, 'split': split, 'image_id': format(i)}
            for j in range(args.toy_objects):
                categories = np.random.choice(vocabulary.records,
                                              args.toy_attributesPobject,
                                              replace=False)
                obj = {cat: np.random.choice(vocabulary[cat], 1)[0]
                       for cat in categories}
                scene['objects'].update({str(len(scene['objects'])): obj})
            cls.sceneGraphs[str(i)] = scene

    @classmethod
    def load_visual_dataset(cls, sceneGraphs):
        cls.sceneGraphs = sceneGraphs

    @classmethod
    def build_question_dataset(cls):
        args = cls.args
        info = cls.info
        cls.questions = {}

        visual_concepts = info.vocabulary['total']
        cls.visual_concepts = {'all': visual_concepts}
        cls.visual_concepts['val'] = np.random.choice(visual_concepts,
                                                      int(len(visual_concepts)*args.generalization_ratio),
                                                      replace=False)
        cls.visual_concepts['train'] = np.setdiff1d(visual_concepts, cls.visual_concepts['val'])
        args.visual_concepts = cls.visual_concepts

        selected_ids = np.random.choice(list(cls.sceneGraphs.keys()), args.max_sizeDataset)
        print('building question dataset')
        for scene_id in tqdm(selected_ids):
            scene = cls.sceneGraphs[scene_id]
            if 'objects' not in scene:
                continue
            split = scene['split']
            for j in range(args.questionsPimage):
                if not args.conceptual or \
                        np.random.random() > args.conceptual_question_ratio:
                    if 'exist' in args.subtask:
                        question = cls.exist_question(scene)
                    elif 'filter' in args.subtask:
                        question = cls.filter_question(scene)
                    elif 'query' in args.subtask:
                        question = cls.query_question(scene)
                    else:
                        raise Exception('not such task supported as %s' % args.subtask)
                else:
                    if 'synonym' in args.subtask:
                        question = cls.synonym_question(split)
                    else:
                        raise Exception('no conceptual question type found')

                if question is not None:
                    question['image_id'] = scene_id
                    question['split'] = split
                    cls.questions.update({str(len(cls.questions)): question})

        cls.built = True

    @classmethod
    def exist_question(cls, scene):
        # filter-exist questions
        info = cls.info
        args = cls.args
        queried = np.random.choice(info.vocabulary['total'], 1)[0]
        which, answer = cls.filter_objects(scene, queried)

        if args.conceptual:
            queried = cls.rename(queried, queried+args.conceptual_tokens['synonym'], 0.5)

        question = {
            'question': 'Are there any %s objects in the image?' % queried,
            'semantic': [
                {'operation': 'select', 'argument': '{0} ({1})'.format(queried, ', '.join(which)),
                'dependencies': []},
                {'operation': 'exist', 'argument': '?',
                'dependencies': [0]}
            ],
            'answer': answer,
            'type': 'filter-exist',
        }

        return question

    @classmethod
    def filter_question(cls, scene):
        # filter-filter-exist questions
        info = cls.info
        args = cls.args
        cat_1, cat_2 = np.random.choice(info.vocabulary.records, 2, replace=False)
        queried_1 = np.random.choice(info.vocabulary[cat_1], 1)[0]
        queried_2 = np.random.choice(info.vocabulary[cat_2], 1)[0]
        which_1, answer_1 = cls.filter_objects(scene, queried_1)
        which_2, answer_2 = cls.filter_objects(scene, queried_2)
        which_2 = list(set(which_1).intersection(set(which_2)))
        if which_2 == []:
            which_2 = ['-']
        answer = 'yes' if which_2 != ['-'] else 'no'

        if args.conceptual:
            queried_1 = cls.rename(queried_1, queried_1+args.conceptual_tokens['synonym'], 0.5)
            queried_2 = cls.rename(queried_2, queried_2+args.conceptual_tokens['synonym'], 0.5)

        question = {
            'question': 'Are there any %s %s objects in the image?' %
            (queried_1, queried_2),
            'semantic': [
                {'operation': 'select', 'argument': '{0} ({1})'.format(queried_1, ', '.join(which_1)),
                 'dependencies': []},
                {'operation': 'filter', 'argument': '{0} ({1})'.format(queried_2, ', '.join(which_2)),
                 'dependencies': [0]},
                {'operation': 'exist', 'argument': '?',
                 'dependencies': [1]}
            ],
            'answer': answer,
            'type': 'filter-filter-exist',
        }
        return question

    @classmethod
    def query_question(cls, scene):
        info = cls.info
        found = False
        obj_id = np.random.choice(list(scene['objects']), 1)[0]
        obj = scene['objects'][obj_id]
        cat_1, cat_2, cat_3, cat_4 = np.random.choice(info.vocabulary.records, 4, replace=False)
        attr_1, attr_2, attr_3 = obj[cat_1], obj[cat_2], obj[cat_3]
        which_1 = cls.filter_objects(scene, attr_1)[0]
        which_2 = cls.filter_objects(scene, attr_2)[0]
        which_2 = list(set(which_1).intersection(set(which_2)))
        which_3 = cls.filter_objects(scene, attr_3)[0]
        which_3 = list(set(which_2).intersection(set(which_3)))
        answer = obj[cat_4]
        if len(set(which_1).intersection(set(which_2)).intersection(set(which_3))) == 1:
            found = True
        if not found:
            return None

        question = {
            'question': 'What is the %s of the %s, %s, %s objects in the image'
                % (cat_4, attr_1, attr_2, attr_3),
            'semantic': [
                {'operation': 'select',
                 'argument': '{0} ({1})'.format(attr_1, ', '.join(which_1)),
                 'dependencies': []},
                {'operation': 'filter',
                 'argument': '{0} ({1})'.format(attr_2, ', '.join(which_2)),
                 'dependencies': [0]},
                {'operation': 'filter',
                 'argument': '{0} ({1})'.format(attr_3, ', '.join(which_3)),
                 'dependencies': [1]},
                {'operation': 'query', 'argument': cat_4, 'dependencies': [2]},
            ],
            'answer': answer,
            'type': 'filter-filter-filter-query',
        }
        return question

    @classmethod
    def synonym_question(cls, split):
        args = cls.args
        visual_concepts = cls.visual_concepts[split]
        queried_1 = np.random.choice(visual_concepts, 1)[0]
        if len(visual_concepts) == 1 or np.random.random() >= 0.5 * (1 + 1/(len(visual_concepts)-1)):
            queried_2 = queried_1
            answer = 'yes'
        else:
            queried_2 = np.random.choice(visual_concepts, 1)[0]
            answer = 'yes' if queried_2 == queried_1 else 'no'

        if cls.args.conceptual_question_ratio < 1:
            queried_1 = cls.rename(queried_1, queried_1+args.conceptual_tokens['synonym'], 0.5)
            queried_2 = cls.rename(queried_2, queried_2+args.conceptual_tokens['synonym'], 0.5)

        question ={
            'question': 'Is {} a synonym of {} ?'.format(queried_2, queried_1),
            'semantic': [
                {'operation': 'select_concept', 'argument': format(queried_1),
                 'dependencies': []},
                {'operation': 'synonym', 'argument': format(queried_2),
                 'dependencies': [1]}
            ],
            'answer': answer,
            'type': 'synonym',
        }
        return question

    @classmethod
    @property
    def image_ids(cls):
        return list(cls.sceneGraphs)

    @classmethod
    def filter_objects(cls, scene, queried):
        which = ['-']
        answer = 'no'
        for obj_id, obj in scene['objects'].items():
            for at in obj.values():
                if at == queried:
                    if '-' in which:
                        which.remove('-')
                    which.append(obj_id)
                    answer = 'yes'
        return which, answer

    @classmethod
    def rename(cls, x, y, p=0.5):
        if np.random.random() > p:
            return x
        else:
            return y



class ToyVisualDataset(Dataset):
    def __init__(self):
        self.sceneGraphs = ToyDataset.sceneGraphs

    def __getitem__(self, index):
        return self.sceneGraphs[index]

    def __len__(self):
        return len(self.sceneGraphs)

    def items(self):
        return self.sceneGraphs.items()

class ToyQuestionDataset(Dataset):
    def __init__(self):
        self.questions = ToyDataset.questions

    def __getitem__(self, index):
        return self.questions[index]

    def __len__(self):
        return len(self.questions)

    def items(self):
        return self.questions.items()

    def values(self):
        return self.questions.values()
