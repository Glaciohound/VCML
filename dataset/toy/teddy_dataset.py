from dataset.tools import protocol
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
args = sys.args
info = sys.info

class ToyDataset:

    def __init__(self):
        pass

    @classmethod
    def build_visual_dataset(cls):
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
                categories = np.random.choice(list(vocabulary.records),
                                              args.toy_attributesPobject,
                                              replace=False)
                obj = {cat: np.random.choice(vocabulary[cat], 1)[0]
                       for cat in categories}
                scene['objects'].update({str(len(scene['objects'])): obj})
            cls.sceneGraphs[str(i)] = scene

    @classmethod
    def load_visual_dataset(cls, dataset):
        cls.visual_dataset = dataset

    @classmethod
    def build_question_dataset(cls):
        cls.questions = {}

        if 'synonym' in args.subtask or 'query_isinstance' in args.subtask:
            all_concepts = info.vocabulary['total']
            task_concepts = {'val': np.random.choice(all_concepts,
                                                     int(len(all_concepts)*args.generalization_ratio),
                                                     replace=False)}
            task_concepts['train'] = np.setdiff1d(all_concepts, task_concepts['val'])
            task_concepts['total'] = all_concepts
            args.task_concepts['random'] = task_concepts
        elif 'filter_isinstance' in args.subtask:
            all_concepts = info.vocabulary['total']
            task_concepts = {'val': np.random.choice(info.vocabulary['color']
                                                     if 'color' in info.vocabulary.records
                                                     else all_concepts,
                                                     int(len(all_concepts)*args.generalization_ratio),
                                                     replace=False)}
            task_concepts['train'] = np.setdiff1d(all_concepts, task_concepts['val'])
            task_concepts['total'] = all_concepts
            args.task_concepts['random'] = task_concepts
        elif 'antonym' in args.subtask:
            binary_cats = [cat for cat, items in info.vocabulary.records.items() if len(items) == 2]
            val_cats = np.random.choice(binary_cats,
                                        int(len(binary_cats)*args.generalization_ratio),
                                        replace=False)
            train_cats = np.setdiff1d(binary_cats, val_cats)
            get_concepts = lambda cats: [concept for cat in cats for concept in info.vocabulary[cat]]
            args.task_concepts['antonym'] = {
                'val': get_concepts(val_cats),
                'train': get_concepts(train_cats),
            }

        selected_ids = np.random.choice(list(cls.visual_dataset.keys()), args.max_sizeDataset)
        print('building question dataset')
        for scene_id in tqdm(selected_ids):
            scene = cls.visual_dataset[scene_id]['scene']
            if 'objects' not in scene:
                continue
            split = scene['split'] if not args.no_validation else 'train'
            for j in range(args.questionsPimage):
                if not args.conceptual or \
                        np.random.random() > args.conceptual_question_ratio:
                    if 'exist' in args.subtask:
                        question = cls.exist_question(scene)
                    elif 'filter' in args.subtask:
                        question = cls.filter_question(scene)
                    elif 'query' in args.subtask:
                        question = cls.query_question(scene, split)
                    else:
                        raise Exception('not such task supported as %s' % args.subtask)
                else:
                    if 'synonym' in args.subtask:
                        question = cls.synonym_question(split)
                    elif 'antonym' in args.subtask:
                        question = cls.antonym_question(split)
                    elif 'isinstance' in args.subtask:
                        question = cls.multiselect_isinstance_question(split)
                    else:
                        raise Exception('no such conceptual question type found: %s' % args.subtask)

                if question is not None:
                    question['image_id'] = scene_id
                    question['split'] = split
                    cls.questions.update({str(len(cls.questions)): question})

        cls.built = True

    @classmethod
    def exist_question(cls, scene):
        # filter-exist questions
        queried = np.random.choice(info.vocabulary['total'], 1)[0]
        which, answer = cls.filter_objects(scene, queried)

        if args.conceptual:
            queried = cls.rename(queried, queried+'_syn', 0.5)

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
        cat_1, cat_2 = np.random.choice(list(info.vocabulary.records), 2, replace=False)
        queried_1 = np.random.choice(info.vocabulary[cat_1], 1)[0]
        queried_2 = np.random.choice(info.vocabulary[cat_2], 1)[0]
        which_1, answer_1 = cls.filter_objects(scene, queried_1)
        which_2, answer_2 = cls.filter_objects(scene, queried_2)
        which_2 = list(set(which_1).intersection(set(which_2)))
        if which_2 == []:
            which_2 = ['-']
        answer = 'yes' if which_2 != ['-'] else 'no'

        if args.conceptual and 'synonym' in args.subtask:
            queried_1 = cls.rename(queried_1, queried_1+'_syn', 0.5)
            queried_2 = cls.rename(queried_2, queried_2+'_syn', 0.5)

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
    def query_question(cls, scene, split):
        found = False
        obj_id = np.random.choice(list(scene['objects']), 1)[0]
        obj = scene['objects'][obj_id]
        for i in range(10):
            cat_1, cat_2, cat_3, cat_4 = np.random.choice(list(info.vocabulary.records), 4, replace=False)
            attr_1, attr_2, attr_3 = obj[cat_1], obj[cat_2], obj[cat_3]
            which_1 = cls.filter_objects(scene, attr_1)[0]
            which_2 = cls.filter_objects(scene, attr_2)[0]
            which_2 = list(set(which_1).intersection(set(which_2)))
            which_3 = cls.filter_objects(scene, attr_3)[0]
            which_3 = list(set(which_2).intersection(set(which_3)))
            answer = obj[cat_4]
            if len(set(which_1).intersection(set(which_2)).intersection(set(which_3))) == 1 and \
                    (args.subtask != 'query_isinstance_rev' or answer in args.task_concepts['random'][split]):
                found = True
                break
            else:
                continue

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
                {'operation': 'query', 'argument': cat_4, 'dependencies': [2]}
            ],
            'answer': answer,
            'type': 'filter-filter-filter-query',
        }
        return question

    @classmethod
    def synonym_question(cls, split):
        task_concepts = args.task_concepts['random'][split]
        queried_1 = np.random.choice(task_concepts, 1)[0]
        if len(task_concepts) == 1 or\
                np.random.random() >= 0.5 * (1 + 1/(len(task_concepts)-1)):
            queried_2 = queried_1
            answer = 'yes'
        else:
            queried_2 = np.random.choice(task_concepts, 1)[0]
            answer = 'yes' if queried_2 == queried_1 else 'no'

        if args.conceptual_question_ratio < 1:
            queried_1 = cls.rename(queried_1, queried_1+'_syn', 0.5)
            queried_2 = cls.rename(queried_2, queried_2+'_syn', 0.5)

        question = {
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
    def antonym_question(cls, split):
        task_concepts = args.task_concepts['antonym'][split]
        queried_1 = np.random.choice(task_concepts, 1)[0]
        queried_1_anto = info.vocabulary[info.vocabulary.belongs_to(queried_1)]
        queried_1_anto = queried_1_anto[1-queried_1_anto.index(queried_1)]

        if len(task_concepts) == 1 or np.random.random() >= 0.5 * (1 + 1/(len(task_concepts)-1)):
            queried_2 = queried_1_anto
            answer = 'yes'
        else:
            queried_2 = np.random.choice(task_concepts, 1)[0]
            answer = 'yes' if queried_2 == queried_1_anto else 'no'

        question = {
            'question': 'Is {} a antonym of {} ?'.format(queried_2, queried_1),
            'semantic': [
                {'operation': 'select_concept', 'argument': format(queried_1),
                 'dependencies': []},
                {'operation': 'antonym', 'argument': format(queried_2),
                 'dependencies': [1]}
            ],
            'answer': answer,
            'type': 'antonym',
        }
        return question

    @classmethod
    def binary_isinstance_question(cls, split):
        if args.subtask == 'query_isinstance_rev':
            split = 'total'
        task_concepts = args.task_concepts['random'][split]
        queried_1 = np.random.choice(task_concepts, 1)[0]
        queried_1_cat = info.vocabulary.belongs_to(queried_1)

        if np.random.random() >= 0.5 * (1 + 1/(len(info.vocabulary.records)-1)):
            queried_2 = queried_1_cat
            answer = 'yes'
        else:
            queried_2 = np.random.choice(list(info.vocabulary.records), 1)[0]
            answer = 'yes' if queried_2 == queried_1_cat else 'no'

        question ={
            'question': 'Is {} an instance of {} ?'.format(queried_1, queried_2),
            'semantic': [
                {'operation': 'select_concept', 'argument': format(queried_1),
                 'dependencies': []},
                {'operation': 'isinstance', 'argument': format(queried_2),
                 'dependencies': [1]}
            ],
            'answer': answer,
            'type': 'isinstance',
        }
        return question

    @classmethod
    def multiselect_isinstance_question(cls, split):
        if args.subtask == 'query_isinstance_rev':
            split = 'total'
        task_concepts = args.task_concepts['random'][split]
        if split == 'train':
            queried_1_cat = np.random.choice(list(info.vocabulary.records), 1)[0]
            queried_1 = np.random.choice(list(set(task_concepts).intersection(
                set(info.vocabulary[queried_1_cat]))), 1)[0]
        else:
            queried_1 = np.random.choice(task_concepts, 1)[0]
            queried_1_cat = info.vocabulary.belongs_to(queried_1)

        question = {
            'question': 'What is {} an instance of?'.format(queried_1),
            'semantic': [
                {'operation': 'select_concept', 'argument': format(queried_1),
                 'dependencies': []},
                {'operation': 'isinstance', 'argument': '<NULL>',
                 'dependencies': [1]}
            ],
            'answer': queried_1_cat,
            'type': 'isinstance',
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
                if isinstance(at, str) and at == queried:
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
    meta_dataset = ToyDataset
    def __init__(self):
        self.sceneGraphs = ToyDataset.sceneGraphs

    def __getitem__(self, index):
        return self.sceneGraphs[index]

    def __len__(self):
        return len(self.sceneGraphs)

    def items(self):
        return self.sceneGraphs.items()

class ToyQuestionDataset(Dataset):
    meta_dataset = ToyDataset
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
