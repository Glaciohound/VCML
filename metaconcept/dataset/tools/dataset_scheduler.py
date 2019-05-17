from torch.utils.data.dataset import Dataset
from metaconcept import args, info
from metaconcept.dataset.dataloader import get_dataloaders


def build_incremental_training_datasets(
    visual_dataset_class, question_dataset_class):

    dataset_all = []
    for config in args.incremental_training:
        visual_dataset = visual_dataset_class(config)
        info.visual_dataset = visual_dataset
        question_dataset = question_dataset_class(visual_dataset, config)
        train, val, test = question_dataset_class.get_splits(question_dataset)

        new_dataset = {
            'visual_dataset': visual_dataset,
            'question_dataset': question_dataset,
            'train': train,
            'val': val,
            'test': test,
        }

        (new_dataset['train_loader'],
         new_dataset['val_loader'],
         new_dataset['test_loader']) = get_dataloaders(new_dataset)
        dataset_all.append(new_dataset)
    return dataset_all



class CurriculumLearningDataset(Dataset):
    def __init__(self, proxy_dataset, scene_curriculum):
        self.proxy_dataset = proxy_dataset
        self.scene_curriculum = scene_curriculum
        self.indices = self.gen_indices()

    def gen_indices(self):
        indices = list()
        for idx, qidx in enumerate(self.proxy_dataset.index):
            q = self.proxy_dataset.questions[qidx]
            sid = q['image_id']

            scene = info.visual_dataset.sceneGraphs[sid]
            nr_objects = len(scene['objects'])

            if nr_objects <= self.scene_curriculum:
                indices.append(idx)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.proxy_dataset[self.indices[index]]

    @classmethod
    def init(cls):
        self.proxy_dataset.__class__.init()

    def to_mode(self, *args, **kwargs):
        return self.proxy_dataset.to_mode(*args, **kwargs)

    def assertion_checks(self, entry):
        pass

    def collate(self, data):
        return self.proxy_dataset.collate(data)


def build_curriculum_training_datasets(visual_dataset_cls, question_dataset_cls):
    dataset_all = list()

    assert len(args.incremental_training) == 1 and args.incremental_training[0] == 'full'
    config = args.incremental_training[0]

    visual_dataset = visual_dataset_cls(config)
    info.visual_dataset = visual_dataset
    question_dataset = question_dataset_cls(visual_dataset, config)
    train, val, test = question_dataset_cls.get_splits(question_dataset)

    new_dataset_base = {
        'visual_dataset': visual_dataset,
        'question_dataset': question_dataset,
        'train': train,
        'val': val,
        'test': test,
    }

    scene_curriculum = [3, 4, 5, 6, 7, 8, 9, 10]
    for sc in scene_curriculum:
        new_dataset = new_dataset_base.copy()

        for key in ['train', 'val', 'test']:
            new_dataset[key] = CurriculumLearningDataset(new_dataset[key], sc)

        (new_dataset['train_loader'],
         new_dataset['val_loader'],
         new_dataset['test_loader']) = get_dataloaders(new_dataset)
        dataset_all.append(new_dataset)

    return dataset_all



class DatasetScheduler:
    def __init__(self):
        self.dataset_count = -1
        self.step(1)

    def step(self, acc):
        all_ = info.dataset_all

        if acc >= args.perfect_th and self.dataset_count < len(all_)-1:
            self.dataset_count += 1
            message = 'Dataset is scheduled to %s' % args.incremental_training[self.dataset_count]
            print(message)

            this_dataset = all_[self.dataset_count]

            info.visual_dataset = this_dataset['visual_dataset']
            info.visual_dataset.to_mode(
                'detected' if args.task.endswith('dt') else
                'pretrained' if args.task.endswith('pt') else
                'encoded_sceneGraph'
            )

            info.question_dataset = this_dataset['question_dataset']
            info.train = this_dataset['train_loader']
            info.val = this_dataset['val_loader']
            info.test = this_dataset['test_loader']

            self.to_mode(args.dataset_mode)

    def to_mode(self, mode, visual=False):
        this_dataset = info.dataset_all[self.dataset_count]
        if not visual:
            for dataset_ in [this_dataset['question_dataset'],
                             this_dataset['train'],
                             this_dataset['val'],
                             this_dataset['test']]:
                dataset_.to_mode(mode)
        else:
            this_dataset['visual_dataset'].to_mode(mode)

    @property
    def config(self):
        return args.incremental_training[self.dataset_count]
