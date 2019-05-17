from metaconcept import args, info
from metaconcept.dataset.dataloader import get_dataloaders
import pickle


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


class DatasetScheduler:
    def __init__(self):
        self.dataset_count = -1
        self.step(1)

    def step(self, acc):
        all_ = info.dataset_all

        if acc >= args.perfect_th and self.dataset_count < len(all_)-1:
            self.dataset_count += 1
            message = 'Dataset is scheduled to %s' %\
                args.incremental_training[self.dataset_count]
            if len(getattr(info, 'pbars', [])) >= 1:
                info.pbars[0].write(message)
            else:
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

    def save(self, filename):
        _output = [{
            'sceneGraphs': dataset_kit['visual_dataset'].sceneGraphs,
            'questions': dataset_kit['question_dataset'].questions}
            for dataset_kit in info.dataset_all
        ]
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(_output, f)
