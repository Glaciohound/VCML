from torch.utils.data import DataLoader
import sys
args = sys.args
info = sys.info


def get_dataloaders(dataset_kit):

    kwargs = {'num_workers': args.num_workers,
              'collate_fn': dataset_kit['question_dataset'].__class__.collate,
              'drop_last': True,
              'pin_memory': True,
              }

    train_loader = DataLoader(
        dataset=dataset_kit['train'],
        batch_size=args.batch_size * args.num_gpus,
        shuffle=not args.no_train_shuffle,
        **kwargs)

    val_loader = DataLoader(
        dataset=dataset_kit['val'],
        batch_size=args.num_gpus,
        shuffle=False,
        **kwargs
    )

    test_loader = DataLoader(
        dataset=dataset_kit['test'],
        batch_size=args.num_gpus,
        shuffle=False,
        **kwargs
    )
    return train_loader, val_loader, test_loader
