from torch.utils.data import DataLoader
import sys
args = sys.args
info = sys.info


def get_dataloaders(dataset):
    train_dataset, val_dataset, test_dataset = dataset.get_datasets()

    kwargs = {'num_workers': args.num_workers,
              'collate_fn': train_dataset.__class__.collate,
              'drop_last': True,
              'pin_memory': True,
              }

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size * args.num_gpus,
        shuffle=not args.no_train_shuffle,
        **kwargs)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.num_gpus,
        shuffle=False,
        **kwargs
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.num_gpus,
        shuffle=False,
        **kwargs
    )
    return train_loader, val_loader, test_loader
