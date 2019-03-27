from torch.utils.data import DataLoader


def get_dataloaders(args, dataset, info=None):
    train_dataset, val_dataset, test_dataset = dataset.get_datasets(args, info)

    kwargs = {'num_workers': args.num_workers,
              'collate_fn': train_dataset.__class__.collate,
              'drop_last': True,
              'pin_memory': True,
              }

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size * args.num_gpus,
        shuffle=args.train_shuffle,
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
