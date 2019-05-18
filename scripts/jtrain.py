import sys
import os

from IPython.core import ultratb; sys.excepthook = ultratb.FormattedTB(mode='Plain', color_scheme='Linux', call_pdb=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import embed

import numpy as np
import torch

from jacinle.logging import get_logger; logger = get_logger(__file__)
from jactorch.utils.meta import as_float

from metaconcept.config import Config, Info
from metaconcept.dataset import visual_dataset, question_dataset
from metaconcept.dataset.tools import protocol, dataset_scheduler
from metaconcept.utils.recording import Recording
from metaconcept.utils.common import tqdm, contains, equal_ratio, equal_items, recall
from metaconcept.utils.basic import init_seed, save_log

args = Config()
info = Info(args)


def unnormalize_image(image):
    return image * torch.tensor([0.229, 0.224, 0.225]).to(image)[:, None, None] + torch.tensor([0.485, 0.456, 0.406]).to(image)[:, None, None]

from jaclearn.visualize.box import vis_bboxes
from jaclearn.visualize.plot import plot2pil

def scene_representation(data, batch_idx):
    lengths = np.cumsum(data['object_lengths'])
    start = 0 if batch_idx == 0 else lengths[batch_idx - 1]
    end = start + data['object_lengths'][batch_idx]

    object_bboxes = data['objects'][start:end].cpu().numpy()
    object_classes = data['object_classes'][batch_idx].cpu().numpy()

    object_classes = [
        [info.vocabulary.concepts[j] for j in np.where(object_classes[i])[0]]
        for i in range(object_classes.shape[0])
    ]

    return object_bboxes, object_classes


def run_symbolic_execution(data, batch_idx):
    scene = scene_representation(data, batch_idx)
    program = data['program'][idx]

    stack = list()
    for p in program:
        op = p['operation']
        arg = p['argument']

        if op == 'select':
            pass


def run_batch(data):
    loss, monitors, _ = info.model(data)
    return loss, as_float(monitors)


def train_epoch():
    info.model.train()
    recording = info.train_recording
    for data in tqdm(info.train):
        info.optimizer.zero_grad()
        loss, monitors = run_batch(data)
        recording.update(monitors)
        loss.backward(retain_graph=False)
        info.optimizer.step()
        info.pbars[0].set_description(
            recording.format_string(f'Train[{info.epoch}]: ', max_depth=2)
        )
    logger.critical(
        recording.format_string(f'Train[{info.epoch}]:\n  ', sep='\n  ')
    )


def val_epoch():
    info.model.eval()
    recording = info.val_recording
    with torch.no_grad():
        for data in tqdm(info.val):
            recording.update(run_batch(data)[1])
            info.pbars[0].set_description(
                recording.format_string(f'Val[{info.epoch}]: ', max_depth=2)
            )
    logger.critical(
        recording.format_string(f'Val[{info.epoch}]:\n  ', sep='\n  ')
    )


def init():
    info.model.init()
    info.to(info.model)
    logger.critical('Building the recorder.')
    info.train_recording = Recording(name='train', mode='decaying')
    info.val_recording = Recording(name='val', mode='average')
    logger.critical('Building the dataset.')
    info.dataset_scheduler = dataset_scheduler.DatasetScheduler()


def run():
    for info.epoch in range(1, args.epochs + 1):
        # TODO(Jiayuan Mao @ 05/16): implement.
        # if args.visualize_dir and not args.silent:
        #     if not isinstance(info.model, Classification):
        #         info.model.visualize_embedding(args.visualize_relation)
        #         info.model.visualize_logit()
        #     info.train_recording.visualize()
        #     info.val_recording.visualize()

        train_epoch()
        if not args.no_validation:
            val_epoch()

        info.scheduler.step(info.train_recording.data['loss'])
        info.dataset_scheduler.step(info.train_recording.data['acc'])
        info.val_recording.clear()

        if not args.silent:
            info.model.save(args.name)
            save_log(os.path.join(args.log_dir, args.name+'.pkl'), info.val_recording.history, args.__dict__)


def main():
    info.embed = embed
    info.protocol = protocol.Protocol(args.allow_output_protocol, args.protocol_file)
    info.plt = plt
    info.np = np

    if args.random_seed:
        init_seed(args.random_seed)

    if args.curriculum_training:
        info.dataset_all = dataset_scheduler.build_curriculum_training_datasets(visual_dataset.Dataset, question_dataset.Dataset)
    else:
        info.dataset_all = dataset_scheduler.build_incremental_training_datasets(visual_dataset.Dataset, question_dataset.Dataset)
    args.names = info.vocabulary.concepts

    logger.critical('Building the model.')
    if args.model == 'j_embedding':
        from metaconcept.model.jEmbedding_model import JEmbedding
        info.model = JEmbedding()
    else:
        from metaconcept.model.classification import Classification
        raise ValueError('Unknown model: {}.'.format(args.model))

    args.print()
    info.pbars = []
    info.log = {}

    init()
    if args.embed:
        embed()
    run()
    embed()


if __name__ == '__main__':
    main()

