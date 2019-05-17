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


def run_batch(data):
    loss, monitors, _ = info.model(data)
    return loss, as_float(monitors)


def train_epoch():
    info.model.train()
    info.pbars[0].write('epoch {}'.format(info.epoch))
    recording = info.train_recording
    for data in tqdm(info.train):
        info.optimizer.zero_grad()
        loss, monitors = run_batch(data)
        recording.update(monitors)
        loss.backward(retain_graph=False)
        info.optimizer.step()
        info.pbars[1].set_description(str(recording)[:70])
    info.pbars[0].write('[TRAIN]\t' + recording.strings()[0][:100])


def val_epoch():
    info.model.eval()
    recording = info.val_recording
    with torch.no_grad():
        for data in tqdm(info.val):
            recording.update(run_batch(data)[1])
            info.pbars[1].set_description(str(recording)[:70])

    info.pbars[0].write('[VAL]\t%s' % recording.strings()[0][:100])


def init():
    info.model.init()
    info.to(info.model)
    logger.critical('Building the recorder.')
    info.train_recording = Recording(name='train', mode='decaying')
    info.val_recording = Recording(name='val', mode='average')
    logger.critical('Building the dataset.')
    info.dataset_scheduler = dataset_scheduler.DatasetScheduler()


def run():
    for info.epoch in tqdm(range(1, args.epochs + 1)):

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
        info.dataset_scheduler.step(info.train_recording.data['accuracy'])
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

    info.dataset_all = dataset_scheduler.build_incremental_training_datasets(visual_dataset.Dataset, question_dataset.Dataset)
    args.names = info.vocabulary.concepts

    logger.critical('Building the model.')
    if args.model in ['h_embedding_mul', 'h_embedding_add', 'h_embedding_add2']:
        from metaconcept.model.hEmbedding_model import HEmbedding
        info.model = HEmbedding()
    elif args.model == 'j_embedding':
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
