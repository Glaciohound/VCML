import sys
import os

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Plain', color_scheme='Linux', call_pdb=1)
if os.getcwd().endswith('scripts'):
    sys.path.append('../')

from config import Config, Info

args = Config()
info = Info()

import matplotlib; matplotlib.use('Agg')

from IPython import embed

import torch
import torch.nn.functional as F

from dataset import visual_dataset, question_dataset
from dataset.tools import protocol, dataset_scheduler
from model.relation_model import RelationModel
from model.uEmbedding_model import UEmbedding
from model.hEmbedding_model import HEmbedding
from model.classification import Classification
from utils.recording import Recording
from utils.common import tqdm, contains, equal_ratio, equal_items, recall
from utils.basic import init_seed, save_log
import numpy as np

def accuracy_by_type(outputs, data):
    _output = {}
    def binary_right(x, y):
        return x * y + (1 - x) * (1 - y)

    for i in range(len(outputs)):
        _type = data['type'][i]
        if not _type in _output:
            _output[_type] = []
        if _type == 'classification':
            right = binary_right((outputs[i].cpu()>0).long(), data['object_classes'][i].long()).sum()
            total = (outputs[i] * 0 + 1).sum()
            _output[_type].append(right / total)
        else:
            _output[_type].append((outputs[i].argmax() == data['answer'][i]).cpu().float().sum())

    return _output

def run_batch(data):
    losses, outputs = info.model(data)
    conceptual = np.array(list(map(lambda x: contains(x, args.conceptual_subtasks),
                                   data['type'].tolist()))).astype(float)
    weight = info.to(torch.Tensor(conceptual * args.conceptual_weight + 1 - conceptual))
    loss = (weight * losses).sum()
    accuracy = accuracy_by_type(outputs, data)

    total_accuracy = []
    for v in accuracy.values():
        total_accuracy = total_accuracy + v
    total_accuracy = torch.stack(total_accuracy).mean()
    log = {'loss': loss, 'accuracy': total_accuracy}

    for k, v in accuracy.items():
        log[k + '_accuracy'] = torch.stack(v).mean()

    return log


def train_epoch():
    info.model.train()
    info.pbars[0].write('epoch {}'.format(info.epoch))
    recording = info.train_recording
    for data in tqdm(info.train):
        info.optimizer.zero_grad()
        recording.update(run_batch(data))
        recording.previous['loss'].backward(retain_graph=False)
        info.optimizer.step()

        info.pbars[1].set_description(str(recording)[:70])

    info.pbars[0].write('[TRAIN]\t' + recording.strings()[0][:100])


def val_epoch():
    info.model.eval()
    recording = info.val_recording
    with torch.no_grad():
        for data in tqdm(info.val):
            recording.update(run_batch(data))

            info.pbars[1].set_description(str(recording)[:70])

    info.pbars[0].write('[VAL]\t%s' % recording.strings()[0][:100])

def init():
    if args.random_seed:
        init_seed(args.random_seed)
    info.model.init()
    info.to(info.model)
    info.train_recording = Recording(name='train', mode='decaying')
    info.val_recording = Recording(name='val', mode='average')
    info.dataset_scheduler = dataset_scheduler.DatasetScheduler()

def run():
    for info.epoch in tqdm(range(1, args.epochs + 1)):

        if args.visualize_dir and not args.silent:
            if not isinstance(info.model, Classification):
                info.model.visualize_embedding(args.visualize_relation)
                info.model.visualize_logit()
            info.train_recording.visualize()
            info.val_recording.visualize()

        train_epoch()
        if not args.no_validation:
            val_epoch()

        info.scheduler.step(info.train_recording.data['loss'])
        info.dataset_scheduler.step(info.train_recording.data['accuracy'])

        info.val_recording.clear()

        if not args.silent:
            info.model.save(args.name)
            save_log(os.path.join(args.log_dir, args.name+'.pkl'),
                    info.val_recording.history,
                    args.__dict__)

def main():
    info.embed = embed
    info.protocol = protocol.Protocol(args.allow_output_protocol, args.protocol_file)
    info.plt = plt
    info.np = np

    if args.random_seed:
        init_seed(args.random_seed)

    info.dataset_all = dataset_scheduler.\
        build_incremental_training_datasets(visual_dataset.Dataset,
                                            question_dataset.Dataset)
    args.names = info.vocabulary.concepts

    if args.model == 'relation_model':
        info.model = RelationModel()
    elif args.model == 'u_embedding':
        info.model = UEmbedding()
    elif args.model in ['h_embedding_mul', 'h_embedding_add',
                        'h_embedding_add2']:
        info.model = HEmbedding()

    args.print()
    info.pbars = []
    info.log = {}

    init()
    run()
    embed()

if __name__ == '__main__':
    main()
