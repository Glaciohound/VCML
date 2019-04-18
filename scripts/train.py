import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)
import os
if os.getcwd().endswith('scripts'):
    sys.path.append('../')
from IPython import embed
import torch
from tqdm import tqdm
import torch.nn.functional as F
from config import Config
from dataset import visual_dataset, question_dataset, dataloader
from argparse import Namespace
from model.relation_model import RelationModel
from model.uEmbedding_model import UEmbedding
from model.hEmbedding_model import HEmbedding
from time import time
from utils.recording import Recording
from utils.embedding import visualize_tb as vistb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def run_batch(args, info, data):
    output, history = info.model(data)
    output = output.to(info.device)
    loss = info.loss_fn(output, torch.LongTensor(data.answer).to(info.device),
                        reduction='mean')
    answers = output.argmax(1).cpu().detach().numpy()
    accuracy = (answers == data.answer).astype(float).mean()
    yes = (answers == info.protocol['concepts', 'yes']).astype(float).mean()
    no = (answers == info.protocol['concepts', 'no']).astype(float).mean()
    output = {'loss': loss, 'accuracy': accuracy,
              'yes': yes, 'no': no,
              }
    if accuracy != 1:
        info.log['flawed'] = data

    if args.conceptual:
        yes_items = (answers == info.protocol['concepts', 'yes']).astype(float)
        no_items = (answers == info.protocol['concepts', 'no']).astype(float)
        right_items = (answers == data.answer).astype(float)
        types = info.question_dataset.types
        for t in types:
            type_items = (data.type == t).astype(float)
            if type_items.sum() > 0:
                for select_items, select_name in\
                        ((yes_items, 'yes'), (no_items, 'no'), (right_items, 'right')):
                    output['{}_{}'.format(t, select_name)] = (type_items * select_items).sum() / type_items.sum()

    return output


def train_epoch(args, info):
    info.model.train()
    info.pbar.write('epoch {}'.format(info.epoch))
    recording = info.train_recording
    for data in tqdm(info.train):
        info.optimizer.zero_grad()
        recording.update(run_batch(args, info, data))
        recording.previous['loss'].backward(retain_graph=True)
        info.optimizer.step()

        info.pbar.set_description(str(recording)[:70])

    info.pbar.write('[TRAIN]\t' + recording.strings()[0][:100])


def val_epoch(args, info):
    info.model.eval()
    recording = info.val_recording
    with torch.no_grad():
        for data in tqdm(info.val):
            recording.update(run_batch(args, info, data))

    info.pbar.write('[VAL]\t%s' % recording.strings()[0][:100])

def init(args, info):
    info.model.init()
    if args.use_cuda:
        info.model.to(info.device)
    info.train_recording = Recording(args, info, name='train', mode='decaying')
    info.val_recording = Recording(args, info, name='val', mode='average')

def run(args, info):
    init(args, info)
    if args.ipython:
        info.embed()
    for info.epoch in range(1, args.epochs + 1):
        info.model.visualize_embedding('synonym', normalizing=True)
        train_epoch(args, info)
        if not args.no_validation:
            val_epoch(args, info)
        info.scheduler.step(info.val_recording.values['loss']
                            if not args.no_validation
                            else info.train_recording.values['loss'])
        info.val_recording.clear()
        info.pbar.update()
    info.pbar.close()
    embed()

def main():
    info = Namespace()
    info.compact_data = True
    info.embed = embed
    info.vistb = vistb.visualize_word_embedding_tb
    info.plt = plt
    info.np = np
    args = Config(info)

    if args.model == 'relation_model':
        info.model = RelationModel(args, info)
    elif args.model == 'u_embedding':
        info.model = UEmbedding(args, info)
    else:
        info.model = HEmbedding(args, info)

    info.visual_dataset = visual_dataset.Dataset(args, info)
    info.train, info.val, info.test =\
        dataloader.get_dataloaders(args, question_dataset.Dataset, info)
    info.visual_dataset.to_mode(
        'recognized' if args.task.endswith('rc') else
        'pretrained' if args.task.endswith('pt') else
        'encoded_sceneGraphs'
    )
    args.names = info.question_dataset.get_names()

    info.loss_fn = F.nll_loss
    info.pbar = tqdm(total=args.epochs)
    info.ipython = False
    info.timestamp = [('start', time(), 0)]
    info.log = {}

    args.print()
    run(args, info)


if __name__ == '__main__':
    main()
