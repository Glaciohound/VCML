import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Plain',
                                     color_scheme='Linux', call_pdb=1)
import os
if os.getcwd().endswith('scripts'):
    sys.path.append('../')
from config import Config, Info
info = Info()
args = Config()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from IPython import embed
import torch
from tqdm import tqdm
import torch.nn.functional as F
from dataset import visual_dataset, question_dataset, dataloader
from model.relation_model import RelationModel
from model.uEmbedding_model import UEmbedding
from model.hEmbedding_model import HEmbedding
from utils.recording import Recording
from utils.embedding import visualize_tb as vistb
import numpy as np

def run_batch(data):
    output, target, history = info.model(data)
    output = output.to(info.device)
    losses = info.loss_fn(output, target, reduction='none')
    is_bool = np.array(list(map(args.bool_question, data.type.tolist()))).astype(float)
    weight = torch.Tensor(is_bool + args.non_bool_weight * (1 - is_bool)).to(info.device)
    loss = (weight * losses).sum()
    answers = output.argmax(1).cpu().detach().numpy()
    accuracy = (answers == data.answer).astype(float).mean()
    #yes = (answers == info.protocol['concepts', 'yes']).astype(float).mean()
    #no = (answers == info.protocol['concepts', 'no']).astype(float).mean()
    output = {'loss': loss, 'accuracy': accuracy,
              #'yes': yes, 'no': no,
              #'isinstance_length': info.model.get_embedding('isinstance', True).pow(2).sum().sqrt(),
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


def train_epoch():
    info.model.train()
    info.pbars[0].write('epoch {}'.format(info.epoch))
    recording = info.train_recording
    for data in info.tqdm_fn(info.train):
        info.optimizer.zero_grad()
        recording.update(run_batch(data))
        recording.previous['loss'].backward(retain_graph=True)
        info.optimizer.step()

        info.pbars[1].set_description(str(recording)[:70])
    info.pbars.pop(1)

    info.pbars[0].write('[TRAIN]\t' + recording.strings()[0][:100])


def val_epoch():
    info.model.eval()
    recording = info.val_recording
    with torch.no_grad():
        for data in info.tqdm_fn(info.val):
            recording.update(run_batch(data))
    info.pbars.pop(1)

    info.pbars[0].write('[VAL]\t%s' % recording.strings()[0][:100])

def init():
    info.model.init()
    if args.use_cuda:
        info.model.to(info.device)
    info.train_recording = Recording(name='train', mode='decaying')
    info.val_recording = Recording(name='val', mode='average')

def run():
    pbar = info.tqdm_fn(args.epochs)
    for info.epoch in range(1, args.epochs + 1):
        info.model.visualize_embedding(None if not args.conceptual
                                       else args.subtask.split('_')[1],
                                       normalizing=True)
        train_epoch()
        if not args.no_validation:
            val_epoch()
        #info.scheduler.step(info.val_recording.values['loss']
        #                    if not args.no_validation
        #                    else info.train_recording.values['loss'])
        info.scheduler.step(info.train_recording.values['loss'])
        info.val_recording.clear()
        pbar.update()
        info.model.save(args.name)
        info.train_recording.visualize()
        info.val_recording.visualize()
    pbar.close()

def main():
    info.compact_data = True
    info.embed = embed
    info.vistb = vistb.visualize_word_embedding_tb
    info.to_numpy = lambda x: x.cpu().detach().numpy()
    info.normalize = lambda x: F.normalize(x, dim=-1)\
        if isinstance(x, torch.Tensor)\
        else F.normalize(torch.Tensor(x), dim=-1).numpy()
    if args.no_random:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
    info.plt = plt
    info.np = np

    if args.model == 'relation_model':
        info.model = RelationModel()
    elif args.model == 'u_embedding':
        info.model = UEmbedding()
    else:
        info.model = HEmbedding()
    info.loss_fn = F.nll_loss if args.loss == 'cross_entropy' else\
        F.mse_loss if args.loss == 'mse' else\
        F.binary_cross_entropy

    info.visual_dataset = visual_dataset.Dataset()
    info.train, info.val, info.test =\
        dataloader.get_dataloaders(question_dataset.Dataset)
    info.visual_dataset.to_mode(
        'recognized' if args.task.endswith('rc') else
        'pretrained' if args.task.endswith('pt') else
        'encoded_sceneGraphs'
    )
    args.names = info.question_dataset.get_names()
    args.print()

    info.pbars = []
    info.log = {}
    info.tqdm_fn = lambda x: (info.pbars.append(tqdm(total=x)
                                                if isinstance(x, int)
                                                else tqdm(x)),
                              info.pbars[-1])[-1]
    init()
    if args.ipython:
        embed()
    run()
    if not args.ipython:
        embed()

if __name__ == '__main__':
    main()
