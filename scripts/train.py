import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Plain',
                                     color_scheme='Linux', call_pdb=1)
import os
if os.getcwd().endswith('scripts'):
    sys.path.append('../')
from config import Config, Info
args = Config()
info = Info()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from IPython import embed
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import visual_dataset, question_dataset, dataloader
from model.relation_model import RelationModel
from model.uEmbedding_model import UEmbedding
from model.hEmbedding_model import HEmbedding
from utils.recording import Recording
from utils.embedding import visualize_tb as vistb
from utils.common import tqdm
import numpy as np

def run_batch(data):
    output, target, history = info.model(data)
    output = info.to(output)
    is_bool = np.array(list(map(args.bool_question, data['type'].tolist()))).astype(float)
    weight = info.to(torch.Tensor(is_bool + args.non_bool_weight * (1 - is_bool)))

    if info.new_torch:
        losses = info.loss_fn(output, target, reduction='none')
        answers = output.argmax(1).cpu().detach().numpy()
        accuracy = (answers == data['answer']).astype(float).mean()
    else:
        losses = info.loss_fn(output, Variable(target), reduce=False)
        weight = Variable(weight)
        answers = output.data.max(1)[1].cpu().numpy()
        accuracy = (answers == data['answer']).astype(float).mean()

    loss = (weight * losses).sum()
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
        right_items = (answers == data['answer']).astype(float)
        types = info.question_dataset.types
        for t in types:
            type_items = (data['type'].numpy() == t).astype(float)
            if type_items.sum() > 0:
                for select_items, select_name in\
                        ((yes_items, 'yes'), (no_items, 'no'), (right_items, 'right')):
                    output['{}_{}'.format(t, select_name)] = (type_items * select_items).sum() / type_items.sum()

    return output


def train_epoch():
    info.model.train()
    info.pbars[0].write('epoch {}'.format(info.epoch))
    recording = info.train_recording
    for data in tqdm(info.train, pbar_list=info.pbars):
        info.optimizer.zero_grad()
        recording.update(run_batch(data))
        #recording.previous['loss'].backward(retain_graph=True)
        recording.previous['loss'].backward(retain_graph=False)
        info.optimizer.step()

        info.pbars[1].set_description(str(recording)[:70])

    info.pbars[1].pop_self()
    info.pbars[0].write('[TRAIN]\t' + recording.strings()[0][:100])


def val_epoch():
    info.model.eval()
    recording = info.val_recording
    with torch.no_grad():
        for data in tqdm(info.val, pbar_list=info.pbars):
            recording.update(run_batch(data))

    info.pbars[1].pop_self()
    info.pbars[0].write('[VAL]\t%s' % recording.strings()[0][:100])

def init():
    info.model.init()
    if args.use_cuda:
        info.to(info.model)
    info.train_recording = Recording(name='train', mode='decaying')
    info.val_recording = Recording(name='val', mode='average')

def run():
    for info.epoch in tqdm(range(1, args.epochs + 1), pbar_list=info.pbars):
        info.model.visualize_embedding(None if not args.conceptual
                                       else args.subtask.split('_')[1],
                                       normalizing=True)
        train_epoch()
        if not args.no_validation:
            val_epoch()
        info.scheduler.step(info.train_recording.values['loss'])
        info.val_recording.clear()
        info.model.save(args.name)
        info.train_recording.visualize()
        info.val_recording.visualize()

def main():
    info.compact_data = True
    info.embed = embed
    info.vistb = vistb.visualize_word_embedding_tb
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
        'detected' if args.task.endswith('dt') else
        'pretrained' if args.task.endswith('pt') else
        'encoded_sceneGraph'
    )
    args.names = info.question_dataset.get_names()
    args.print()

    info.pbars = []
    info.log = {}
    init()
    run()
    embed()

if __name__ == '__main__':
    main()
