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
import torch.optim as optim
from config import Config
from dataset import visual_dataset, question_dataset, dataloader
from argparse import Namespace
from model.relation_model import RelationModel
from time import time
from utils.recording import Recording


def run_batch(args, info, data):
    output, history = info.model(data)
    output = output.to(info.device)
    loss = info.loss_fn(output, torch.LongTensor(data.answer).to(info.device),
                        reduction='mean')
    answers = output.argmax(1).cpu().detach().numpy()
    accuracy = (answers == data.answer).astype(int).sum()\
        / data.answer.shape[0]
    yes = (answers == info.protocol['concepts', 'yes']).astype(int).sum()\
        / data.answer.shape[0]
    no = (answers == info.protocol['concepts', 'no']).astype(int).sum()\
        / data.answer.shape[0]
    return loss, accuracy, yes, no


def train_epoch(args, info):
    info.model.train()
    pbar = tqdm(total=len(info.train))
    recording = Recording(args, info, ['loss', 'accuracy', 'yes', 'no'], mode='decaying')
    for data in info.train:
        info.optimizer.zero_grad()
        recording.update(run_batch(args, info, data))
        recording.previous['loss'].backward(retain_graph=True)
        info.optimizer.step()

        info.pbar.set_description(str(recording))
        pbar.update()

    info.pbar.write('[TRAIN] ' + str(recording))
    pbar.close()


def val_epoch(args, info):
    info.model.eval()
    recording = Recording(args, info, ['loss', 'accuracy', 'yes', 'no'], mode='average')
    for data in tqdm(info.val):
        recording.update(run_batch(args, info, data))

    info.pbar.write('[VAL]' + str(recording))


def main():
    info = Namespace()
    info.compact_data = True
    info.embed = embed
    args = Config(info)
    info.visual_dataset = visual_dataset.Dataset(args, info)
    info.train, info.val, info.test =\
        dataloader.get_dataloaders(args, question_dataset.Dataset, info)

    info.model = RelationModel(args, info)
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        info.model.load_state_dict(ckpt['net_dict'])
    if args.use_cuda:
        info.model.to(info.device)
    info.optimizer = optim.Adam(info.model.parameters(),
                                lr=args.lr)
    info.loss_fn = F.nll_loss
    info.pbar = tqdm(total=args.epochs)
    info.ipython = False
    info.timestamp = [('start', time(), 0)]
    info.log = []
    info.visual_dataset.to_mode(
        'features' if args.task.endswith('pt') else
        'encoded_sceneGraphs'
    )

    for epoch in range(1, args.epochs + 1):
        train_epoch(args, info)
        val_epoch(args, info)
        info.pbar.update()
        if args.name:
            torch.save({
                'epoch': epoch,
                'net_dict': info.model.state_dict(),
                'optimizer': info.optimizer.state_dict(),
            }, '{}.tar'.format(args.name))
    info.pbar.close()
    embed()

if __name__ == '__main__':
    main()
