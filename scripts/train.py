import sys
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

def train_epoch(args, info):
    info.model.train()
    pbar = tqdm(total=len(info.train))
    Yes, No, accuracy, loss = 0, 0, 0, 0
    for data in info.train:
        info.optimizer.zero_grad()
        output, attentions = info.model(data)
        output = output.to(args.device)
        loss = info.loss_fn(output, torch.LongTensor(data.answer).cuda(), reduction='mean')
        answers = output.argmax(1).cpu().detach().numpy()
        accuracy = accuracy * 0.9 +\
            0.1 * (answers == data.answer).astype(int).sum() /\
            data.answer.shape[0]
        Yes = Yes * 0.9 +\
            0.1 * (answers == info.protocol['concepts', 'yes'])\
            .astype(int).sum() / data.answer.shape[0]
        No = No * 0.9 +\
            0.1 *  (answers == info.protocol['concepts', 'no'])\
            .astype(int).sum() / data.answer.shape[0]
        loss.backward(retain_graph=True)
        info.optimizer.step()

        #pbar.set_description('Loss: {:.6f}'.format(loss.item()))
        message = 'Loss: {:.6f}, Acc: {:.4f}, Yes/No: {:.4f}-{:.4f}'\
            .format(loss.item(), accuracy, Yes, No)
        info.pbar.set_description(message)
        pbar.update()
    info.pbar.write(message)
    pbar.close()

def main():
    args = Config()
    info = Namespace()
    info.compact_data = True
    info.visual_dataset = visual_dataset.Dataset(args, info)
    info.train, info.val, info.test = dataloader.get_dataloaders(args, question_dataset.Dataset, info)

    relation_net = RelationModel(args, info)
    if args.use_cuda:
        relation_net.cuda()
    optimizer = optim.Adam(relation_net.parameters(),
                            lr=args.lr)
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        relation_net.load_state_dict(ckpt['net_dict'])
    pbar = tqdm(total=args.epochs)

    info.model = relation_net
    info.optimizer = optimizer
    info.loss_fn = F.nll_loss
    info.pbar = pbar
    info.ipython = False
    info.timestamp = [('start', time(), 0)]
    info.log_time = lambda x: info.timestamp.append((x, time(), time() - info.timestamp[-1][1]))

    for epoch in range(1, args.epochs + 1):
        train_epoch(args, info)
        pbar.update()
        if args.name:
            torch.save({
                'epoch': epoch,
                'net_dict': relation_net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, '{}.tar'.format(args.name))
    pbar.close()
    embed()

if __name__ == '__main__':
    main()
