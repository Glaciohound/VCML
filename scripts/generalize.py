# coding: utf-8
import torch
get_ipython().run_line_magic('cd', '~/iLab/concept-net/')
import config
from argparse import Namespace
import utils.recording as recording
get_ipython().run_line_magic('cd', 'model/')
from importlib import reload
import isinstance as Is
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch.nn.init as init

ckpt = torch.load('~/iLab/data/gqa/checkpoints/relation_net/clevr_query.tar')
info = Namespace()
args = config.Config(info)

dataset = Is.instance_dataset(args, info, ckpt)
info.loss_fn = F.mse_loss
info.log = {}
init_ = lambda x: [init.normal_(p, 0, 0.1) for p in x.parameters()].append(0)
n = 5000
stopping_th = 0.9

reload(Is)
info.model = Is.InstanceNet(args, info)
info.optimizer = optim.Adam(info.model.parameters(), lr = args.lr)
dataset = Is.instance_dataset(args, info, ckpt)

init_(info.model)
info.recording = recording.Recording(args, info, ['train_loss', 'train_acc', 'val_loss', 'val_acc'], mode='decaying')
for i in tqdm(range(n)):
    info.optimizer.zero_grad()
    dataset.to_split('train')
    data = dataset.get_random(10)
    output = info.model(data[0])
    loss = info.loss_fn(output, data[1])
    loss.backward()
    acc_train = ((output > 0.5).int() == data[1].int()).float().mean()

    dataset.to_split('val')
    data = dataset.get_random(10)
    output = info.model(data[0])
    acc_val = ((output > 0.5).int() == data[1].int()).float().mean()
    loss_ = info.loss_fn(output, data[1])
    info.recording.update((loss, acc_train, loss_, acc_val))
    info.optimizer.step()
    if info.recording['acc_train'] > stopping_th and i > 100:
        break

for i in range(60):
    init_(info.model)
    dataset = Is.instance_dataset(args, info, ckpt)
    info.log.append([])

    dataset.to_split('train')
    for j in tqdm(range(n)):
        info.optimizer.zero_grad()
        data = dataset.get_random(10)
        output = info.model(data[0])
        loss = info.loss_fn(output, data[1])
        loss.backward()
        info.optimizer.step()

    dataset.to_split('val')
    for j in tqdm(range(n//10)):
        data = dataset.get_random(10)
        output = info.model(data[0])
        acc_val = ((output > 0.5).int() == data[1].int()).float().mean()
        info.log[-1].append(acc_val)
    info.log.append(dataset.exclude)
