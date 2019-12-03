#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : run_train.py
# Author            : Chi Han
# Email             : haanchi@gmail.com
# Date              : 19.11.2019
# Last Modified Date: 03.12.2019
# Last Modified By  : Chi Han
#
# Welcome to this little kennel of Glaciohound!


import torch
from reason.options.train_options import TrainOptions
from scripts.utils.prepare import questions_directly
# from reason.executors import get_executor
from reason.models.parser import Seq2seqParser
from reason.trainer import Trainer
from scripts.utils import register
from utility.logging import Logger

import os
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=True)


opt = TrainOptions().parse()

logger = Logger('', True, True)
tools = register.init_word2index(logger)
tools.load(opt.vcml_question_path)
tools.operations.register_special()

suite = questions_directly(
    os.path.join(opt.vcml_question_path),
    opt, logger)[0]['question_splits']
train = suite['train']
train.base_dataset.load_parts(None, tools)
val = suite['val']
val.base_dataset.load_parts(None, tools)
train_loader = train.get_dataloader()
val_loader = val.get_dataloader()
# train_loader = get_dataloader(opt.vcml_question_path, 'train')
# val_loader = get_dataloader(opt, 'val')

device = torch.device(f'cuda: {0}')
model = Seq2seqParser(opt, tools, device)
# executor = get_executor(opt)
# trainer = Trainer(opt, train_loader, val_loader, model, executor)
trainer = Trainer(opt, train_loader, val_loader, model, None, tools)

trainer.train(opt)
