#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : play.py
# Author            : Chi Han
# Email             : haanchi@gmail.com
# Date              : 19.11.2019
# Last Modified Date: 20.11.2019
# Last Modified By  : Chi Han
#
# Welcome to this little kennel of Glaciohound!


from reason.models.parser import Seq2seqParser
from scripts.utils import register
from utility.logging import Logger
from IPython import embed


class fixed_opt:
    def __init__(self, **kwarg):
        self.__dict__.update(kwarg)


opt = fixed_opt(
    load_checkpoint_path='../data/log/reason/CLEVR/checkpoint_best.pt',
    gpu_ids=[0],
    fix_embedding=False
)

logger = Logger('', True, True)
tools = register.init_word2index(logger)
tools.load('../data/vcml_data/CLEVR/augmentation/questions/normal_exist_both')

model = Seq2seqParser(opt, tools)

embed()
