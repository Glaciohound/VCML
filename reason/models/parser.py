#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : parser.py
# Author            : Chi Han
# Email             : haanchi@gmail.com
# Date              : 19.11.2019
# Last Modified Date: 03.12.2019
# Last Modified By  : Chi Han
#
# Welcome to this little kennel of Glaciohound!


import torch
import torch.nn as nn
from torch.autograd import Variable

from . import create_seq2seq_net  # , get_vocab
from dataset.question_dataset.utils import \
    question_utils, collate_utils, program_utils
# import reason.utils.utils as utils


class Seq2seqParser():
    """Model interface for seq2seq parser"""

    def __init__(self, opt, tools, device):
        self.opt = opt
        # self.vocab = get_vocab(opt)
        self.tools = tools
        self.device = device

        if opt.load_checkpoint_path is not None:
            self.load_checkpoint(opt.load_checkpoint_path)
        else:
            print('| creating new network')
            self.net_params = self._get_net_params(self.opt, self.tools)
            self.seq2seq_op, self.seq2seq_arg = create_seq2seq_net(**self.net_params)

        self.variable_lengths = self.net_params['variable_lengths']
        self.end_id = self.net_params['end_id']
        self.gpu_ids = opt.gpu_ids
        self.criterion = nn.NLLLoss()
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq_op.set_device(self.device)
            self.seq2seq_arg.set_device(self.device)

    def translate(self, questions):
        if isinstance(questions, str):
            only_one = True
            questions = (questions,)
        else:
            only_one = False

        questions_encoded = [
            question_utils.encode_question(
                one_question, self.tools.words
            )
            for one_question in questions
        ]
        pad_setting = {
            'type': 'pad-stack',
            'pad_value': self.tools.words['<NULL>'],
            'tensor': True
        }
        questions_encoded = torch.LongTensor(collate_utils.pad_stack(
            questions_encoded, pad_setting
        ))

        self.set_input(questions_encoded)
        op, arg = self.parse()
        # op = op.argmax(2)
        # arg = arg.argmax(2)
        programs = torch.stack([op, arg], 2).tolist()

        programs_plain = [
            program_utils.decode_program(
                one_program, self.tools.operations, self.tools.arguments
            )
            for one_program in programs
        ]

        if only_one:
            programs_plain = programs_plain[0]
        return programs_plain

    def load_checkpoint(self, load_path):
        # print('| loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path)
        self.net_params = checkpoint['net_params']
        if 'fix_embedding' in vars(self.opt): # To do: change condition input to run mode
            self.net_params['fix_embedding'] = self.opt.fix_embedding
        self.seq2seq_op, self.seq2seq_arg = create_seq2seq_net(**self.net_params)
        self.seq2seq_op.load_state_dict(checkpoint['op_net_state'])
        self.seq2seq_arg.load_state_dict(checkpoint['arg_net_state'])

    def save_checkpoint(self, save_path):
        checkpoint = {
            'net_params': self.net_params,
            'op_net_state': self.seq2seq_op.cpu().state_dict(),
            'arg_net_state': self.seq2seq_arg.cpu().state_dict(),
        }
        torch.save(checkpoint, save_path)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq_op.set_device(self.device)
            self.seq2seq_arg.set_device(self.device)

    def set_input(self, x, y=None):
        input_lengths, idx_sorted = None, None
        if self.variable_lengths:
            x, y, input_lengths, idx_sorted = self._sort_batch(x, y)
        self.x = self._to_var(x)
        if y is not None:
            self.y = self._to_var(y)
        else:
            self.y = None
        self.input_lengths = input_lengths
        self.idx_sorted = idx_sorted

    def set_reward(self, reward):
        self.reward = reward

    def supervised_forward(self):
        assert self.y is not None, 'Must set y value'
        output_logprob_op = self.seq2seq_op(
            self.x, self.y[:, :, 0], self.input_lengths)
        output_logprob_arg = self.seq2seq_arg(
            self.x, self.y[:, :, 1], self.input_lengths)
        '''
        self.loss = self.criterion(
            output_logprob_op[:, :-1, :].contiguous().view(
                -1, output_logprob_op.size(2)),
            self.y[:, 1:, 0].contiguous().view(-1)) +\
            self.criterion(
                output_logprob_arg[:, :-1, :].contiguous().view(
                    -1, output_logprob_arg.size(2)),
                self.y[:, 1:, 1].contiguous().view(-1))
        '''
        self.loss = self.criterion(
            output_logprob_op[:, :-1].contiguous().view(
                -1, output_logprob_op.size(2)
            ),
            self.y[:, 1:, 0].contiguous().view(-1)
        ) + self.criterion(
            output_logprob_arg[:, :-1].contiguous().view(
                -1, output_logprob_arg.size(2)
            ),
            self.y[:, 1:, 1].contiguous().view(-1)
        )

        return output_logprob_op, output_logprob_arg,\
            self._to_numpy(self.loss).sum()

    def supervised_backward(self):
        assert self.loss is not None, 'Loss not defined, must call supervised_forward first'
        self.loss.backward()

    def reinforce_forward(self):
        self.rl_seq_op = self.seq2seq_op.reinforce_forward(
            self.x, self.input_lengths)
        self.rl_seq_op = self._restore_order(self.rl_seq_op.data.cpu())

        self.rl_seq_arg = self.seq2seq_arg.reinforce_forward(
            self.x, self.input_lengths)
        self.rl_seq_arg = self._restore_order(self.rl_seq_arg.data.cpu())

        self.reward_op = None  # Need to recompute reward from environment each time a new sequence is sampled
        self.reward_arg = None  # Need to recompute reward from environment each time a new sequence is sampled
        return self.rl_seq_op, self.rl_seq_arg

    def reinforce_backward(self, entropy_factor=0.0):
        assert self.reward_op is not None, 'Must run forward sampling and set reward before REINFORCE'
        self.seq2seq_op.reinforce_backward(self.reward_op, entropy_factor)
        self.seq2seq_arg.reinforce_backward(self.reward_arg, entropy_factor)

    def parse(self):
        output_sequence_op = self.seq2seq_op.sample_output(
            self.x, self.input_lengths)
        output_sequence_op = self._restore_order(
            output_sequence_op.data.cpu())
        output_sequence_arg = self.seq2seq_arg.sample_output(
            self.x, self.input_lengths)
        output_sequence_arg = self._restore_order(
            output_sequence_arg.data.cpu())
        return output_sequence_op, output_sequence_arg

    def _get_net_params(self, opt, tools):
        net_params = {
            # 'input_vocab_size': len(vocab['question_token_to_idx']),
            # 'output_vocab_size': len(vocab['program_token_to_idx']),
            'input_vocab_size': len(tools.words),
            'output_operation_size': len(tools.operations),
            'output_argument_size': len(tools.arguments),
            'hidden_size': opt.hidden_size,
            'word_vec_dim': opt.word_vec_dim,
            'n_layers': opt.n_layers,
            'bidirectional': not opt.not_bidirectional,
            'variable_lengths': opt.variable_lengths,
            'use_attention': opt.use_attention,
            'encoder_max_len': opt.encoder_max_len,
            'decoder_max_len': opt.decoder_max_len,
            'start_id': opt.start_id,
            'end_id': opt.end_id,
            'word2vec_path': opt.word2vec_path,
            'fix_embedding': opt.fix_embedding,
        }
        return net_params

    def _sort_batch(self, x, y):
        _, lengths = torch.eq(x, self.end_id).max(1)
        lengths += 1
        lengths_sorted, idx_sorted = lengths.sort(0, descending=True)
        x_sorted = x[idx_sorted]
        y_sorted = None
        if y is not None:
            y_sorted = y[idx_sorted]
        lengths_list = lengths_sorted.numpy()
        return x_sorted, y_sorted, lengths_list, idx_sorted

    def _restore_order(self, x):
        if self.idx_sorted is not None:
            inv_idxs = self.idx_sorted.clone()
            inv_idxs.scatter_(0, self.idx_sorted, torch.arange(x.size(0)).long())
            return x[inv_idxs]
        return x

    def _to_var(self, x):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            x = x.to(self.device)
        return Variable(x)

    def _to_numpy(self, x):
        return x.data.cpu().numpy().astype(float)
