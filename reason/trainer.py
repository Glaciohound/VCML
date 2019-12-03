#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : trainer.py
# Author            : Chi Han
# Email             : haanchi@gmail.com
# Date              : 19.11.2019
# Last Modified Date: 27.11.2019
# Last Modified By  : Chi Han
#
# Welcome to this little kennel of Glaciohound!


from itertools import chain
import json
import torch
import reason.utils.utils as utils

from tqdm import tqdm


class Trainer():
    """Trainer"""

    def __init__(self, opt, train_loader, val_loader, model, executor, tools):
        self.tools = tools
        self.opt = opt
        self.reinforce = opt.reinforce
        self.reward_decay = opt.reward_decay
        self.entropy_factor = opt.entropy_factor
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        '''
        if self.reinforce:
            if opt.dataset == 'clevr':
                self.vocab = utils.load_vocab(opt.clevr_vocab_path)
            elif opt.dataset == 'clevr-humans':
                self.vocab = utils.load_vocab(opt.human_vocab_path)
            else:
                raise ValueError('Invalid dataset')
        '''

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.executor = executor
        # self.optimizer = torch.optim.Adam(filter(
        self.optimizer = torch.optim.SGD(filter(
            lambda p: p.requires_grad,
            chain(model.seq2seq_op.parameters(),
                  model.seq2seq_arg.parameters())),
                                          lr=opt.learning_rate)

        self.stats = {
            'train_losses': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0
        }
        if opt.visualize_training:
            from reason.utils.logger import Logger
            self.logger = Logger('%s/logs' % opt.run_dir)

    def train(self, opt):
        training_mode = 'reinforce' if self.reinforce else 'seq2seq'
        print('| start training %s, running in directory %s' % (training_mode, self.run_dir))
        t = 0
        epoch = 0
        baseline = 0
        pbar = tqdm(range(self.num_iters))
        while t < self.num_iters:
            epoch += 1

            # for x, y, ans, idx in self.train_loader:
            for data in self.train_loader:

                # log and save checkpoint
                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    sample_questions = data['question']
                    program = self.model.translate(sample_questions)
                    pbar.write(f'iter: {t}')
                    for i in range(1):
                        pbar.write('| Translating example: {0} {1}'.format(
                            sample_questions[i], program[i]))

                    print('| checking validation accuracy')
                    val_acc = self.check_val_accuracy()
                    print('| validation accuracy %f' % val_acc)
                    if val_acc >= self.stats['best_val_acc']:
                        print('| best model')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir)
                        self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' % (self.run_dir, t))
                    self.stats['val_accs'].append(val_acc)
                    self.log_stats('val accuracy', val_acc, t)
                    self.stats['val_accs_ts'].append(t)

                    '''
                    print('| checking valication loss')
                    val_loss = self.check_val_loss()
                    print('| validation loss %f' % val_loss)
                    self.stats['val_losses'].append(val_loss)
                    self.log_stats('val loss', val_loss, t)
                    '''

                    self.model.save_checkpoint(
                        '%s/checkpoint.pt' % self.run_dir)
                    with open('%s/stats.json' % self.run_dir, 'w') as fout:
                        json.dump(self.stats, fout)
                    self.log_params(t)

                # training
                x = data['question_encoded']
                y = concat_pad(data['program_encoded'], opt.decoder_max_len,
                               self.tools.operations,
                               self.tools.arguments)
                ans = torch.LongTensor(data['answer_encoded'])
                idx = None
                t += 1
                pbar.update()
                loss, reward = None, None
                self.model.set_input(x, y)
                self.optimizer.zero_grad()
                if self.reinforce:
                    pred = self.model.reinforce_forward()
                    reward = self.get_batch_reward(pred, ans, idx, 'train')
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward(advantage)
                    self.model.reinforce_backward(self.entropy_factor)
                else:
                    _, _, loss = self.model.supervised_forward()
                    self.model.supervised_backward()
                self.optimizer.step()

                if self.reinforce:
                    self.stats['train_batch_accs'].append(reward)
                    self.log_stats('training batch reward', reward, t)
                    pbar.set_description_str(
                        '| epoch %d, reward %f' % (epoch, reward))
                else:
                    self.stats['train_losses'].append(loss)
                    self.log_stats('training batch loss', loss, t)
                    pbar.set_description_str(
                        '| epoch %d, loss %f' % (epoch, loss))
                self.stats['train_accs_ts'].append(t)

                if t >= self.num_iters:
                    break

    def check_val_loss(self):
        loss = 0
        t = 0
        # for x, y, _, _ in self.val_loader:
        for data in tqdm(self.val_loader):
            x = data['question_encoded']
            y = concat_pad(data['program_encoded'], self.opt.decoder_max_len,
                           self.tools.operations,
                           self.tools.arguments)
            # ans = torch.LongTensor(data['answer_encoded'])
            # idx = None
            self.model.set_input(x, y)
            loss += self.model.supervised_forward()[2]
            t += 1
        return loss / t if t != 0 else 0

    def check_val_accuracy(self):
        reward = 0
        t = 0

        val_it = iter(self.val_loader)
        for t in tqdm(range(self.opt.val_size)):
            # for data in self.val_loader:
            data = val_it.next()
            x = data['question_encoded']
            y = concat_pad(data['program_encoded'], self.opt.decoder_max_len,
                           self.tools.operations,
                           self.tools.arguments)
            # ans = torch.LongTensor(data['answer_encoded'])
            # idx = None
            self.model.set_input(x, y)
            # reward += self.get_batch_reward(pred, ans, idx, 'val')
            op, arg = self.model.parse()
            pred = torch.stack([op, arg], 2)
            reward += (pred[:, :-1] == y[:, 1:]).numpy().astype(float).mean()

        reward = reward / self.opt.val_size
        return reward

    def get_batch_reward(self, programs, answers, image_idxs, split):
        pg_np = programs.numpy()
        ans_np = answers.numpy()
        idx_np = image_idxs.numpy()
        reward = 0
        for i in range(pg_np.shape[0]):
            pred = self.executor.run(pg_np[i], idx_np[i], split)
            ans = self.vocab['answer_idx_to_token'][ans_np[i]]
            if pred == ans:
                reward += 1.0
        reward /= pg_np.shape[0]
        return reward

    def log_stats(self, tag, value, t):
        if self.visualize_training and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)

    def log_params(self, t):
        if self.visualize_training and self.logger is not None:
            named_parameters = chain(self.model.seq2seq_op.named_parameters(),
                                     self.model.seq2seq_arg.named_parameters())
            for tag, value in named_parameters:
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, self._to_numpy(value), t)
                if value.grad is not None:
                    self.logger.histo_summary('%s/grad' % tag, self._to_numpy(value.grad), t)

    def _to_numpy(self, x):
        return x.data.cpu().numpy()


def concat_pad(programs, max_length, operation, argument):

    def pad(one_program):
        background = torch.zeros((max_length, 2), dtype=int)
        background[0, 0] = operation['<START>']
        background[0, 1] = argument['<START>']
        background[1:, 0] = operation['<END>']
        background[1:, 1] = argument['<END>']
        background[1: 1 + one_program.shape[0]] = one_program
        return background

    padded = [pad(one) for one in programs]
    output = torch.stack(padded)
    return output
