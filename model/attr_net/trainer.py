import os
import json
import torch
import utils
from tqdm import tqdm
import numpy as np
import random


class Trainer:

    def __init__(self, opt, model, train_loader, val_loader=None, tester=None):
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.args = opt

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tester = tester
        self.model = model
        self.checkpoint_file_name = '%s/checkpoint_best.pt' % self.run_dir
        with open(opt.dicts_file, 'r') as f:
            self.dicts = json.load(f)

        self.stats = {
            'train_losses': [],
            'train_losses_ts': [],
            'val_losses': [],
            'val_losses_ts': [],
            'best_val_loss': 9999,
            'model_t': 0
        }

    def train(self):
        print('==> start training')
        t = 0
        epoch = 0
        loss = 0
        pbar = tqdm(total = self.num_iters)
        while t < self.num_iters:
            epoch += 1
            for data, label, relation, _, _, boxes, image_shape in self.train_loader:
                self.model.set_input(data, label, relation)
                if data.shape[1] == 0:
                    continue
                self.model.step()
                loss = loss*0.99 + self.model.get_loss()*0.01
                pbar.update()
                t += 1

                if t % self.display_every == 0:
                    self.stats['train_losses'].append(loss.item())
                    pbar.write('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                    self.stats['train_losses_ts'].append(t)

                if t % self.checkpoint_every == 0:
                    if self.val_loader is not None:
                        print('| checking validation loss and accuracy')
                        val_loss, test_score, avg_objects, max_value = self.check_val_loss()
                        print('| validation loss %f, test_score %f, average num_objects %f, max_value %f'\
                              % (val_loss.item(), test_score, avg_objects, max_value))
                        if val_loss <= self.stats['best_val_loss']:
                            print('| best model')
                            self.stats['best_val_loss'] = val_loss.item()
                            self.stats['model_t'] = t
                            self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir)
                        self.stats['val_losses'].append(val_loss.item())
                        self.stats['val_losses_ts'].append(t)
                    pbar.write('| saving checkpoint')
                    self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' %
                                                (self.run_dir, t))
                    self.model.save_checkpoint(os.path.join(self.run_dir, 'checkpoint.pt'))
                    with open('%s/stats.json' % self.run_dir, 'w') as fout:
                        json.dump(self.stats, fout)

                if t >= self.num_iters:
                    break
        pbar.close()

    def check_val_loss(self):
        self.model.eval_mode()
        loss = 0
        t = 0
        total_correct, total, total_objects, total_max = 0, 0, 0, 0
        with torch.no_grad():
            for data, label, relation, imageId, _, boxes, image_shape in self.val_loader:
                if data.shape[1] == 0:
                    continue
                t += 1
                if t >= self.args.test_samples:
                    break
                self.model.set_input(data, label, relation)
                label_pred, relation_pred = self.model.forward()
                loss += self.model.get_loss()

                for i in range(boxes.shape[0]):
                    correct, num, n_objects, max_value =\
                        self.tester.test(imageId[i],
                                        (np.array(boxes[i]),
                                        np.array(label_pred[i].cpu().detach()),
                                        np.array(relation_pred[i].cpu().detach()),
                                        image_shape[i]))

                    total_objects += n_objects
                    total_correct += correct
                    total += num
                    total_max += max_value


        self.model.train_mode()
        return loss / t if t is not 0 else 0,\
            total_correct / total if total != 0 else 0,\
            total_objects / total if total != 0 else 0,\
            total_max / total if total != 0 else 0

def get_trainer(opt, model, train_loader, val_loader=None, tester=None):
    return Trainer(opt, model, train_loader, val_loader, tester)
