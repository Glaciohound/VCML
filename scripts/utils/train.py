#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 27.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# codes for running training

import torch
from . import evaluate
from .referential import ref_epoch
# from utility.common import detach
from dataset.question_dataset.utils import program_utils


def loss_classification(loss, confidence, gt_class, args):
    yes = (gt_class == 1) * confidence
    no = (gt_class == 0) * confidence
    yes_num = yes.sum()
    no_num = no.sum()

    yes_tensor = torch.Tensor(yes).to(loss.device)
    no_tensor = torch.Tensor(no).to(loss.device)
    yes_loss = (loss * yes_tensor).sum()
    no_loss = (loss * no_tensor).sum()

    if args.balance_classification:
        if no_num == 0:
            output = yes_loss / yes_num
        elif yes_num == 0:
            output = no_loss / no_num
        else:
            output = (yes_loss / yes_num + no_loss / no_num) / 2
    else:
        output = (yes_loss + no_loss) / (yes_num + no_num)

    return output


def loss_plain(loss, confidence, category, args):
    if category == 'conceptual':
        output = loss * args.conceptual_weight
    else:
        output = loss
    return output


def object_penalty(objects_in_one, penalty_weight):
    if objects_in_one is None:
        return 0
    norm_sum = objects_in_one.pow(2).sum()
    penalty = norm_sum * penalty_weight
    return penalty


def calculate_loss(losses, data, objects, args):
    processed = []
    for i, loss in enumerate(losses):
        q_type = data['type'][i]
        if q_type == 'classification':
            this_loss = loss_classification(
                loss, data['confidence'][i],
                data['answer'][i], args
            )
        else:
            this_loss = loss_plain(
                loss, data['confidence'][i],
                data['category'][i], args
            )
        one_obj_penalty = object_penalty(objects[i], args.length_penalty)
        processed.append(this_loss + one_obj_penalty)
    return torch.stack(processed).mean()


def run_batch(data, model, args):
    losses, outputs, debugs, objects = model(data)
    if len(args.concept_filter) > 0:
        for i, q_type in enumerate(data['type']):
            if q_type == 'classification':
                mask = data['confidence'][i] * 0
                mask[:, args.concept_filter] = 1
                data['confidence'][i] = data['confidence'][i] * mask
    loss = calculate_loss(losses, data, objects, args)
    penalty = model.penalty()

    return loss + penalty, outputs


def max_grad(parameters):
    parameters = list(filter(lambda param: param.grad is not None, parameters))
    max_grad = max(param.grad.data.abs().max() for param in parameters)
    return max_grad


def any_epoch(coach, prepare, recording, dataloader, reset, is_train):
    prepare()
    args = coach.args
    model = coach.model
    if reset:
        recording.reset()

    def inner():
        pbar = coach.logger.tqdm(dataloader)
        for data in pbar:
            if is_train:
                coach.optimizer.zero_grad()
                if not args.use_gt_program:
                    data['program_parsed'] = coach.question_parser.translate(
                        data['question']
                    )
                    data['program_parsed_encoded'] = [
                        program_utils.encode_program(
                            one_program,
                            coach.tools.operations, coach.tools.arguments,
                        ) for one_program in data['program_parsed']
                    ]
                loss, outputs = run_batch(data, model, args)
                loss.backward()
                coach.optimizer.step()
                model.update()
            else:
                loss, outputs = run_batch(data, model, args)

            analyze_result = {'loss': loss.item()}
            analyze_result.update(evaluate.eval(outputs, data, args))
            # analyze_result['max_grad'] = detach(max_grad(model.parameters()))
            recording.record(analyze_result)

            pbar.set_description_str(
                str(recording)[: coach.logger.get_ncols() - 40])

    if is_train:
        inner()
    else:
        with torch.no_grad():
            inner()


def run_epoch(coach, args, i_epoch):
    with coach.logger.levelup():
        coach.model.visualize(coach.local_dir, coach.plt)

    coach.send(i_epoch)

    if 'train' in args.in_epoch:
        coach.synchronize()
        coach.logger.line_break()
        coach.logger('Training')
        any_epoch(
            coach, coach.model.train, coach.train_recording, coach.train,
            False, True
        )
        coach.logger('Visualizing plots')
        coach.send(coach.train_recording)

    if 'val' in args.in_epoch:
        coach.synchronize()
        coach.logger.line_break()
        coach.logger('Validation')
        any_epoch(
            coach, coach.model.eval, coach.val_recording, coach.val,
            True, False
        )
        coach.logger('Visualizing plots')
        coach.send(coach.val_recording)

    if (i_epoch + 1) % args.test_every == 0:

        if 'test' in args.in_epoch:
            coach.synchronize()
            coach.logger.line_break()
            coach.logger('Testing')
            any_epoch(
                coach, coach.model.eval, coach.test_recording, coach.test,
                True, False
            )
            coach.logger('Visualizing plots')
            coach.send(coach.test_recording)

        if 'test_ref' in args.in_epoch:
            coach.synchronize()
            coach.logger.line_break()
            coach.logger('Testing by Referential Expression')
            ref_epoch(coach, coach.model.eval, coach.ref_recording,
                      coach.ref_dataset)
            coach.logger('Visualizing plots')
            coach.send(coach.ref_recording)

    coach.synchronize()
    coach.logger.line_break()


def train(coach, args):
    for epoch in range(args.epochs):
        coach.epoch += 1
        coach.logger.split_line()
        coach.step(coach.epoch)

        coach.logger.line_break()
        coach.logger('epoch {}'.format(coach.epoch))
        with coach.logger.levelup():
            run_epoch(coach, args, coach.epoch)

            if not args.silent:
                coach.logger('Saving checkpoint')
                coach.save(coach.epoch)
                coach.logger('Saving best-version checkpoint')
                coach.copy_ckpt(coach.epoch, 'best')
                coach.logger('Saving models and tools')
                coach.save_partial(coach.epoch)
