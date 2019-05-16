#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
from torch.autograd import Variable
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import sys
args = sys.args
info = sys.info
import matplotlib.pyplot as plt
import numpy as np
from model.utils.resnet import Attribute_Network
from utils.common import to_numpy, to_normalized, min_fn,\
    matmul, to_tensor, vistb, arange, logit_exist, log_or, logit_xand
import pprint

class HEmbedding(nn.Module):
    def __init__(self):
        super(HEmbedding, self).__init__()
        if args.model == 'h_embedding_mul':
            self.model = 'mul'
        elif args.model == 'h_embedding_add':
            self.model = 'add'
        elif args.model == 'h_embedding_add2':
            self.model = 'add2'
        self.build()

    def build(self):
        if self.model == 'add2':
            self.obj_embed_dim = args.embed_dim // 2
        else:
            self.obj_embed_dim = args.embed_dim

        self.attribute_embedding = self.build_embedding(args.max_concepts, args.feature_dim,
                                                        'attribute', 0)
        self.feature_mlp = self.build_mlp(args.feature_dim, self.obj_embed_dim,
                                             'feature', args.hidden_dim)

        self.concept_embedding = self.build_embedding(args.max_concepts, args.embed_dim,
                                                      'concept', args.hidden_dim)
        self.relation_embedding = self.build_embedding(args.max_concepts, args.embed_dim,
                                                    'relation', 0, matrix=self.model=='mul')

        self.resnet_model = Attribute_Network()

        self.true_th_ = nn.Parameter(info.to(torch.Tensor([args.true_th])))
        self.same_class_th_ = nn.Parameter(info.to(torch.Tensor([args.true_th])))
        self.temperature_ = nn.Parameter(info.to(torch.Tensor([args.temperature_init])))

        self.similarity = F.cosine_similarity
        self.scale = lambda x: self.temperature * (x-self.true_th)
        self.huge_value = 100


    @property
    def temperature(self):
        return self.temperature_.detach()

    @property
    def true_th(self):
        return self.true_th_.detach()
        #return self.true_th_

    @property
    def same_class_th(self):
        return self.same_class_th_.detach()


    ''' logit function taking into (x, y) and compute the logit for the similarity probability between them '''
    def logit_fn(self, *arg, threshold=None, **kwarg):
        if not threshold:
            threshold = self.true_th
        return self.temperature * (self.similarity(*arg, **kwarg) - threshold)

    def build_mlp(self, dim_in, dim_out, name, dim_hidden):
        if dim_hidden <= 0:
            return nn.Linear(dim_in, dim_out)
        linear1 = nn.Linear(dim_in, dim_hidden)
        linear2 = nn.Linear(dim_hidden, dim_out)
        setattr(self, name+'_linear1', linear1)
        setattr(self, name+'_linear2', linear2)
        #return lambda x: linear2(torch.sigmoid(linear1(x)))
        return lambda x: linear2(linear1(x))

    def build_embedding(self, n, dim, name, dim_hidden, matrix=False):
        '''  building the parameters for embedding '''
        if dim_hidden <= 0:
            if not matrix:
                hidden_embedding = nn.Embedding(n, dim)
            else:
                hidden_embedding = nn.Embedding(n, dim * dim)
            embedding = hidden_embedding
            setattr(self, name+'_hidden_embedding', hidden_embedding)
        else:
            hidden_embedding = nn.Embedding(n, dim_hidden)
            if not matrix:
                hidden_linear = nn.Linear(dim_hidden, dim)
            else:
                hidden_linear = nn.Linear(dim_hidden, dim * dim)
            setattr(self, name+'_hidden_embedding', hidden_embedding)
            setattr(self, name+'_hidden_linear', hidden_linear)
            embedding = lambda x: hidden_linear(hidden_embedding(x))

        '''
        branching conditioned on whether the concept embedding will actually
        be used in form of a linear transformation, i.e., a matrix
        '''
        if not matrix:
            return embedding
        else:
            def matrix_embedding(x):
                oneD_embedding = hidden_embedding(x)
                matrix_shape = oneD_embedding.shape[:-1] + torch.Size((dim, dim))
                return oneD_embedding.view(matrix_shape)
            return matrix_embedding


    def forward(self, data):
        batch_size = data['answer'].shape[0]
        program_length = data['program'][0].shape[0]
        processed = dict()

        processed['concept_arguments'] = self.concept_embedding(info.to(data['program'])[:, :, 1])
        processed['relation_arguments'] = self.relation_embedding(info.to(data['program'])[:, :, 1])
        processed['all_concepts'] = info.to(self.concept_embedding(Variable(info.to(torch.arange(args.max_concepts)).long())))
        processed['program_length'] = program_length
        def load_list(k):
            if k in data:
                processed[k] = [info.to(to_tensor(v)) for v in data[k]]
        load_list('scene')
        load_list('object_classes')
        load_list('answer')
        if info.visual_dataset.mode == 'detected':
            processed['feature'], processed['recognized'] = self.resnet_model(data)

        losses = [None for i in range(batch_size)]
        outputs = [None for i in range(batch_size)]

        for i in range(batch_size):
            self.run_piece(data, processed, i,
                           losses, outputs)

        losses = torch.stack(losses)
        program_length = data['program'].shape[1]

        return losses, outputs

    ''' fun one piece of data '''
    def run_piece(self, data, processed, i,
                  losses, outputs):

        ''' object inputs and concept embeddings '''

        if info.visual_dataset.mode == 'encoded_sceneGraph':
            object_input = self.embed_without_bg(processed['scene'][i])
        elif info.visual_dataset.mode == 'pretrained':
            object_input = processed['scene'][i]
        elif info.visual_dataset.mode == 'detected':
            object_input = processed['recognized'][i][1]
        objects = self.feature_mlp(object_input)
        objects = to_normalized(objects)

        all_concepts = processed['all_concepts']
        if self.model == 'add2':
            all_concepts = torch.cat([all_concepts[:, :self.obj_embed_dim],
                                      to_normalized(all_concepts[:, self.obj_embed_dim:])], dim=1)

        ''' pre-processing: calculating all obj-feasible-concept logits '''
        concept_names = info.vocabulary.concepts
        concept_index = [info.protocol['concepts', name] for name in concept_names]
        operators = all_concepts[concept_index, :self.obj_embed_dim]
        latters = all_concepts[concept_index, self.obj_embed_dim:]
        projected = objects[:, None] + operators[None, :]

        ''' logits[i, j, k] = <obj_i + operator_j, concept_k> '''
        logits = self.logit_fn(projected[:, :, None], latters[None, None], dim=3)
        ''' self_logits[i, j] = <obj_i, operator_j, concept_j> '''
        self_logits = torch.diagonal(logits, dim1=1, dim2=2)
        other_logits = min_fn(self_logits[:, None, :], logits)
        other_logits[(self_logits[:, :, None] == logits)] = -self.huge_value
        submax_logits, subargmax = other_logits.max(2)
        '''
        conditinal probability:
            Pr(concept_i <- obj_j | ∃ concept_i'~concept_i, s.t. concept_i' <- obj_j)
        '''
        conditional_logit = logit_exist(self_logits, submax_logits)

        self.log_logits(logits, self_logits, submax_logits, concept_index)


        ''' if dealing with a classification task '''
        if data['type'][i] == 'classification':
            to_classify = data['program'][i, :, 1].tolist()
            to_classify = [concept_index.index(_index) for _index in to_classify]

            binary_loss = F.binary_cross_entropy_with_logits(
                conditional_logit, processed['object_classes'][i], reduction='none')
            losses[i] = binary_loss[:, to_classify].mean()
            outputs[i] = conditional_logit[:, to_classify]
            data['object_classes'][i] = data['object_classes'][i][:, to_classify]
            return


        ''' otherwise, dealing with QA tasks '''

        if info.visual_dataset.mode in ['encoded_sceneGraph', 'pretrained']:
            num_objects = data['scene'][i].shape[0]
        else:
            num_objects = data['object_lengths'][i]

        ''' initializing attentions '''
        init_attention = lambda n: Variable(info.to(torch.ones(n))) * self.huge_value
        attention = {'concepts': init_attention(args.max_concepts),
                     'objects': init_attention(num_objects)}
        def attention_copy(attention_):
            return {k: v * 1 for k, v in attention_.items()}

        ''' operation[Filter] '''
        def filter_op(attention_, concept_, arg_i):
            _output = {}

            ''' if h_embedding_add2 model: '''
            if self.model == 'add2':
                index = concept_index.index(arg_i)

                _output['concepts'] = min_fn(attention_['concepts'], self.logit_fn(
                  all_concepts, concept_[None]))
                _output['objects'] = min_fn(attention_['objects'],
                                            conditional_logit[:, index])

                ''' unused as in default args.penalty == 0 '''
                sanity_loss = info.to(torch.tensor(0.))\
                    -log_or(self_logits, submax_logits).min()

            else:
                _output['objects'] = self.logit_fn(objects, concept_[None])
                _output['concepts'] = self.logit_fn(all_concepts, concept_[None])
            return _output, sanity_loss

        def assign(attention_, value):
            for k, v in attention_.items():
                attention_[k] = v * 0 + value

        penalty_loss = info.to(torch.tensor(0.))

        '''
        running the program
        definition of the modules are in dataset/tools/program_utils.py
        '''
        for j, (op, arg) in enumerate(data['program'][i]):
            op_s = info.protocol['operations', int(op)]
            arg_s = info.protocol['concepts', int(arg)]
            for k, v in attention.items():
                attention[k] = v * 1

            if op_s == 'select':
                if arg_s == 'object_only':
                    attention['concepts'][:] = -self.huge_value
                elif arg_s == 'concept_only':
                    attention['objects'][:] = -self.huge_value
                else:
                    raise Exception('unsupported select argument')

            elif op_s == 'filter':
                attention, exist_loss = filter_op(attention, processed['concept_arguments'][i, j],
                                                  data['program'][i, j, 1])
                penalty_loss = penalty_loss + exist_loss

            elif op_s == 'verify':
                attention, exist_loss = filter_op(attention, processed['concept_arguments'][i, j],
                                                  data['program'][i, j, 1])
                penalty_loss = penalty_loss + exist_loss
                attention['concepts'][torch.arange(args.max_concepts).long() != arg] =\
                    -self.huge_value

            elif op_s == 'choose':
                assign(attention, -self.huge_value)
                attention['concepts'][arg] = self.huge_value

            elif op_s == 'exist':
                attention['concepts'][len(info.protocol['concepts']):] = -self.huge_value

                s = max(attention['concepts'].max(), attention['objects'].max())
                yes = s
                no = -yes

                assign(attention, -self.huge_value)
                attention['concepts'][info.protocol['concepts', 'yes']] = yes
                attention['concepts'][info.protocol['concepts', 'no']] = no

            elif op_s.startswith('transfer'):

                if op_s.startswith('transfer_o'):
                    gather = torch.matmul(F.softmax(attention['objects'], -1), to_normalized(objects))
                else:
                    gather = torch.matmul(F.softmax(attention['concepts'], -1), to_normalized(all_concepts))
                dim = gather.shape[0]

                if op_s.endswith('c'):
                    to_compare = all_concepts
                else:
                    to_compare = objects

                if self.model == 'mul':
                    matrix = processed['relation_arguments'][i, j]
                    transferred = torch.matmul(gather, matrix)
                    to_compare = torch.matmul(to_compare, matrix)
                    _output = self.scale(self.similarity(to_compare, transferred[None])-
                                        self.similarity(all_concepts, transferred[None]))

                elif self.model == 'add':
                    transferred = gather + processed['relation_arguments'][i, j]
                    _output = self.logit_fn(to_compare, transferred[None])

                elif self.model == 'add2':
                    transferred = gather + processed['relation_arguments'][i, j][:dim]
                    to_compare = to_compare[:, -dim:]
                    _output = self.logit_fn(to_compare, transferred[None])

                assign(attention, -self.huge_value)
                if op_s.endswith('c'):
                    attention['concepts'] = _output
                else:
                    attention['objects'] = _output


            elif op_s in ['<NULL>', '<START>', '<END>', '<UNKNOWN>']:
                pass

            else:
                raise Exception('no such operation %s supported' % op_s)

        ''' modyfying values'''
        attention['concepts'][len(info.protocol['concepts']):] = - self.huge_value
        log_softmax = F.log_softmax(attention['concepts'], dim=0)
        losses[i] = F.nll_loss(log_softmax[None], processed['answer'][i][None]) +\
            penalty_loss * args.penalty
        outputs[i] = log_softmax

    '''
    (only used in ground-truth mode)
    get object embeddings if they are not background
        ('paddings' of value -1)
    '''
    def embed_without_bg(self, x):
        if isinstance(x, list):
            x = info.to(x)

        x = x+1
        return self.attribute_embedding(Variable(x)).sum(-2)


    def get_embedding(self, name, relational=False):
        embedding = self.concept_embedding\
            if not relational\
            else self.relation_embedding
        return embedding(Variable(info.to(to_tensor([
            info.protocol['concepts', name]]))))[0]



    ''' Codes for visualizing '''


    def visualize_embedding(self, relation_type=None):
        to_visualize = {}

        if relation_type is not None:
            matrix = to_numpy(self.get_embedding(relation_type, True))
            if self.model == 'add':
                to_visualize[relation_type] = matrix

        names = info.vocabulary.concepts

        for name in names:
            vec = to_numpy(self.get_embedding(name))
            if self.model != 'add2':
                vec_norm = to_normalized(vec)
            else:
                vec_norm = np.concatenate([vec[:self.obj_embed_dim],
                                           to_normalized(vec[self.obj_embed_dim:])])
            to_visualize[name+'_ori'] = vec_norm

            if relation_type is not None:
                if self.model == 'mul':
                    to_visualize[name+'_convert'] = to_normalized(np.matmul(vec, matrix))
                else:
                    to_visualize[name+'_convert'] = to_normalized(to_normalized(vec) + matrix)

        to_visualize['zero_point'] = list(to_visualize.values())[0] * 0

        original = np.array([to_visualize[name+'_ori'] for name in names])

        if relation_type == 'isinstance':
            original = np.concatenate([original, np.array([to_numpy(to_normalized(self.get_embedding(cat, False)))
                for cat in sorted(info.vocabulary.records)])])
            for cat in info.vocabulary.records:
                to_visualize[cat+'_concept'] = to_numpy(to_normalized(self.get_embedding(cat, False)))

        if 'query' in args.subtasks and 'add' in self.model:
            for cat in info.vocabulary.records:
                to_visualize[cat+'_operation'] = to_numpy(to_normalized(self.get_embedding(cat, True)))

        vistb(to_visualize, args.visualize_dir)

        if relation_type is not None:
            if self.model == 'mul':
                converted = to_normalized(matmul(original, matrix))
                distance_mat = matmul(converted, to_normalized(converted-original).transpose())
            else:
                converted = to_normalized(original + matrix[None])
                distance_mat = matmul(converted, to_normalized(original).transpose())

            self.matshow(distance_mat, 'distance')
            self.matshow(matmul(converted, converted.transpose()), 'cosine_converted')
            self.matshow(matrix[None] if matrix.ndim < 2 else matrix, relation_type+'_matrix')
        else:
            converted = None
        self.matshow(matmul(to_normalized(original), to_normalized(original).transpose()),
                     'cosine_ori')

        if self.model == 'add2':
            for dim, name in [(arange(self.obj_embed_dim),
                               'cosine_prefix'),
                              (arange(self.obj_embed_dim, self.obj_embed_dim*2),
                               'cosine_postfix')]:
                self.matshow(matmul(to_normalized(original[:, dim]),
                                    to_normalized(original[:, dim]).transpose()),
                             name)

        plt.close()
        return to_visualize, original, converted

    def log_logits(self, logits, self_logits, submax_logits, concept_index):
        if 'logit_scatter' not in info.log:
            self.init_logits()
        def to_list(tensor):
            return tensor.contiguous().view(-1).tolist()
        subargmax = logits.argmax(2)

        for i in range(logits.shape[1]):
            index = concept_index[i]
            info.log['logit_scatter']['self_logits'][index] += to_list(self_logits[:, i])
            info.log['logit_scatter']['submax_logit'][index] += to_list(submax_logits[:, i])
            info.log['logit_scatter']['believed_logit'][index] += to_list(logits[:, i])
            info.log['logit_scatter']['ref'][index] += to_list(self_logits)

            for j in range(logits.shape[0]):
                info.log['match'][i, subargmax[j, i]] += 1

    def visualize_logit(self):
        if 'logit_scatter' not in info.log:
            return

        min_ = min([min(logits)
                    for series in info.log['logit_scatter'].values()
                    for logits in series if logits])
        max_ = max([max(logits)
                    for series in info.log['logit_scatter'].values()
                    for logits in series if logits])
        for i in range(args.max_concepts):
            if info.log['logit_scatter']['self_logits'][i]:
                self.scatter(info.log['logit_scatter']['self_logits'][i],
                             info.log['logit_scatter']['submax_logit'][i],
                             (min_, max_),
                             ('self_logits', 'submax_logit',
                              info.protocol['concepts', i] + '_final'))
                self.scatter(info.log['logit_scatter']['believed_logit'][i],
                             info.log['logit_scatter']['ref'][i],
                             (min_, max_),
                             ('believed_logit', 'ref',
                              info.protocol['concepts', i] + '_believed'))

        self.matshow(info.log['match'], 'match')

        self.init_logits()

    def init_logits(self):
        info.log['logit_scatter'] = {'self_logits': [[] for i in range(args.max_concepts)],
                                    'submax_logit': [[] for i in range(args.max_concepts)],
                                    'believed_logit': [[] for i in range(args.max_concepts)],
                                    'ref': [[] for i in range(args.max_concepts)]}
        n_concepts = len(info.vocabulary.concepts)
        info.log['match'] = np.zeros(shape=(n_concepts, n_concepts), dtype=int)

    def savefig(self, name):
        image_dir = os.path.join(args.visualize_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        plt.savefig(os.path.join(image_dir, name))

    def matshow(self, matrix, name):
        plt.matshow(matrix)
        plt.colorbar()
        if '.' not in name:
            name + '.jpg'
        self.savefig(name)
        plt.clf()

    def scatter(self, x, y, lim, names):
        plt.scatter(x, y, s=0.1)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.xlim(lim)
        plt.ylim(lim)
        self.savefig(names[2])
        plt.clf()


    ''' Utility codes '''


    def init(self):
        inited = []
        for name, param in self.named_parameters():
            if not name.startswith('resnet_model')\
                    and name not in ['temperature_', 'true_th_', 'same_class_th_']:
                inited.append(name)
                if info.new_torch:
                    init.normal_(param, 0, args.init_variance)
                else:
                    init.normal(param, 0, args.init_variance)

        pprint.pprint({'initalized parameters:': inited})
        self.new_optimizer()

    def new_optimizer(self):
        info.optimizer = optim.Adam(self.parameters(),
                                   lr=args.lr)
        info.scheduler = ReduceLROnPlateau(info.optimizer, patience=2, verbose=True)

    def save(self, name):
        torch.save({'model': self.state_dict(),
                    'optimizer': info.optimizer.state_dict(),
                    'scheduler': info.scheduler.state_dict(),
                    'protocol': (info.protocol['operations'], info.protocol['concepts'])},
                   os.path.join(args.ckpt_dir, name+'.tar'))

    def load(self, name, retrain=False):
        ckpt = torch.load(os.path.join(args.ckpt_dir, name+'.tar'))
        info.model.load_state_dict(ckpt['model'])
        if retrain:
            self.new_optimizer()
        else:
            info.optimizer.load_state_dict(ckpt['optimizer'])
            info.scheduler.load_state_dict(ckpt['scheduler'])
            for state in info.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        protocol = ckpt['protocol']
        old_protocol = deepcopy(info.protocol)
        info.protocol.reset()
        [info.protocol['operations', o] for o in protocol[0]]
        [info.protocol['concepts', c] for c in protocol[1]]
        [info.protocol['operations', o] for o in old_protocol['operations']]
        [info.protocol['concepts', c] for c in old_protocol['concepts']]
