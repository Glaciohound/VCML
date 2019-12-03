#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : embedding.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 23.07.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# This file is part of the VCML codebase
# Distributed under MIT license

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.init as init

from utility.common import detach, matrix_dict, to_normalized

from .functional import HalfGaussianConditionalLogit, logit_ln
from ...visualize import visualize_sets_v2 as visualize


class ConceptEmbedding(nn.Module):
    def __init__(self, args, tools, device, version):
        super().__init__()
        self.args = args
        self.tools = tools
        self.device = device
        self.version = version

        self.subfn_dict = matrix_dict(
            keys_x=(
                'v2.0', 'VCML', 'BERTvariant', 'NSCL'
            ),

            keys_y=(
                'get_concept_embeddings',
                'get_all_concept_embeddings',
                'get_metaconcept_net',
                'get_logit_fn',
                'get_train',
                'get_visualize',
                'get_penalty',
                'get_init',
            ),

            values=[
                [self.get_concept_embeddings,
                 lambda: self.get_all_concept_embeddings,
                 self.get_metaconcept_net_v0,
                 self.get_feasible_fn,
                 self.get_train,
                 self.get_visualize_v0,
                 self.get_penalty_metaconcept,
                 self.get_init,
                 ],

                [self.get_concept_embeddings,
                 lambda: self.get_all_concept_embeddings,
                 self.get_metaconcept_net_v1,
                 # self.get_metaconcept_net_v1_lambda,
                 self.get_feasible_fn,
                 self.get_train,
                 self.get_visualize_v1,
                 self.get_penalty_metaconcept,
                 self.get_init,
                 ],

                [self.get_bert_embeddings,
                 lambda: self.get_all_bert_embeddings,
                 self.get_metaconcept_net_nscl,
                 self.get_cos_fn,
                 self.get_bert_train,
                 self.get_visualize_bert,
                 self.get_null_penalty,
                 self.get_bert_init,
                 ],

                [self.get_concept_embeddings,
                 lambda: self.get_all_concept_embeddings,
                 self.get_metaconcept_net_nscl,
                 self.get_cos_fn,
                 self.get_train,
                 self.get_visualize_nscl,
                 self.get_penalty_metaconcept,
                 self.get_init,
                 ],
            ]
        )
        self.build()

    def build(self):
        args = self.args
        self.concept_embed_dim = args.embed_dim

        self.concept_embedding = self.subfn_dict[
            self.version, 'get_concept_embeddings'
        ]()
        self.all_concept_embeddings = self.subfn_dict[
            self.version, 'get_all_concept_embeddings'
        ]()
        self.metaconcept_net = self.subfn_dict[
            self.version, 'get_metaconcept_net'
        ]()
        self.logit_fn = self.subfn_dict[
            self.version, 'get_logit_fn'
        ]()
        self.train = self.subfn_dict[
            self.version, 'get_train'
        ]()
        self.visualize = self.subfn_dict[
            self.version, 'get_visualize'
        ]()
        self.penalty = self.subfn_dict[
            self.version, 'get_penalty'
        ]()
        self.init = self.subfn_dict[
            self.version, 'get_init'
        ]()

    def sub_net(self, in_dim, hidden_dim, out_dim):
        if hidden_dim != 0:
            net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            net = nn.Linear(in_dim, out_dim)
        return net

    def determine_relation(self, concepts1, concepts2, detach=(True, True)):
        # when some of the arguments contains only one vector
        if concepts1.dim() == 1:
            return self.determine_relation(
                concepts1[None], concepts2, detach)[0]
        elif concepts2.dim() == 1:
            return self.determine_relation(
                concepts1, concepts2[None], detach)[:, 0]

        # detaching input concepts
        if detach[0]:
            concepts1 = concepts1.detach()
        if detach[1]:
            concepts2 = concepts2.detach()

        output = self.metaconcept_net(concepts1, concepts2)

        return output

    # align a tensor with indexes
    @staticmethod
    def align(logits, is_concepts, n):
        return [
            None if i not in is_concepts
            else logits[:, is_concepts.index(i)]
            for i in range(n)
        ]

    # main function
    def calculate_logits(self, objects, program_indexes):
        if objects is None:
            return None, None

        # filter out concept-type arguments
        is_concepts = [
            i for i, index in enumerate(program_indexes)
            if self.tools.arguments_in_concepts[index] != -1
        ]
        length = program_indexes.shape[0]
        concept_program_indexes = \
            self.tools.arguments_in_concepts[
                detach(program_indexes[is_concepts])
            ]
        concept_program_indexes = torch.LongTensor(concept_program_indexes)\
            .to(self.device)

        # calculating the raw similarity logits
        concepts_used = self.concept_embedding(
            concept_program_indexes)

        logits = self.logit_fn(objects, concepts_used)
        logits = self.align(logits, is_concepts, length)

        return logits

    def get_concept_embeddings(self):
        return nn.Embedding(
            self.tools.n_concepts, self.concept_embed_dim
        )

    def get_bert_embeddings(self):
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        _, self.bert_dim = \
            self.bert.embeddings.position_embeddings.weight.shape
        self.bert_mlp = self.sub_net(self.bert_dim, 0, self.args.embed_dim)
        return self.bert_embed

    def bert_embed(self, concepts_indexes):
        shape = concepts_indexes.shape
        flattened = concepts_indexes.flatten()
        concepts = [self.tools.concepts[int(index)] for index in flattened]
        embeddings = [self.bert_embed_one(concept) for concept in concepts]
        reshaped = torch.stack(embeddings).reshape(tuple(shape) + (-1,))
        return reshaped

    def bert_embed_one(self, phrase):
        tokens = self.tokenizer.encode(phrase)
        tensor = torch.LongTensor(tokens).to(self.device)
        bert_attention, last_state = self.bert(tensor[None])
        output = self.bert_mlp(last_state)[0]
        return output

    def get_metaconcept_net_v0(self):
        self.metaconcept_subnet = self.sub_net(
            4 * self.concept_embed_dim,
            self.args.metaconcept_hidden_dim, 5
        )
        self.transform_matrix = nn.Parameter(
            torch.zeros(self.concept_embed_dim, self.concept_embed_dim)
        )
        return self.relation_net_v0

    def get_metaconcept_net_v1(self):
        self.metaconcept_subnet = self.sub_net(
            3, self.args.metaconcept_hidden_dim, 5
        )
        return self.relation_net_v1

    def get_metaconcept_net_v1_lambda(self):
        self.metaconcept_subnet = self.sub_net(
            1, self.args.metaconcept_hidden_dim, 5
        )
        return self.relation_net_v1_lambda

    def get_metaconcept_net_nscl(self):
        self.metaconcept_subnet = self.sub_net(
            4 * self.args.embed_dim,
            self.args.metaconcept_hidden_dim, 5
        )
        return self.relation_net_nscl

    def get_feasible_fn(self):
        return HalfGaussianConditionalLogit(
            self.args.sample_size, 400, self.device, slack=False)

    def get_cos_fn(self):
        '''
        self.offset = nn.Parameter(torch.tensor(0.))
        self.tau = nn.Parameter(torch.tensor(1.))
        '''
        self.offset = 0.15
        self.tau = (1 - self.offset) / 4
        return self.cos_fn

    def cos_fn(self, embeddings1, embeddings2):
        cos = torch.matmul(
            to_normalized(embeddings1),
            to_normalized(embeddings2).t()
        )
        return (cos - self.offset) / self.tau

    def get_all_concept_embeddings(self):
        return self.concept_embedding.weight

    def get_all_bert_embeddings(self):
        all_indexes = torch.LongTensor(
            self.tools.concepts.indexes()).to(self.device)
        return self.bert_embed(all_indexes)

    def get_init(self):
        return self.normal_init

    def get_bert_init(self):
        return self.bert_init

    def get_visualize_v0(self):
        return self.visualize_v0

    def get_visualize_v1(self):
        return self.visualize_v1

    def get_visualize_bert(self):
        return self.visualize_bert

    def get_visualize_nscl(self):
        return self.visualize_nscl

    def get_penalty_metaconcept(self):
        return self.penalty_metaconcept

    def get_null_penalty(self):
        return self.null_penalty

    def get_train(self):
        return self.normal_train

    def get_bert_train(self):
        return self.bert_train

    def relation_net_v0(self, concepts1, concepts2):
        '''
        concepts1 = torch.matmul(concepts1, self.transform_matrix)
        concepts2 = torch.matmul(concepts2, self.transform_matrix)
        batch1 = concepts1.shape[0]
        batch2 = concepts2.shape[0]
        dim = concepts1.shape[1]

        norm1 = concepts1.pow(2).sum(1).sqrt()
        norm2 = concepts2.pow(2).sum(1).sqrt()
        tensor = torch.stack([
            concepts1[:, None, :].repeat(1, batch2, 1),
            norm1[:, None, None].repeat(1, batch2, dim),
            concepts2[None, :, :].repeat(batch1, 1, 1),
            norm2[None, :, None].repeat(batch1, 1, dim),
        ], dim=3)

        output = self.metaconcept_subnet(tensor).max(2)[0]
        '''

        # Use a NS-CL -like metaconcept net
        c1 = concepts1[:, None]
        c2 = concepts2[None]
        n1 = concepts1.shape[0]
        n2 = concepts2.shape[0]
        abdp = torch.cat([c1.repeat(1, n2, 1),
                          c2.repeat(n1, 1, 1),
                          c1 - c2,
                          c1 * c2],
                         dim=-1)
        return self.metaconcept_subnet(abdp)

    def relation_net_v1(self, concepts1, concepts2):
        A_to_B = self.logit_fn(concepts1, concepts2)
        B_to_A = self.logit_fn(concepts2, concepts1).t()
        logit_lambda = logit_ln(self.logit_fn.ln_lambda(
            concepts1, concepts2
        ))

        tensor = torch.stack([A_to_B, B_to_A, logit_lambda], dim=2)

        output = self.metaconcept_subnet(tensor)
        return output

    def relation_net_v1_lambda(self, concepts1, concepts2):
        logit_lambda = logit_ln(self.logit_fn.ln_lambda(
            concepts1, concepts2
        ))

        tensor = logit_lambda[:, :, None]

        output = self.metaconcept_subnet(tensor)
        return output

    def relation_net_nscl(self, concepts1, concepts2):
        c1 = to_normalized(concepts1)[:, None]
        c2 = to_normalized(concepts2)[None]
        n1 = concepts1.shape[0]
        n2 = concepts2.shape[0]
        abdp = torch.cat([c1.repeat(1, n2, 1),
                          c2.repeat(n1, 1, 1),
                          c1 - c2,
                          c1 * c2],
                         dim=-1)
        return self.metaconcept_subnet(abdp)

    def get_embedding(self, category, name):
        if category == 'concept':
            index = torch.LongTensor(
                [self.tools.concepts[name]]).to(self.device)
            return self.concept_embedding(index)[0]
        else:
            raise Exception(f'no embedding for {category}-{name} can be found')
        '''
        elif category == 'attribute':
            index = torch.LongTensor(
                [self.tools.attributes[name]]).to(self.device)
            return self.attribute_embedding(index)[0]
        '''

    def normal_init(self):
        for name, param in self.named_parameters():
            try:
                init.kaiming_normal_(param)
            except Exception:
                init.normal_(param, 0, self.args.init_variance)

    def bert_init(self):
        self.bert.eval()
        for name, param in self.named_parameters():
            if not name.startswith('bert.'):
                try:
                    init.kaiming_normal_(param)
                except Exception:
                    init.normal_(param, 0, self.args.init_variance)
            else:
                param.requires_grad_(False)

    def normal_train(self, mode=True):
        super().train(mode=mode)

    def bert_train(self, mode=True):
        super().train(mode=mode)
        self.bert.eval()

    def visualize_v0(self, path, plt):
        pass

    def visualize_v1(self, path, plt):
        visualize.concept_length(self, path, plt)
        return
        visualize.probability_matrix(self, path, plt)
        visualize.intercosine_matrix(self, path, plt)
        visualize.pca_embeddings(
            self, path, plt,
            ['cube', 'gray', 'yellow', 'red', 'purple'],
            with_origin=True
        )
        visualize.cosmat_samekind(self, path, plt)
        visualize.cosmat_synonym(self, path, plt)

    def visualize_bert(self, path, plt):
        pass

    def visualize_nscl(self, path, plt):
        return
        visualize.intercosine_matrix(self, path, plt)
        visualize.pca_embeddings(
            self, path, plt,
            ['cube', 'gray', 'yellow', 'red', 'purple'],
            with_origin=True
        )
        visualize.cosmat_samekind(self, path, plt)
        visualize.cosmat_synonym(self, path, plt)

    def penalty_metaconcept(self):
        net = self.metaconcept_subnet
        penalty = 0
        for name, param in net.named_parameters():
            penalty = penalty + param.pow(2).sum()
        output = penalty * self.args.penalty
        return output

    def penalty_length(self):
        length_sum = self.all_concept_embeddings().pow(2).sum()
        output = length_sum * self.args.penalty
        return output

    def null_penalty(self):
        return 0
