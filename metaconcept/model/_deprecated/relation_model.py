#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init

class RelationModel(nn.Module):
    def __init__(self, args, info=None):
        super(RelationModel, self).__init__()
        self.args = args
        self.info = info
        self.build()
        #self.init()

    def build(self):
        args = self.args
        self.attribute_embeddingIn = nn.Embedding(args.max_concepts, args.embed_dim)
        self.attribute_embeddingOut = nn.Embedding(args.max_concepts, args.embed_dim)
        self.attribute_embeddingId = nn.Embedding(args.max_concepts, args.identity_dim)
        self.feature_linearIn = nn.Linear(args.feature_dim, args.embed_dim)
        self.feature_linearOut = nn.Linear(args.feature_dim, args.embed_dim)
        self.feature_linearId = nn.Linear(args.feature_dim, args.identity_dim)
        self.concept_embeddingIn = nn.Embedding(args.max_concepts, args.embed_dim)
        self.concept_embeddingOut = nn.Embedding(args.max_concepts, args.embed_dim)
        self.concept_embeddingId = nn.Embedding(args.max_concepts, args.identity_dim)

        self.embed_init = nn.Parameter(torch.randn(args.embed_dim))
        self.identity_init = nn.Parameter(torch.randn(args.identity_dim))
        self.attention_init = nn.Parameter(torch.zeros(args.attention_dim), requires_grad=False)

        def build_mlp(dim_in, dim_hidden, dim_out, name):
            linear1 = nn.Linear(dim_in, dim_hidden)
            linear2 = nn.Linear(dim_hidden, dim_out)
            setattr(self, name+'_linear1', linear1)
            setattr(self, name+'_linear2', linear2)
            return lambda x: linear2(torch.relu(linear1(x)))

        self.axon_mlp = build_mlp(args.attention_dim+args.identity_dim if not args.identity_only\
                                  else args.identity_dim,
                                  args.hidden_dim,
                                  args.attention_dim,
                                  'axon')

        self.meta_mlp = build_mlp(args.embed_dim+args.attention_dim,
                                  args.hidden_dim,
                                  args.attention_dim,
                                  'meta')

    def init(self):
        for param in self.parameters():
            if param.requires_grad:
                init.normal_(param, 0.5, 0.1)

    def forward(self, data):
        args = self.args
        info = self.info
        batch_size = data.answer.shape[0]
        num_objects = max([data.scene[i].shape[0] for i in range(batch_size)])
        dim_concept = num_objects + args.max_concepts

        is_mode = (data.program[:, :, 0] == info.protocol['operations', 'mode']).astype(int)
        is_insert = (data.program[:, :, 0] == info.protocol['operations', 'insert']).astype(int)
        is_transfer = (data.program[:, :, 0] == info.protocol['operations', 'transfer']).astype(int)

        def embed_without_bg(embedding, x):
            non_bg = (x != -1).astype(int)
            x = torch.LongTensor(x+1).to(info.device)
            non_bg = torch.Tensor(non_bg).to(info.device)
            return (embedding(x) * non_bg[:,:,None]).sum(1)

        objectsIn = self.embed_init[None, None].repeat((batch_size, num_objects, 1))
        objectsOut = self.embed_init[None, None].repeat((batch_size, num_objects, 1))
        objectsId = self.identity_init[None, None].repeat((batch_size, num_objects, 1))
        for i in range(batch_size):
            num_here = data.scene[i].shape[0]
            if data.scene.dtype.name == 'int64':
                objectsIn[i, :num_here] =\
                    embed_without_bg(self.attribute_embeddingIn, data.scene[i])
                objectsOut[i, :num_here] =\
                    embed_without_bg(self.attribute_embeddingOut, data.scene[i])
                objectsId[i, :num_here] =\
                    embed_without_bg(self.attribute_embeddingId, data.scene[i])
            else:
                objectsIn[i, :num_here] = self.feature_linearIn(
                    torch.Tensor(data.scene[i]).to(info.device))
                objectsOut[i, :num_here] = self.feature_linearOut(
                    torch.Tensor(data.scene[i]).to(info.device))
                objectsId[i, :num_here] = self.feature_linearId(
                    torch.Tensor(data.scene[i]).to(info.device))

        dendron = torch.cat((self.concept_embeddingIn.weight[None].repeat((batch_size, 1, 1)), objectsIn), 1)
        axon = torch.cat((self.concept_embeddingOut.weight[None].repeat((batch_size, 1, 1)), objectsOut), 1)
        weight_matrix = torch.matmul(dendron, axon.transpose(2, 1))
        identity = torch.cat((self.concept_embeddingId.weight[None].repeat((batch_size, 1, 1)), objectsId), 1)

        meta = self.attention_init[None].repeat((batch_size, 1))
        attention = self.attention_init[None, None].repeat((batch_size, dim_concept, 1))

        arguments = self.concept_embeddingOut(torch.LongTensor(data.program[:, :, 1]).to(info.device))
        max_program_length = data.program.shape[1]
        history = []

        for i in range(max_program_length):
            is_mode_ = torch.Tensor(is_mode[:, i, None]).to(info.device)
            is_insert_ = torch.Tensor(is_insert[:, i, None, None]).to(info.device)
            is_transfer_ = torch.Tensor(is_transfer[:, i, None, None]).to(info.device)

            new_meta = self.meta_mlp(torch.cat((meta, arguments[:, i]), 1))
            meta = is_mode_ * new_meta + (1-is_mode_) * meta
            meta_broadcast = meta[:, None].repeat((1, dim_concept, 1))
            attention_insert = meta_broadcast * (arguments[:, i, None] * axon).mean(2)[:, :, None]

            attended_weight = weight_matrix * attention.mean(2)[:, None]

            if not args.identity_only:
                identity_attention = torch.cat([identity, attention], 2)
                gathered = (attended_weight[:, :, :, None] * identity_attention[:, None]).mean(2)
                attention_transfer = self.axon_mlp(gathered) + gathered[:, :, args.identity_dim:]
            else:
                gathered = (attended_weight[:, :, :, None] * identity[:, None]).mean(2)
                attention_transfer = self.axon_mlp(gathered)


            attention = attention + is_insert_ * attention_insert + is_transfer_ * attention_transfer
            #attention = (1-is_insert_-is_transfer_) * attention + is_insert_ * attention_insert + is_transfer_ * attention_transfer
            attention = torch.clamp(F.leaky_relu(attention), -1, 2)
            history.append({'attention': attention.data, 'insert': attention_insert.data,
                            'transfer': attention_transfer.data})

        history = {k: torch.stack([item[k] for item in history]) for k in history[0].keys()}

        output_length = attention.mean(2)
        output_softmax = F.log_softmax(output_length, 1)
        return output_softmax, history
