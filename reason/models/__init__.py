#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : __init__.py
# Author            : Chi Han
# Email             : haanchi@gmail.com
# Date              : 19.11.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# Welcome to this little kennel of Glaciohound!


from .encoder import Encoder
from .decoder import Decoder
from .seq2seq import Seq2seq
import reason.utils.utils as utils


def get_vocab(opt):
    if opt.dataset == 'clevr':
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    vocab = utils.load_vocab(vocab_json)
    return vocab


def create_seq2seq_net(input_vocab_size, output_operation_size,
                       output_argument_size, hidden_size,
                       word_vec_dim, n_layers, bidirectional, variable_lengths,
                       use_attention, encoder_max_len, decoder_max_len, start_id,
                       end_id, word2vec_path=None, fix_embedding=False):
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len,
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional,
                      variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding)
    decoder_op = Decoder(output_operation_size, decoder_max_len,
                         word_vec_dim, hidden_size, n_layers, start_id, end_id,
                         bidirectional=bidirectional,
                         use_attention=use_attention)
    decoder_arg = Decoder(output_argument_size, decoder_max_len,
                          word_vec_dim, hidden_size, n_layers, start_id,
                          end_id, bidirectional=bidirectional,
                          use_attention=use_attention)

    return Seq2seq(encoder, decoder_op), Seq2seq(encoder, decoder_arg)
