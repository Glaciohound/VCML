#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualize_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 01.08.2019
# Last Modified Date: 26.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


"""
Codes for visualizing
"""

import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utility.common \
    import make_parent_dir, to_normalized, matmul, detach


def concatenate_origin(names, embeddings):
    new_names = ['origin'] + names
    new_embeddings = np.concatenate(
        [embeddings[0, None] * 0, embeddings],
        axis=0
    )
    return new_names, new_embeddings


def pca_embeddings(names, embeddings, filename, path, plt, with_origin):
    pca = PCA(n_components=2)
    embeddings = detach(embeddings)
    if with_origin:
        names, embeddings = concatenate_origin(names, embeddings)
    result = pca.fit_transform(embeddings)

    fig, ax = plt.subplots()
    ax.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(names):
        ax.annotate(word, xy=tuple(result[i]))

    savefig(fig, filename, path, plt)
    plt.close('all')


def tsne_embeddings(names, embeddings, filename, path, plt, with_origin):
    tsne = TSNE(n_components=2, perplexity=10, n_iter=1000000)
    embeddings = detach(embeddings)
    if with_origin:
        names, embeddings = concatenate_origin(names, embeddings)
    result = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots()
    ax.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(names):
        ax.annotate(word, xy=tuple(result[i]))

    savefig(fig, filename, path, plt)
    plt.close('all')


def inter_cosine(embedding1_tuple, embedding2_tuple,
                 name, path, plt,
                 vmin=None, vmax=None, show_value=False):
    embedding1 = to_normalized(detach(embedding1_tuple[1]))
    embedding2 = to_normalized(detach(embedding2_tuple[1]))
    cos_matrix = matmul(embedding1, embedding2.transpose(1, 0))

    matshow(
        cos_matrix,
        name, path, plt,
        embedding1_tuple[0], embedding2_tuple[0],
        vmin, vmax, show_value
    )


"""
Below are utilities
"""


def bar_graph(values, labels, name, path, plt):
    values = detach(values)
    fig, ax = plt.subplots()
    y_pos = np.arange(len(values))
    ax.bar(y_pos, values, align='center', alpha=0.5)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(labels)

    savefig(fig, name, path, plt)
    plt.close('all')


def matshow(matrix, name, path, plt,
            ylabel=None, xlabel=None,
            vmin=None, vmax=None, show_value=False):
    if isinstance(matrix, torch.Tensor):
        matrix = detach(matrix)

    fig, ax = plt.subplots()
    im = ax.matshow(matrix, vmin=vmin, vmax=vmax)

    if xlabel is not None:
        ax.set_xticks(np.arange(len(xlabel)))
        ax.set_yticks(np.arange(len(ylabel)))
        ax.set_xticklabels(xlabel)
        ax.set_yticklabels(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
                 rotation_mode="anchor")

    if show_value:
        for (i, j), v in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(v),
                    fontsize=6, ha='center', va='center')

    fig.colorbar(im)

    savefig(fig, name, path, plt)
    plt.clf()


def savefig(fig, name, path, plt):
    fig.set_size_inches(10, 8)
    filename = get_filename(name, path)
    make_parent_dir(filename)
    fig.tight_layout()
    plt.savefig(filename)


def get_filename(name, path):
    return os.path.join(path, 'images', f'{name}.jpg')
