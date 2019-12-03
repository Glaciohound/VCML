#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : temp.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 07.08.2019
# Last Modified Date: 14.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license

import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utility.common import make_parent_dir, detach


def tsne_hierarchy(embeddings, concepts, test_concepts, tree,
                   filename, sub_folder, path, plt):
    tsne = TSNE(n_components=2, perplexity=30,
                n_iter=1000000000, learning_rate=200)
    embeddings = np.stack(detach(embeddings))
    result = tsne.fit_transform(embeddings)
    result_dict = dict(zip(concepts, result))

    fig, ax = plt.subplots()

    def trace_down(root, forest):
        root_xy = result_dict[root]
        ax.scatter(*root_xy, c='r' if root in test_concepts else 'b')
        ax.annotate(root, xy=tuple(root_xy))

        if len(forest) != 0:
            for kid, sub_forest in forest.items():
                kid_xy = result_dict[kid]
                ax.arrow(*root_xy, *(kid_xy - root_xy),
                         head_width=0)

                if isinstance(sub_forest, dict):
                    trace_down(kid, sub_forest)

    trace_down(list(tree.keys())[0], list(tree.values())[0])

    savefig(fig, filename, sub_folder, path, plt)
    plt.close('all')


def PCA_hierarchy(embeddings, concepts, test_concepts, tree,
                  filename, sub_folder, path, plt):
    pca = PCA(n_components=2)
    embeddings = detach(embeddings)
    with_zero = np.concatenate([
        embeddings,
        np.zeros_like(embeddings[0])[None]
    ], axis=0)
    result = pca.fit_transform(with_zero)
    result_dict = dict(zip(list(concepts) + ['root'], result))

    fig, ax = plt.subplots()

    def trace_down(root, forest):
        root_xy = result_dict[root]
        ax.scatter(*root_xy, c='r' if root in test_concepts else 'b')
        ax.annotate(root, xy=tuple(root_xy))

        if len(forest) != 0:
            for kid, sub_forest in forest.items():
                kid_xy = result_dict[kid]
                ax.arrow(*root_xy, *(kid_xy - root_xy),
                         head_width=0)

                if isinstance(sub_forest, dict):
                    trace_down(kid, sub_forest)

    trace_down('root', tree)

    savefig(fig, filename, sub_folder, path, plt)
    plt.close('all')


def savefig(fig, name, sub_folder, path, plt):
    fig.set_size_inches(10, 8)
    filename = get_filename(name, sub_folder, path)
    make_parent_dir(filename)
    fig.tight_layout()
    plt.savefig(filename)


def get_filename(name, sub_folder, path):
    sub_folder = str(sub_folder)
    if sub_folder is not None:
        path = os.path.join(path, sub_folder)
    return os.path.join(path, f'{name}.jpg')
