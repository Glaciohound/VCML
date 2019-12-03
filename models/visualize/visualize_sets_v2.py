#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualize_sets_v2.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 24.07.2019
# Last Modified Date: 26.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license

"""
Visualization tools for model V2
"""

import torch.nn.functional as F

from . import visualize_utils as visualize


def get_embedding_dict(embedding):
    concepts = embedding.tools.concepts
    vectors = embedding.all_concept_embeddings()
    return concepts, vectors


def tsne_embeddings(embedding, path, plt,
                    selected_concepts=None, with_origin=False):
    concepts, vectors = get_embedding_dict(embedding)
    if selected_concepts is not None:
        indexes = [concepts[name] for name in selected_concepts]
        vectors_to_visualize = vectors[indexes]
        concepts_to_visualize = selected_concepts
    else:
        vectors_to_visualize = vectors
        concepts_to_visualize = concepts

    visualize.tsne_embeddings(
        concepts_to_visualize,
        vectors_to_visualize,
        'concepts_t-SNE', path, plt,
        with_origin,
    )


def pca_embeddings(embedding, path, plt,
                   selected_concepts=None, with_origin=False):
    concepts, vectors = get_embedding_dict(embedding)
    if selected_concepts is not None:
        indexes = [concepts[name] for name in selected_concepts]
        vectors_to_visualize = vectors[indexes]
        concepts_to_visualize = selected_concepts
    else:
        vectors_to_visualize = vectors
        concepts_to_visualize = concepts

    visualize.pca_embeddings(
        concepts_to_visualize,
        vectors_to_visualize,
        'concepts_PCA', path, plt,
        with_origin,
    )


def intercosine_matrix(embedding, path, plt):
    concepts, vectors = get_embedding_dict(embedding)

    visualize.inter_cosine(
        (concepts, vectors),
        (concepts, vectors),
        'intercosine', path, plt,
        vmin=-0.5,
        vmax=1,
        show_value=True,
    )


def probability_matrix(embedding, path, plt):
    concepts, vectors = get_embedding_dict(embedding)

    visualize.matshow(
        embedding.logit_fn(
            vectors,
            vectors,
        ),
        'entailment-logit', path, plt,
        concepts,
        concepts,
        -10, 10,
        show_value=True,
    )

    visualize.matshow(
        embedding.logit_fn.ln_lambda(
            vectors,
            vectors,
        ),
        'ln_lambda', path, plt,
        concepts,
        concepts,
        -5, 5,
        show_value=True,
    )


def concept_length(embedding, path, plt):
    concepts, vectors = get_embedding_dict(embedding)
    visualize.bar_graph(
        vectors.pow(2).sum(-1).sqrt(),
        concepts,
        'concept-length', path, plt,
    )


def cosmat_synonym(embedding, path, plt):
    concepts, vectors = get_embedding_dict(embedding)

    visualize.matshow(
        embedding.determine_relation(
            vectors,
            vectors,
        )[:, :, 0],
        'synonym_matrix', path, plt,
        concepts,
        concepts,
        -10, 10,
        show_value=True,
    )


def cosmat_isinstanceof(embedding, path, plt):
    concepts, vectors = get_embedding_dict(embedding)

    visualize.matshow(
        F.softmax(embedding.determine_relation(
            vectors,
            vectors,
        )[:, :, 1], 1),
        'isinstanceof_matrix', path, plt,
        concepts,
        concepts,
        -10, 10,
        show_value=True,
    )


def cosmat_samekind(embedding, path, plt):
    concepts, vectors = get_embedding_dict(embedding)

    visualize.matshow(
        embedding.determine_relation(
            vectors,
            vectors,
        )[:, :, 3],
        'samekind_matrix', path, plt,
        concepts,
        concepts,
        -10, 10,
        show_value=True,
    )
