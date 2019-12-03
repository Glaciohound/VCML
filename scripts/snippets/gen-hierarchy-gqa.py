#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen-hierarchy_gqa.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 27.07.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file builds a hierarchy from the gqa_hypernym.json file

import os
import sys
import numpy as np
from IPython.core import ultratb
from IPython import embed

from utility.common import \
    dump, load, union, all_elements, difference

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)


def build_index(all_concepts):
    index = {c: i for i, c in enumerate(all_concepts)}
    return index


def get_matrix(all_concepts, hypernym):
    print('Building concept matrix')
    num = len(all_concepts)
    index = build_index(all_concepts)
    matrix = np.zeros((num, num), dtype=int)
    for hyper, hypos in hypernym.items():
        if hyper in all_concepts:
            for hypo in hypos:
                if hypo in all_concepts:
                    matrix[index[hyper], index[hypo]] = 1
    return matrix


def build_hierarchy(matrix):
    """
    Building a hierarchy from a matrix
    """
    num = matrix.shape[0]
    parent = np.ones(num, dtype=int) * -1
    mark = np.zeros(num, dtype=int)
    rank = np.zeros(num, dtype=int)
    children = [set() for i in range(num)]

    def matrix_subs(ind):
        return matrix[ind].nonzero()[0]

    def trace_root(node):
        if parent[node] == -1:
            return node
        else:
            return trace_root(parent[node])

    def trace_down(node):
        mark[node] = 1
        for kid in matrix_subs(node):
            # sanity check
            if mark[kid] == 1:
                raise Exception('loop detected')
            # recognize and trace down
            if rank[kid] == 0:
                # new node
                rank[kid] = rank[node] + 1
                parent[kid] = node
                trace_down(kid)
            else:
                if trace_root(kid) != kid and \
                        trace_root(node) != trace_root(kid):
                    raise Exception('complex hierarchy')
                if rank[kid] > rank[node] + 1:
                    # the current node-kid link is a shortcut
                    pass
                elif rank[kid] <= rank[node]:
                    # finds a shortcut, and re-branch this kid
                    parent[kid] = node
                    diff = rank[node] + 1 - rank[kid]
                    rank[kid] = rank[node] + 1
                    rank[list(children[kid])] += diff
                else:
                    raise Exception('What a coincidence!')
            # mark all children and sub-children
            children[node].add(kid)
            children[node] = children[node].union(children[kid])
        mark[node] = 0

    for node in range(num):
        if rank[node] == 0:
            rank[node] = 1
            trace_down(node)

    return parent, rank


def match_hierarchy(all_concepts, hierarchy):
    num = len(all_concepts)
    raw = [{} for i in range(num)]
    for i, concept in enumerate(all_concepts):
        parent = hierarchy[i]
        if parent != -1:
            raw[parent][concept] = raw[i]

    output = {
        concept: raw[i]
        for i, concept in enumerate(all_concepts)
        if hierarchy[i] == -1
    }
    return output


def main(hypernym_json, hierarchy_json, forbidden_json):
    hypernym = load(hypernym_json)
    forbidden = load(forbidden_json)
    all_concepts = union(list(hypernym), all_elements(hypernym))
    all_concepts = difference(all_concepts, forbidden)
    matrix = get_matrix(all_concepts, hypernym)
    hierarchy, _ = build_hierarchy(matrix)
    concept_hierarchy = match_hierarchy(all_concepts, hierarchy)

    dump(concept_hierarchy, hierarchy_json)
    embed()


if __name__ == '__main__':
    print('Program Starts')

    root_dir = '.'
    # relative path
    hypernym_json = 'knowledge/gqa_hypernym.json'
    hierarchy_json = 'knowledge/source/gqa_hierarchy.json'
    forbidden_json = 'knowledge/gqa_forbidden.json'
    # get absolute path
    hypernym_json = os.path.join(root_dir, hypernym_json)
    hierarchy_json = os.path.join(root_dir, hierarchy_json)
    forbidden_json = os.path.join(root_dir, forbidden_json)

    # run main function
    main(hypernym_json, hierarchy_json, forbidden_json)
