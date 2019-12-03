#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : knowledge_utils.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 08.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# utilities about knowledge


from utility.common import belongs_to, union


def all_concepts(stats, requirement_on_key=None):
    keys = list(stats.keys())
    if requirement_on_key is not None:
        keys = list(filter(requirement_on_key, keys))

    elements_gather = [e for k in keys for e in stats[k]]

    output = list(set(elements_gather + keys))
    return output


def all_elements(stats, requirement_on_key=None):
    keys = list(stats.keys())
    if requirement_on_key is not None:
        keys = list(filter(requirement_on_key, keys))

    elements_gather = [e for k in keys for e in stats[k]]

    output = list(set(elements_gather))
    return output


def all_secondary_elements(
        stats, requirement_on_key=None, with_keys=False):
    keys = list(stats.keys())
    if requirement_on_key is not None:
        keys = list(filter(requirement_on_key, keys))

    elements_gather = [e for k in keys
                       for group in stats[k].values()
                       for e in group
                       ]

    output = list(set(elements_gather))
    if with_keys:
        output = union(output, keys)

    return output


def all_in_hierarchy(hierarchy):
    output = set()

    def trace_down(forest):
        for sub_root, sub_forest in forest.items():
            output.add(sub_root)
            if sub_forest is not None and len(sub_forest) > 0:
                trace_down(sub_forest)
    trace_down(hierarchy)
    output = list(output)
    return output


def filter_stats(stats, all_concepts):
    output = {
        key: [
            v for v in values
            if v in all_concepts
        ]
        for key, values in stats.items()
    }
    return output


def is_synonym(concept1, concept2, synonym_stats):
    output = belongs_to(synonym_stats, concept1) == \
        belongs_to(synonym_stats, concept2) and \
        belongs_to(synonym_stats, concept1) is not None
    return output


def is_instance_of(concept1, concept2,
                   isinstanceof_stats, synonym_stats):
    synset = get_synset(concept1, synonym_stats)
    output = False
    for concept in synset:
        if concept in isinstanceof_stats.get(concept2, []):
            output = True
    return output


def is_hypernym(concept2, concept1, hypernym_stats, synonym_stats):
    synset1 = get_synset(concept1, synonym_stats)
    synset2 = get_synset(concept2, synonym_stats)
    output = False
    for syn1 in synset1:
        for syn2 in synset2:
            if syn1 in hypernym_stats.get(syn2, []):
                output = True
    return output


def is_samekind(concept1, concept2,
                isinstanceof_stats, synonym_stats):
    synset1 = get_synset(concept1, synonym_stats)
    synset2 = get_synset(concept2, synonym_stats)
    for syn1 in synset1:
        if belongs_to(isinstanceof_stats, syn1) is not None:
            for syn2 in synset2:
                if belongs_to(isinstanceof_stats, syn2) is not None:
                    return belongs_to(isinstanceof_stats, syn1) == \
                        belongs_to(isinstanceof_stats, syn2)
    return None


def is_composed_of(concept2, concept1, meronym_stats):
    if concept2 in meronym_stats:
        if concept1 in meronym_stats[concept2]['true']:
            output = True
        elif concept1 in meronym_stats[concept2]['false']:
            output = False
        else:
            output = None
    else:
        output = None
    return output


def get_synset(concept, synonym_stats):
    if synonym_stats is not None and \
            belongs_to(synonym_stats, concept) is not None:
        synset = set(synonym_stats[belongs_to(
            synonym_stats, concept)])
    else:
        synset = set()
    synset.add(concept)
    return synset
