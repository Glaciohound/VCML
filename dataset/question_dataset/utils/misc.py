#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : misc.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 22.07.2019
# Last Modified Date: 31.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file contains several miscellaneous utility functions for building and
# using question datasets.

import copy

from utility.common import \
    contains, random_one, load_knowledge, belongs_to, union, \
    all_in_hierarchy
from inflect import engine as plural_engine
engine = plural_engine()
engine.classical(all=True)


def int_or(x, y):
    """
    used for getting or-values in binary classification
    """
    return 1 - (1-x) * (1-y)


def cub_proper_group_name(name, in_plural=False):
    """
    This function is only for CUB dataset up to now.
    Given a CUB group name (in an unknown hierarchy), produce a proper
    language name.
    """
    special_dict = {
        'Le_Conte_Sparrow': 'LeConte\'s Sparrow',
        'Forsters_Tern': 'Forster\'s Tern',
    }

    if name in special_dict:
        proper_name = special_dict[name]
    else:
        words = name.split('_')
        x_ed = None
        for i, piece in enumerate(words[1:3]):
            if piece.endswith('ed'):
                x_ed = i + 1
        if x_ed is not None:
            words = \
                words[0:x_ed-1] + \
                ['-'.join(words[x_ed-1:x_ed+1])] + \
                words[x_ed+1:]
        proper_name = ' '.join(words)

    if in_plural:
        proper_name = plural(proper_name)
    proper_name = proper_name.capitalize()

    return proper_name


def cub_proper_body_name(body_concept, in_plural=False):
    """
    This function is only for CUB dataset up to now.
    Given a body-concept, e.g. <has_bill_shape::needle>, it returns a proper
    name phrase, like 'needle-shaped bill' in this case.

    CUB has the following body parts:
    ['bill', 'wing', 'upperparts', 'back',
    'tail', 'upper tail', 'head', 'breast', 'throat', 'eye',
    'forehead', 'nape', 'belly' ,'leg', 'crown']

    and the following attributes regarding the body parts:
    ['shape', 'color', 'pattern'].

    Also CUB contains some body-level attributes:
    ['size', 'shape', 'primary color']
    in which case we use 'body' as the part name
    """
    attribute, adj = body_concept.split('::')

    # parse bodies and types
    part = ' '.join(attribute.split('_')[1:-1])
    if part in ['', 'primary']:
        part = 'body'
    # attr_type = attribute.split('_')[-1]
    if in_plural:
        part = plural(part)

    adj_plain = adj.replace('_', ' ')
    # determine if the adj willed be phrased in a 'which are ...' manner
    # cases for this include:
    # (1) comparing descriptions (e.g. 'longer than head')
    # (2) detailed descriptions with parenthesis (e.g. 'large (16 - 32 in)')
    use_which_are = \
        '(' in adj_plain or \
        contains(adj_plain, ['same', 'than'])

    # some times, the adj part will include redundant body part nouns, like
    # 'has_tail_shape::forked_tail', so we should correct them in these cases.
    if part in adj_plain:
        if '-' in adj_plain:
            adj_plain = adj_plain.split('-')[0]
        else:
            adj_plain = adj_plain.split(' ')[0]

    if use_which_are:
        output = f'{part} which are {adj_plain}'
    else:
        output = f'{adj_plain} {part}'

    return output


def plural(name):
    """
    This function returns the plural of the given word / phrase
    All higher (than genera) categories are already plural in Latin/Greek,
    however, this may not be recognized by inflect.engine which mainly
    supports English words. The main problem occurs in Family groups, which,
    however, almost always ends with 'ae', so can be easily checked.
    """
    if is_plural(name):
        # the word is already in plural form
        output = name
    else:
        # the word is in singular form
        output = engine.plural_noun(name)
    return output


def singular(name):
    """
    This function returns the singular form of the given word
    """
    if is_singular(name):
        # the word is already in singular form
        output = name
    else:
        output = engine.singular_noun(name)
    return output


def is_singular(name):
    output = not is_plural(name)
    return output


def is_plural(name):
    output = engine.singular_noun(name) is not False or \
        is_greek_or_latin(name)
    return output


def is_greek_or_latin(name):
    """
    This function determins if one name is of latin or greek origin.
    Note that this function only applies to concepts in CUB dataset,
    and should not be regarded as a universal judgement
    """
    if name.endswith('dae') or name.endswith('formes'):
        _is = True
    else:
        _is = False
    return _is


def exist_checktable(all_concepts, args, logger):
    """
    This function returns a look-up table for determining the
    entailment and mutual-exclusion among concepts.
    """
    synonym_stats = load_knowledge(args.task, 'synonym', logger)
    isinstanceof_stats = load_knowledge(args.task, 'isinstanceof', logger)
    hierarchy = load_knowledge(
        args.task, 'hierarchy', logger, from_source=True)

    all_concepts = set(all_concepts)
    cues = {
        concept: {
            True: set([concept]),
            False: set(),
        }
        for concept in all_concepts
    }
    results = copy.deepcopy(cues)

    # Dealing with synonyms first
    if synonym_stats is not None:
        logger('Dealing with synonyms first')
        if isinstanceof_stats is not None:
            logger('expand the isinstanceof stats', resume=True)

        ambiguous = 0
        for examplar, synset in synonym_stats.items():
            group = set(synset)
            group.add(examplar)
            for x in group:
                for y in group:
                    if x in all_concepts:
                        cues[x][True].add(y)
                    if y in all_concepts:
                        results[y][True].add(x)

            # expanding isinstanceof knowledge
            if isinstanceof_stats is not None:
                categories = set([belongs_to(isinstanceof_stats, name)
                                  for name in group])
                if None in categories:
                    categories.remove(None)
                if len(categories) == 1:
                    cat = list(categories)[0]
                    isinstanceof_stats[cat] = \
                        union(isinstanceof_stats[cat], group)
                else:
                    ambiguous += 1
        logger(f'{ambiguous} out of {len(synonym_stats)} synsets '
               'are ambiguous', resume=True)

    # Dealing with hierarchy information then, by walking through the forest
    trace_line = set()

    def trace_down(forest):
        # tracing down the current tree
        for sub_root, sub_forest in forest.items():
            trace_line.add(sub_root)
            for hyper in trace_line:
                if hyper in all_concepts:
                    cues[hyper][True].add(sub_root)
                if sub_root in all_concepts:
                    results[sub_root][True].add(hyper)
            if sub_forest is not None:
                trace_down(sub_forest)
            trace_line.remove(sub_root)

    if hierarchy is not None:
        logger('Dealing with hierarchy information')
        if isinstanceof_stats is not None:
            logger('expand the isinstanceof_stats', resume=True)

        ambiguous = 0
        for root, forest in hierarchy.items():
            trace_down({root: forest})
            # expand the isinstanceof stats
            if isinstanceof_stats is not None:
                all_nodes = all_in_hierarchy({root: forest})
                categories = set([belongs_to(isinstanceof_stats, name)
                                  for name in all_nodes])

                if None in categories:
                    categories.remove(None)
                if len(categories) == 1:
                    cat = list(categories)[0]
                    isinstanceof_stats[cat] = \
                        union(isinstanceof_stats[cat], all_nodes)
                else:
                    ambiguous += 1

        logger(f'{ambiguous} out of {len(hierarchy)} hierarchies '
               'are ambiguous', resume=True)

    # Dealing with the isinstanceof information.
    if isinstanceof_stats is not None:
        logger('Dealing with the isinstanceof information')
        for group in isinstanceof_stats.values():
            for x in group:
                if x in all_concepts:
                    for y in group:
                        if y in all_concepts:
                            if y not in cues[x][True] and \
                                    y not in results[x][True]:
                                cues[x][False].add(y)
                                results[y][False].add(x)

    return {'cues': cues, 'results': results}


# Utility functions specifically for visual tasks


def alternative(concepts, synonyms):
    if isinstance(concepts, str):
        return random_one(synonyms.get(concepts, [concepts]))

    elif isinstance(concepts, tuple):
        return tuple([alternative(x, synonyms) for x in concepts])

    elif isinstance(concepts, list):
        return [alternative(x, synonyms) for x in concepts]

    else:
        raise Exception('something wrong')


def filter_objects(scene, queried):
    which = ['-']
    answer = 'no'

    for obj_id, obj in scene['objects'].items():
        for at in obj.values():
            if (isinstance(at, str) and at == queried)\
                    or (isinstance(at, list) and queried in at):
                if '-' in which:
                    which.remove('-')
                which.append(obj_id)
                answer = 'yes'

    return which, answer
