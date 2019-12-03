#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-synonyms.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/20/2019
#
# Distributed under terms of the MIT license.

import jacinle.io as io
from nltk.corpus import wordnet as wn


def get_synonyms_and_antonyms(word):
    synonyms = []
    antonyms = []

    if ' ' in word:
        return synonyms, antonyms

    try:
        # NB(Jiayuan Mao @ 05/20): use only the noun meaning.
        for syn in [wn.synset(word + '.n.01')]:
        # for syn in wn.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
    except:
        pass

    synonyms.append(word)

    def f(word):
        return '_' not in word and '-' not in word

    def g(word_list):
        return list(set(filter(f, word_list)))

    return map(g, [synonyms, antonyms])


def main():
    io.set_fs_verbose(True)
    concepts = io.load('./gqa_concepts.json')
    synonyms = dict()

    for dataset, word2freq in concepts.items():
        this_synonyms = dict()
        for word in word2freq:
            syn, ant = get_synonyms_and_antonyms(word)
            if len(syn) > 1:
                this_synonyms[word] = syn

        synonyms[dataset] = this_synonyms

    io.dump('./gqa_synonyms.json', synonyms, compressed=False)


if __name__ == '__main__':
    main()

