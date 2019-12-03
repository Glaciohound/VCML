#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen-hypernym-cub.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 17.07.2019
# Last Modified Date: 12.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# This file describes code used for generating hypernym data for CUB-birds
# dataset. The taxonomy knowledge is from "the eBird Taxonomy" checklist,
# maintained by the Cornell Lab of Ornithology. Please check out their website
# (https://ebird.org/home) for more information.

import os
import pandas
import sys
from IPython.core import ultratb
from IPython import embed

from utility.common import dump

sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Linux', call_pdb=1)

"""
Special data comments for outliers
"""

special_class_doc = {
    'Cardinal': {
        'other names': ('Cardinalidae',),
        'level': 'family',
    },
    'Frigatebird': {
        'other names': ('Fregata',),
        'level': 'genus',
    },
    'Green_Violetear': {
        'other names': ('Mexican Violetear', 'Lesser Violetear'),
        'level': 'spuh',
        'parent node': {
            'name': 'Colibri',
            'level': 'genus',
        }
    },
    'Florida_Jay': {
        'other names': ('Florida Scrub-Jay',),
        'level': 'species',
    },
    'White_breasted_Kingfisher': {
        'other names': ('White-throated Kingfisher',),
        'level': 'species',
    },
    'Mockingbird': {
        'level': 'spuh',
        'parent node': {
            'name': 'Mimidae',
            'level': 'family',
        }
    },
    'Nighthawk': {
        'other names': ('Chordeilinae'),
        'level': 'sub-family',
        'parent node': {
            'name': 'Caprimulgidae',
            'level': 'family'
        }
    },
    'White_Pelican': {
        'other names': ('American White Pelican', 'Great White Pelican'),
        'level': 'spuh',
        'parent node': {
            'name': 'Pelecanus',
            'level': 'genus',
        },
    },
    'Sayornis': {
        'level': 'genus',
    },
    'Whip_poor_Will': {
        'other names': ('Eastern Whip-poor-will', 'Mexican Whip-poor-will'),
        'level': 'spuh',
        'parent node': {
            'name': 'Antrostomus',
            'level': 'genus',
        }
    },
    'Geococcyx': {
        'level': 'genus',
    },
    'Great_Grey_Shrike': {
        'other names': ('Great Gray Shrike',),
        'level': 'species',
    },
    'Le_Conte_Sparrow': {
        'other names': ('LeConte\'s Sparrow',),
        'level': 'species',
    },
    'Nelson_Sharp_tailed_Sparrow': {
        'other names': ('Saltmarsh Sparrow',),
        'level': 'species',
    },
    'Tree_Sparrow': {
        'other names': ('American Tree Sparrow', 'Eurasian Tree Sparrow'),
        'level': 'spuh',
        'parent node': {
            'name': 'Passeridae',
            'level': 'family',
        }
    },
    'Cape_Glossy_Starling': {
        'other names': ('Cape Starling',),
        'level': 'species',
    },
    'Artic_Tern': {
        'other names': ('Arctic Tern',),
        'level': 'species',
    },
    'Forsters_Tern': {
        'other names': ('Forster\'s Tern',),
        'level': 'species',
    },
    'Myrtle_Warbler': {
        'other names': ('Yellow-rumped Warbler',),
        'level': 'species',
    }
}


"""
Component functions for Main
"""


def read_classes(class_txt):
    with open(class_txt, 'r') as f:
        read_lines = f.readlines()

    classes = [s.lstrip('0123456789. ').rstrip('\n')
               for s in read_lines]
    return classes


def read_taxonomy(taxonomy_xlsx):
    sheet = pandas.read_excel(taxonomy_xlsx)
    _Null, y_axis = sheet.axes
    y_axis = y_axis.to_list()
    matrix = sheet.fillna('').values
    return matrix, y_axis


def build_hierarchy(classes, taxonomy, y_axis):
    index_y = {
        'common name': y_axis.index('PRIMARY_COM_NAME'),
        'binomial name': y_axis.index('SCI_NAME'),
        'order': y_axis.index('ORDER1'),
        'family': y_axis.index('FAMILY'),
        'category': y_axis.index('CATEGORY'),
    }
    tree = TaxonomyTree()
    for name in classes:
        if name not in special_class_doc:
            node = Species(name)
            node.get_chain(taxonomy, index_y)
        else:
            node_info = special_class_doc[name]
            if 'parent node' in node_info:
                parent_info = node_info['parent node']

                node = TaxonomyBasicNode(name, level=node_info['level'])
                parent_node = get_node_by_level(
                    parent_info['name'], parent_info['level']
                )
                parent_node.get_chain(taxonomy, index_y)
                node.set_parent(parent_node)
            else:
                if 'other names' in node_info:
                    actual_name = node_info['other names'][0]
                else:
                    actual_name = None
                node = get_node_by_level(
                    name, node_info['level'], actual_name
                )
                node.get_chain(taxonomy, index_y)
        tree.merge(node)

    return tree


def get_hypernym_dict(hierarchy):
    hypernym = {}
    for new_sub, node in hierarchy.record.items():
        parent = node.parent

        while parent is not None:
            if parent.name not in hypernym:
                hypernym[parent.name] = set()
            hypernym[parent.name].add(new_sub)
            parent = parent.parent

    hypernym = {
        parent: list(nodes)
        for parent, nodes in hypernym.items()
    }
    return hypernym


def output_hierarchy_json(hierarchy):

    def get_tree(node):
        if len(node.children) == 0:
            sub_forest = {}
        else:
            sub_forest = {
                sub_root: sub_branches
                for child in node.children
                for sub_root, sub_branches in get_tree(child).items()
            }
        return {node.name: sub_forest}

    tree = get_tree(hierarchy.record['Aves'])
    return tree


def main(class_txt, taxonomy_xlsx, hypernym_json, hierarchy_file):
    assert class_txt.endswith('.txt')
    assert taxonomy_xlsx.endswith('.xlsx')

    classes = read_classes(class_txt)
    taxonomy_sheet, y_axis = read_taxonomy(taxonomy_xlsx)

    hierarchy = build_hierarchy(classes, taxonomy_sheet, y_axis)
    tree = output_hierarchy_json(hierarchy)

    cub_hypernym = get_hypernym_dict(hierarchy)
    print(f'Outputing hypernym stats in {hypernym_json}')
    dump(cub_hypernym, hypernym_json)
    print(f'Outputing hierarchy in {hierarchy_file}')
    dump(tree, hierarchy_file)

    # for possible debugging
    embed()
    del tree


"""
Utility functions and classes
"""


def plain_name(name):
    output = name.replace('\'s', '')
    output = output.replace(' ', '_')
    output = output.replace('-', '_')
    output = output.capitalize()
    return output


def is_species(item, index_y):
    return item[index_y['category']] == 'species'


class TaxonomyTree:
    """
    A tree structure class, representing a taxonomy tree
    """
    def __init__(self):
        self.root = Class('Aves')
        self.record = {}
        self.add_record(self.root)

    def add_record(self, node):
        assert node.name not in self.record, 'can not add a duplicate node'
        self.record[node.name] = node

    def in_record(self, node):
        return node.name in self.record

    def merge(self, node):
        if not self.in_record(node):
            self.add_record(node)
        if node.parent is None:
            assert node == Class('Aves'), 'root node must be the Aves class'
        else:
            if self.in_record(node.parent):
                node.set_parent(self.record[node.parent.name])
            else:
                self.merge(node.parent)


class TaxonomyNode:
    """
    Super of Taxonomy classes. This should not be used directly
    """
    def __init__(self, name, actual_name=None):
        self.name = name
        self.actual_name = actual_name
        self.parent = None
        self.children = set()
        self.level = None

    def set_parent(self, parent):
        self.parent = parent
        if parent is not None:
            self.parent.add_child(self)

    def add_child(self, child):
        self.children.add(child)

    def get_chain(self, taxonomy, index_y):
        raise NotImplementedError()

    def find_self(self, taxonomy, index_y):
        raise NotImplementedError()

    @classmethod
    def parse_item(cls, item, index_y):
        raise NotImplementedError()

    def __str__(self):
        return f'{self.level}:{self.name}'

    @property
    def valid_name(self):
        return self.actual_name or self.name


class TaxonomyBasicNode(TaxonomyNode):
    def __init__(self, name, level, actual_name=None):
        super().__init__(name, actual_name)
        self.level = level


class Species(TaxonomyNode):
    """
    Node representing a species
    """
    def __init__(self, name, actual_name=None):
        super().__init__(name, actual_name)
        self.level = 'species'

    def get_chain(self, taxonomy, index_y):
        index, item = self.find_self(taxonomy, index_y)
        parent = Genus.parse_item(item, index_y)
        self.set_parent(parent)

    def find_self(self, taxonomy, index_y):
        common_names = taxonomy[:, index_y['common name']]
        plain_common_names = [plain_name(name) for name in common_names]
        index = plain_common_names.index(plain_name(self.valid_name))
        self.actual_name = common_names[index]
        return index, taxonomy[index]

    @classmethod
    def parse_item(cls, item, index_y):
        level_index = index_y['common name']
        name = item[level_index]
        node = cls(name)
        parent = Genus.parse_item(item, index_y)
        node.set_parent(parent)
        return node


class Genus(TaxonomyNode):
    """
    Node representing a genus
    """
    def __init__(self, name, actual_name=None):
        super().__init__(name, actual_name)
        self.level = 'genus'

    def get_chain(self, taxonomy, index_y):
        index, item = self.find_self(taxonomy, index_y)
        parent = Family.parse_item(item, index_y)
        self.set_parent(parent)

    def find_self(self, taxonomy, index_y):
        for i, sci_name in enumerate(taxonomy[:, index_y['binomial name']]):
            item = taxonomy[i]
            if sci_name.startswith(self.valid_name) and \
                    is_species(item, index_y):
                self.actual_name = sci_name
                return i, item
        raise Exception(f'genus not found: {self.name}')

    @classmethod
    def parse_item(cls, item, index_y):
        level_index = index_y['binomial name']
        name = item[level_index].split(' ')[0]
        node = cls(name)
        parent = Family.parse_item(item, index_y)
        node.set_parent(parent)
        return node


class Family(TaxonomyNode):
    """
    Node representing a family
    """
    def __init__(self, name, actual_name=None):
        super().__init__(name, actual_name)
        self.level = 'family'

    def get_chain(self, taxonomy, index_y):
        index, item = self.find_self(taxonomy, index_y)
        parent = Order.parse_item(item, index_y)
        self.set_parent(parent)

    def find_self(self, taxonomy, index_y):
        for i, family_name in enumerate(taxonomy[:, index_y['family']]):
            item = taxonomy[i]
            if family_name.startswith(self.valid_name) and \
                    is_species(item, index_y):
                self.actual_name = family_name
                return i, item
        raise Exception(f'family not found: {self.name}')

    @classmethod
    def parse_item(cls, item, index_y):
        level_index = index_y['family']
        name = item[level_index].split(' ')[0]
        node = cls(name)
        parent = Order.parse_item(item, index_y)
        node.set_parent(parent)
        return node


class Order(TaxonomyNode):
    """
    Node representing an order (a level in biologcal taxonomy)
    """
    def __init__(self, name, actual_name=None):
        super().__init__(name, actual_name)
        self.level = 'order'

    def get_chain(self, taxonomy, index_y):
        index, item = self.find_self(taxonomy, index_y)
        parent = Class('Aves')
        self.set_parent(parent)

    def find_self(self, taxonomy, index_y):
        for i, order_name in enumerate(taxonomy[:, index_y['order']]):
            item = taxonomy[i]
            if order_name == self.valid_name and is_species(item, index_y):
                self.actual_name = order_name
                return i, item
        raise Exception(f'order not found: {self.name}')

    @classmethod
    def parse_item(cls, item, index_y):
        level_index = index_y['order']
        name = item[level_index]
        node = cls(name)
        parent = Class('Aves')
        node.set_parent(parent)
        return node


class Class(TaxonomyNode):
    """
    Node representing a class. In this case, there is only one class: Aves
    """
    def __init__(self, name, actual_name=None):
        super().__init__(name, actual_name)
        self.level = 'class'

    def get_chain(self, taxonomy, index_y):
        pass

    def find_self(self, taxonomy, index_y):
        raise NotImplementedError('Why would you ever call this for a Class?')

    @classmethod
    def parse_item(cls, item, index_y):
        raise NotImplementedError('Why would you ever call this for a Class?')


def get_node_by_level(name, level, actual_name=None):
    if level == 'species':
        cls = Species
    elif level == 'genus':
        cls = Genus
    elif level == 'family':
        cls = Family
    elif level == 'order':
        cls = Order
    elif level == 'class':
        cls = Class
    return cls(name, actual_name)


"""
Running main program
"""
if __name__ == '__main__':
    print('Program Starts')

    root_dir = '.'
    # relative path
    hypernym_json = 'knowledge/cub_hypernym.json'
    taxonomy_graph_file = 'knowledge/source/cub_hierarchy.json'
    data_dir = '../data/cub'
    bird_class_txt = 'raw/classes.txt'
    bird_taxonomy_xlsx = 'processed/taxonomy.xlsx'
    # get absolute path
    data_dir = os.path.join(root_dir, data_dir)
    hypernym_json = os.path.join(root_dir, hypernym_json)
    taxonomy_graph_file = os.path.join(root_dir, taxonomy_graph_file)
    bird_class_txt = os.path.join(data_dir, bird_class_txt)
    bird_taxonomy_xlsx = os.path.join(data_dir, bird_taxonomy_xlsx)

    # run main function
    main(bird_class_txt, bird_taxonomy_xlsx,
         hypernym_json, taxonomy_graph_file)
