#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : colortext.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 07.08.2019
# Last Modified Date: 07.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# providing color configurations for texting


import copy


configs = {
    'HEADER': '\033[95m',
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
}


def colored(text, name):
    head = configs[name]
    tail = configs['ENDC']
    return head + text + tail


def OKGREEN(text):
    return colored(text, 'OKGREEN')


def FAIL(text):
    return colored(text, 'FAIL')


def OKBLUE(text):
    return colored(text, 'OKBLUE')


def remove_color(text):
    text = copy.deepcopy(text)
    for token in configs.values():
        text.replace(token, '')
    return text
