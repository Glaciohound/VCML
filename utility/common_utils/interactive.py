#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : interactive.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 09.08.2019
# Last Modified Date: 09.08.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license


def yes_or_no(question):
    answer = None
    while answer is None:
        message = f'{question} [yes/no]: '
        feedback = input(message)
        if feedback == 'yes':
            answer = True
        elif feedback == 'no':
            answer = False
        else:
            print('Invalid feedback, please answer again.')

    return answer
