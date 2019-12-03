#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : logging.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 25.07.2019
# Last Modified Date: 01.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
# logging and displaying information when running


import os
import datetime
import pprint
from tqdm import tqdm

from utility.common \
    import sum_list, make_parent_dir
from utility import colortext as color


def get_levelup(logger):

    class levelup:
        def __init__(self):
            pass

        def __enter__(self):
            logger._levelup()

        def __exit__(self, _type, _value, _traceback):
            logger._leveldown()

    return levelup


class Logger:
    def __init__(self, log_dir, is_main=True, silent=False):
        self.logbook = []
        self.starttime = self.time
        self.level = 0
        self.levelup = get_levelup(self)
        self.log_dir = log_dir
        self.is_main = is_main
        self.silent = silent

    @property
    def time(self):
        return datetime.datetime.now()

    def time_str(self):
        return str(self.time)

    @property
    def time_message(self):
        return f'Current Time is: {self.time_str()}'

    def showtime(self):
        self.log(self.time_message)

    def _levelup(self):
        self.level += 1

    def _leveldown(self):
        self.level -= 1

    @property
    def arrow(self):
        if self.level == 0:
            return '=> '
        elif self.level == 1:
            return '-----> '
        elif self.level == 2:
            return '---------> '
        elif self.level == 3:
            return '- - - - - - -> '
        elif self.level == 4:
            return '- - - - - - - - -> '
        elif self.level == 5:
            return '-   -   -   -   -   -> '
        elif self.level == 6:
            return '-   -   -   -   -   -   -> '
        else:
            return ' ' * (self.level * 4) + '-> '

    @property
    def time_head(self):
        delta = self.time - self.starttime
        cropped = str(delta)[:10]
        return cropped

    @property
    def colored_head(self):
        time_head = self.time_head
        arrow = self.arrow
        return color.OKGREEN(time_head) + ' | ' + color.OKBLUE(arrow)

    @property
    def uncolored_head(self):
        time_head = self.time_head
        arrow = self.arrow
        return time_head + ' | ' + arrow

    @property
    def empty_time_head(self):
        time_head = self.time_head
        return ' ' * len(time_head) + ' | '

    @property
    def empty_head(self):
        arrow = self.arrow
        return self.empty_time_head + ' ' * len(arrow)

    @property
    def columns_left(self):
        width = self.get_windowsize()[1]
        return width - len(self.empty_head)

    def partition_string(self, string):
        room = self.columns_left
        ls = len(string)
        if ls < room:
            return [string]
        else:
            partitioned = [
                string[i * room: (i+1) * room]
                for i in range(ls // room + 1)
            ]
            return partitioned

    def print(self, content, pretty, resume):
        if pretty:
            string = pprint.pformat(content)
        else:
            string = str(content)

        partitioned = sum_list(
            *[self.partition_string(line)
              for line in string.split('\n')]
        )

        for i, line in enumerate(partitioned):
            self.delete_line()
            if i == 0 and not resume:
                partitioned[i] = self.uncolored_head + line
                if self.is_main:
                    print(self.colored_head + line)
            else:
                partitioned[i] = self.empty_head + line
                if self.is_main:
                    print(self.empty_head + line)
        return partitioned

    def log(self, item, pretty=False, resume=False):
        contents = self.print(item, pretty, resume)
        self.logbook.extend(contents)
        self.dump()

    def __call__(self, *arg, **kwarg):
        return self.log(*arg, **kwarg)

    def split_line(self):
        if self.is_main:
            line = '-' * self.columns_left
            print(self.empty_time_head + color.OKBLUE(line))

    def line_break(self):
        if self.is_main:
            self.delete_line()
            print(self.empty_head)

    def delete_line(self):
        if self.is_main:
            print('\r', end='')

    def state_dict(self):
        ckpt = {
            'logbook': self.logbook
        }
        return ckpt

    def load_state_dict(self, ckpt):
        self.logbook = ckpt['logbook'] + self.logbook

    def set_display(self):
        self.is_main = True

    def off_display(self):
        self.is_main = False

    @property
    def log_filename(self):
        filename = os.path.join(
            self.log_dir,
            'logging.txt'
        )
        return filename

    def dump(self):
        if not self.silent:
            make_parent_dir(self.log_filename)
            with open(self.log_filename, 'w') as f:
                for line in self.logbook:
                    f.write(color.remove_color(line) + '\n')

    def tqdm(self, iterable, *arg, **kwarg):
        output = tqdm(iterable, *arg, disable=not self.is_main,
                      leave=False, dynamic_ncols=True, **kwarg)
        return output

    @staticmethod
    def get_windowsize():
        rows, columns = os.popen('stty size', 'r').read().split()
        rows = int(rows)
        columns = int(columns)
        return rows, columns

    def get_ncols(self):
        return self.get_windowsize()[1]
