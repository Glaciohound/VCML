#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 30.07.2019
# Last Modified Date: 02.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
#
# main function


def precaution():
    import matplotlib
    matplotlib.use('Agg')


def main():
    precaution()
    from config.config import Args
    args = Args()

    if args.mode == 'run-experiment':
        from scripts import run_experiment
        run_experiment.run(args)
    elif args.mode == 'build-dataset':
        from scripts import build_dataset
        build_dataset.build_dataset(args)
    elif args.mode == 'output-dataset':
        from scripts import output_dataset
        output_dataset.output_dataset(args)


if __name__ == '__main__':
    main()
