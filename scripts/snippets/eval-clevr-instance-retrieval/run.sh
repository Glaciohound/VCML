#! /bin/bash
# run.sh
# Copyright (C) 2019 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

echo Evaluating $1
jac-run eval-classification.py --scene-json ./test-data/scenes.json --preds-json $1
jac-run eval.py --scene-json ./test-data/scenes.json --preds-json $1
jac-run eval-referential.py --scene-json ./test-data/scenes.json --preds-json $1

