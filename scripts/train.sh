#!/bin/bash

python train.py --batch_size 10 --toy --allow_output_protocol --protocol_file processed/teddy_protocol.json --max_concepts 40 --batch_size 50 --toy_names 4 --toy_objects 2 --toy_attributesPobject 2 --toy_attributes 6 --toy_mode query
