#!/bin/bash
# File              : prepare.sh
# Author            : Chi Han
# Email             : haanchi@gmail.com
# Date              : 18.11.2019
# Last Modified Date: 19.11.2019
# Last Modified By  : Chi Han
#
# Welcome to this little kennel of Glaciohound!


cwd=$(pwd)
mkdir -p $1
cd $1

wget http://vcml.csail.mit.edu/data/dataset_augmentation.tgz
tar -xvf dataset_augmentation.tgz

mv dataset_augmentation/CLEVR CLEVR/augmentation
mv dataset_augmentation/GQA GQA/augmentation
mv dataset_augmentation/CUB CUB/augmentation

rm -r dataset_augmentation.tgz dataset_augmentation

cd $(pwd)
