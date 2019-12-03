#!/bin/bash
# File              : download.sh
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

wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
wget https://nlp.stanford.edu/data/gqa/images.zip
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

unzip CLEVR_v1.0.zip
unzip images.zip
tar -zxvf CUB_200_2011.tgz

mkdir -p CLEVR CUB GQA
mv CLEVR_v1.0/images CLEVR/raw
mv images/allImages GQA/raw/images
mv CUB_200_2011/images CUB/raw

rm -r attributes.txt CLEVR_v1.0 CUB_200_2011

rm CLEVR_v1.0.zip
rm images.zip
rm CUB_200_2011.tgz

cd $(pwd)
