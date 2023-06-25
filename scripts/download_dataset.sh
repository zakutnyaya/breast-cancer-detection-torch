#!/usr/bin/env bash

pip install --upgrade pip
CUR_DIR=$pwd
DATA_DIR_LOC=data/raw

mkdir /root/.kaggle/
cp kaggle.json /root/.kaggle/

mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

pip install kaggle --upgrade
kaggle competitions download -c rsna-breast-cancer-detection
unzip rsna-breast-cancer-detection.zip
cp -a rsna-breast-cancer-detection/. .
rm -rf rsna-breast-cancer-detection.zip rsna-breast-cancer-detection/
