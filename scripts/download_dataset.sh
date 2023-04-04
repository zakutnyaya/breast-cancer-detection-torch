#!/usr/bin/env bash

pip install --upgrade pip
CUR_DIR=$pwd
DATA_DIR_LOC=data/raw

mkdir /root/.kaggle/
cp kaggle.json /root/.kaggle/

mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

python3 df.py

pip install kaggle --upgrade
kaggle datasets download -d annazakutniaia/breast-cancer-truncated
unzip breast-cancer-truncated.zip
cp -a rsna-breast-cancer-detection/. .
rm -rf breast-cancer-truncated.zip rsna-breast-cancer-detection/
