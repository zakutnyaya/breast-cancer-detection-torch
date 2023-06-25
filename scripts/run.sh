#!/usr/bin/env bash
CUR_DIR=$pwd

# Download dataset
bash scripts/download_dataset.sh

python3 src/roi_extraction.py

python3 src/split_data.py

# Run full model training
python3 src/train_runner.py --action train --model seresnext50_32x4d --debug True --epochs 5 --class_batch_size 4

# python3 src/train_runner.py --action generate_predictions --model seresnext50_32x4d --debug True --class_batch_size 4

# python3 src/train_runner.py --action check_metric

# python3 src/train_runner.py --action test_model --best_epoch Epoch5_ROC0.000_PF10.000 --best_threshold 0.999 --debug True --class_batch_size 4
