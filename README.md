# Breast Cancer Detection using PyTorch
This repository contains code for training and evaluating a deep learning model for the detection of breast cancer in mammography images using PyTorch. This code is related to [competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview) held by Kaggle. It is crucial to develop a system that detects cancer as early as possible:
> According to the WHO, breast cancer is the most commonly occurring cancer worldwide. In 2020 alone, there were 2.3 million new breast cancer diagnoses and 685,
> 000 deaths. Early detection and treatment are critical to reducing cancer fatalities, and your machine learning skills could help streamline the process 
> radiologists use to evaluate screening mammograms.  
> 
> Your work improving the automation of detection in screening mammography may enable radiologists to be more accurate and efficient, improving the quality and
> safety of patient care. It could also help reduce costs and unnecessary medical procedures. - Competition overview

## Dataset
The dataset is provided by the Radiological Society of North America (RSNA) and available for download on the Kaggla [website](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview). The dataset consist of DICOM files with mammograms and csv files with labels for each image (cancer or no cancer) and some additional info. The dataset contains 54.7k mammograms from around 65.5k patients. On average there are 4 images per patient. The dataset is highly unbalanced with 97.89% of images being negative (no cancer) and 2.11% being positive (cancer).

## Metrics
Model evaluation is performed using probabilistic F1-score introduced in the [paper](https://aclanthology.org/2020.eval4nlp-1.9.pdf). This is the official metric from the competition. Python metric implementation can be found [here](https://www.kaggle.com/code/sohier/probabilistic-f-score).

## Models and training
In order to identify breast cancer on images, images classification problem is solved using transfer learning technique. Convolutional neural network, pretrained on ImageNet dataset, is finetuned on the Kaggle breast cancer dataset and used for binary classification. The [SEResNeXt](https://paperswithcode.com/model/seresnext?variant=seresnext50-32x4d) architecture is used for classification.

Model is finetuned using Focal Loss with Adam optimizer and 1cycle learning rate sheduler.

## Images preprocessing
DICOM files with mammograms from the official dataset are converted to images using the following pipeline:
1. A pretrained YOLOv5 detector extracts breast ROIs.
2. The ROI background is colored black to remove mammography artifacts.
3. ROI images are resized to 512x1024 size (width x height).

## Installation and Usage
1. Clone this repository
2. In order to use Kaggle datatset set up a Kaggle account, receive a Kaggle API key, and place it in the .kaggle directory. Detailed instructions can be found [here](https://www.kaggle.com/docs/api).
3. Install Docker.
4. Download the dataset from the Kaggle platform using the script ```bash scripts/dataset_download.sh```.
5. Perform image preprocessing by running ```python src/roi_extraction.py```.
6. Split data on train, test and validation by ```python src/split_data.py```.
7. Train the model by running ```python src/train_runner.py --action train```.
8. Generate predictions on the validation set for all checkpoints from some experiment by running ```python src/train_runner.py --action generate_predictions```.
9. Select the best checkpoint and the best threshold for classification score by running ```python src/train_runner.py --action check_metric```.
10. Test the best checkpoint and threshold on the test set by running python ```src/train_runner.py --action test_model``` and specifying the best checkpoint and threshold in the arguments ```--best_epoch``` and ```--best_threshold```.

## Acknowledgments

This project was inspired by the following paper:

```
@InProceedings{Gabruseva_2020_CVPR_Workshops,
  author = {Gabruseva, Tatiana and Poplavskiy, Dmytro and Kalinin, Alexandr A.},
  title = {Deep Learning for Automatic Pneumonia Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2020}
}
```
