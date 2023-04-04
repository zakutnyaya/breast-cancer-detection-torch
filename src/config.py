from pathlib import Path

# main data paths
SOURCE_DATA = Path("data/raw")
TRAIN_DATA = SOURCE_DATA / "train.csv"
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"

# detection
SOURCE_IMAGES_DIR = SOURCE_DATA / "train_images"
DETECTION_WEIGHTS_DIR = Path("models/pretrained/detection/yolov5s-with-bckgr-best.pt")
YOLOV5_PATH = "yolov5"
DETECTION_IMG_SIZE = 640
CROP_SIZE_HEIGHT = 1024
CROP_SIZE_WIDTH = 512
detection_batch_size = 32

# classification
CLASSIFICATION_WEIGHTS_DIR = Path("models/pretrained/classification/")
