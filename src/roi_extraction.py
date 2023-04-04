"""

Extracting breast ROIs from DICOM files

"""
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import dicomsdl
import pylibjpeg
from typing import List, Tuple

from config import (
    SOURCE_IMAGES_DIR,
    DETECTION_IMG_SIZE,
    CROP_SIZE_HEIGHT,
    CROP_SIZE_WIDTH,
    TRAIN_DATA,
    PROCESSED_IMAGES_DIR,
    YOLOV5_PATH,
    DETECTION_WEIGHTS_DIR
)


def split_given_size(a: np.ndarray, size: int) -> List[np.ndarray]:
    """
    Split array into chunks

    Args:
        a: array
        size: chunk size
    """
    return np.split(a, np.arange(size, len(a), size))


def read_dicom(images_dir: Path, patient_id: int, image_id: int) -> np.ndarray:
    """
    Read dicom file, return image from the dicom

    Args:
        images_dir: directory where dicoms are stored
        patient_id: id of the patient
        image_id: id of the sample

    Outputs:
        image: image stored in the dicom
    """
    dicom = dicomsdl.open(
        str(images_dir / str(patient_id) / f"{image_id}.dcm")
    )
    image = dicom.pixelData()
    image = (image - image.min()) / (image.max() - image.min())

    if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        image = 1 - image

    image = (image * 255).astype(np.uint8)
    return image


def preprocess_image(image_idx: List[int]) -> Tuple[np.ndarray]:
    """
    Read dicom, resize image from dicom

    Args:
        image_idx: dicom id (patient_id + image_id)

    Outputs:
        image: image from dicom
        image_res: resized image from dicom
    """
    patient_id, image_id = image_idx
    image = read_dicom(SOURCE_IMAGES_DIR, patient_id, image_id)
    image_res = cv2.resize(
        image,
        (DETECTION_IMG_SIZE, DETECTION_IMG_SIZE)
    )
    return (image_res, image)


def rmv_background(image: np.ndarray) -> np.ndarray:
    """
    Remove artifacts from images, create black background

    Args:
        image: image to handle

    Outputs:
        contour_image: image with breast ROI and black background
    """
    height, width = image.shape
    _, image_th = cv2.threshold(
        image, 1, 255, cv2.THRESH_BINARY
    )
    if np.sum(image_th) / (255 * height * width) > 0.95:
        _, image_th = cv2.threshold(
            image, 1, 255, cv2.THRESH_OTSU
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    image_th = cv2.morphologyEx(image_th, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        image_th,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    squares = []
    for cont in contours:
        _, _, w, h = cv2.boundingRect(cont)
        squares.append(w * h)   
    max_idx = np.argmax(squares)

    empty_image = np.zeros(image.shape).astype(image.dtype)
    contour_image = cv2.fillPoly(
        empty_image, pts=[contours[max_idx]], color=1
    ) * image
    return contour_image


def get_roi_orientation(image: np.ndarray, image_border_buffer: float = 0.1) -> str:
    """
    Determine image side (left or right) where breast ROI is located

    Args:
        image: image to handle
        image_border_buffer: pixel buffer

    Outputs:
        string that define side
    """
    _, im_width = image.shape
    x_buffer = int(im_width * image_border_buffer)
    left = np.sum(image[:, :x_buffer])
    right = np.sum(image[:, -x_buffer:])
    if left > right:
        return "L"
    return "R"


def pad_roi_to_size(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Pad image with breast ROI to predefined target size

    Args:
        image: image to handle
        target_width: target width
        target_heght: target height

    Outputs:
        image padded to target size
    """
    height, width = image.shape
    pad_height = target_height - height
    pad_width = target_width - width

    pad_w = (0, 0)
    if pad_width > 0:
        side = get_roi_orientation(image)
        if side == "L":
            pad_w = (0, pad_width)
        else:
            pad_w = (pad_width, 0)

    add = target_height - height - 2 * (pad_height // 2)
    pad = (
        (add + pad_height // 2, pad_height // 2),
        pad_w
    )
    return np.pad(image, pad, constant_values=0)


def resize_pad(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize image saving aspect ratio, pad resized image to the target size

    Args:
        image: image to handle
        target_width: target width
        target_heght: target height

    Outputs:
        resized image
    """
    w, h = image.size
    h_ratio = h / target_height
    w_ratio = w / target_width
    ratio = max(h_ratio, w_ratio)

    image = cv2.resize(
        np.array(image), (int(w / ratio), int(h / ratio))
    )
    return pad_roi_to_size(image, target_width, target_height)


def crop_roi(image: np.ndarray, crop_box: List[float]) -> np.ndarray:
    """
    Crop ROI out of image, resize it to target width and height

    Args:
        image: image to handle
        crop_box: coordinates of the ROI box

    Outputs:
        resized image
    """
    crop = Image.fromarray(image).crop(crop_box)
    return resize_pad(crop, CROP_SIZE_WIDTH, CROP_SIZE_HEIGHT)


def crop_save_roi(pred: torch.Tensor, image_idx: List[int], image: np.ndarray) -> None:
    """
    Receive detection predictions (breast ROI box),
    crop ROI out of image, remove artifacts from ROI,
    save ROI.

    Args:
        pred: detection predictions (breast ROI boxes)
        image_idx: idx of the image used for detection
        image: image used for detection
    """
    patient_id, image_id = image_idx
    pred = pred.cpu().numpy()

    if len(pred) == 0:
        box = [0, 0, DETECTION_IMG_SIZE, DETECTION_IMG_SIZE, 0, 0]
    else:
        box_idx = np.argmax(pred, axis=0)[4]
        box = pred[box_idx]

    xmin, ymin, xmax, ymax, score, _ = box
    xmin /= DETECTION_IMG_SIZE
    ymin /= DETECTION_IMG_SIZE
    xmax /= DETECTION_IMG_SIZE
    ymax /= DETECTION_IMG_SIZE

    height, width = image.shape
    crop_box = [
        int(xmin * width),
        int(ymin * height),
        int(xmax * width),
        int(ymax * height)
    ]

    if not (PROCESSED_IMAGES_DIR / str(patient_id)).exists():
        (PROCESSED_IMAGES_DIR / str(patient_id)).mkdir(parents=True, exist_ok=True)

    crop = crop_roi(image, crop_box)
    crop = rmv_background(crop)
    Image.fromarray(crop).save(
        PROCESSED_IMAGES_DIR / str(patient_id) / f"{image_id}.png"
    )


def get_images(images_idx: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Read and preprocess images from dicoms in parallel

    Args:
        images_idx: indices of images to read

    Outputs:
        processed_images: resized images from dicom
        src_images: images from dicom
    """
    images = Parallel(n_jobs=16, prefer="threads")(
        delayed(preprocess_image)(image_idx)
        for image_idx in images_idx
    )
    images = np.array(images)
    processed_images = images[:, 0]
    src_images = images[:, 1]
    return (processed_images, src_images)


def load_yolov5_model():
    """
    Helper to load yolov5 the best model weihts
    """
    print("Loading yolov5 model")
    model = torch.hub.load(
        YOLOV5_PATH,
        'custom',
        path=DETECTION_WEIGHTS_DIR,
        source='local',
        force_reload=True
    )
    return model


def main():
    train_df = pd.read_csv(TRAIN_DATA)

    images_idxs = train_df[["patient_id", "image_id"]].values
    splits = split_given_size(images_idxs, size=32)

    yolov5 = load_yolov5_model()

    PROCESSED_IMAGES_DIR.mkdir(exist_ok=True, parents=True)

    for split in tqdm(splits, total=len(splits)):
        processed_images, src_images = get_images(split)
        output = yolov5(
            processed_images.tolist(),
            size=DETECTION_IMG_SIZE
        ).xyxy
        print("done")
        Parallel(n_jobs=16, prefer='threads')(
            delayed(crop_save_roi)(pred, image_idx, image)
            for pred, image_idx, image in tqdm(zip(output, split, src_images))
        ) 


if __name__ == "__main__":
    main()
