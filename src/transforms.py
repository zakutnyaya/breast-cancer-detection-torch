import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import List, Tuple


class TransformCfg:
    """
    Configuration structure for augmentations
    """

    def __init__(
        self,
        is_train: bool,
        rotate_limit: float = 3,
        shift_limit_x: List[float] = [0, 0],
        shift_limit: float = 0.03,
        scale_limit: List[float] = [-0.05, 0.1],
        hflip: bool = False,
        coerse_max_holes: int = 16,
        coerse_max_height: int = 4,
        coerse_max_width: int = 4,
        blur_sigma_limit: Tuple[int] = (0, 2),
        blur_limit: Tuple[int] = (3, 7),
        noise_var_limit: Tuple[int] = (0, 20)
    ):
        self.is_train = is_train
        self.hflip = hflip,
        self.rotate_limit = rotate_limit
        self.shift_limit_x = shift_limit_x
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.coerse_max_holes = coerse_max_holes
        self.coerse_max_height = coerse_max_height
        self.coerse_max_width = coerse_max_width
        self.blur_sigma_limit = blur_sigma_limit
        self.blur_limit = blur_limit
        self.noise_var_limit = noise_var_limit

    def set_train_transforms(self) -> A.Compose:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                rotate_limit=self.rotate_limit,
                shift_limit=self.shift_limit,
                shift_limit_x=self.shift_limit_x,
                scale_limit=self.scale_limit,
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                p=0.65,
            ),
            A.CoarseDropout(
                max_holes=self.coerse_max_holes,
                max_height=self.coerse_max_height,
                max_width=self.coerse_max_width,
                p=0.6
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=self.blur_limit, sigma_limit=self.blur_sigma_limit, p=0.5),
                    A.GaussNoise(var_limit=self.noise_var_limit, p=0.5)
                ],
                p=0.5
            ),
            A.RandomGamma(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=0, std=1),
            ToTensorV2()
        ])
        return transform

    def set_test_transforms(self) -> A.Compose:
        transform = A.Compose([
            A.Normalize(mean=0, std=1),
            ToTensorV2()
        ])
        return transform

    def transform_image(self, image: np.array) -> np.array:
        if self.is_train:
            transforms = self.set_train_transforms()
        else:
            transforms = self.set_test_transforms()

        image = transforms(image=image)['image']
        return image
