import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from config import PROCESSED_DATA_DIR, PROCESSED_IMAGES_DIR
from transforms import TransformCfg


class BreastROIDataset(Dataset):
    """
    Breast ROIs Dataset
    """

    def __init__(
        self,
        subset: bool,
        debug: bool,
        augmentation_level: int = 1,

    ):
        """
        Args:
            subset            : type of subset (train, val or test)
            debug             : if True, run in debugging mode
            augmentation_level: augmentations set
        """
        super(BreastROIDataset, self).__init__()
        self.debug = debug
        self.subset = subset
        self.augmentation_level = augmentation_level
        samples = pd.read_csv(PROCESSED_DATA_DIR / "data_subsets.csv")

        # samples = samples[samples.subset == self.subset]
        if self.debug:
            samples = samples.head(8)

        self.image_ids = list(samples.image_id.tolist())
        self.patients = list(samples.patient_id.tolist())
        self.targets = list(samples.cancer.tolist())
        self.laterality = list(samples.laterality.tolist())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        patient_id = self.patients[idx]
        label = self.targets[idx]

        image = cv2.imread(
            str(PROCESSED_IMAGES_DIR / str(patient_id) / f"{image_id}.png"),
            cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmentation_sigma = {
            1: dict(
                rotate_limit=3,
                scale_limit=[-0.05, 0.1],
                shift_limit=0.03,
                shift_limit_x=[0, 0],
                hflip=True
            ),
            2: dict(
                rotate_limit=[-6, 6],
                scale_limit=[-0.1, 0.1],
                shift_limit=[-0.1, 0.1],
                shift_limit_x=[0, 0],
                hflip=True,
                coerse_max_holes=32,
                coerse_max_height=8,
                coerse_max_width=8,
                noise_var_limit=(10, 20)
            )
        }[self.augmentation_level]

        if self.subset == "train":
            cfg = TransformCfg(
                is_train=True,
                hflip=augmentation_sigma["hflip"],
                rotate_limit=augmentation_sigma["rotate_limit"],
                shift_limit_x=augmentation_sigma["shift_limit_x"],
                shift_limit=augmentation_sigma["shift_limit"],
                scale_limit=augmentation_sigma["scale_limit"],
                coerse_max_holes=augmentation_sigma["coerse_max_holes"],
                coerse_max_height=augmentation_sigma["coerse_max_height"],
                coerse_max_width=augmentation_sigma["coerse_max_width"],
                noise_var_limit=augmentation_sigma["noise_var_limit"]

            )
        else:
            cfg = TransformCfg(is_train=False)

        image = cfg.transform_image(image)
        target = torch.as_tensor(label, dtype=torch.uint8)

        return image, target
