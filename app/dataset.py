"""Loads Severstal's train.csv into per-image, multi-channel (4, H, W) targets,
one channel per defect class. The raw CSV has one row per ImageId+ClassId, so
we group by image first.

Images are resized down from the native 256x1600 to speed up training. This
machine has no GPU (checked with nvidia-smi before starting), so a smaller
resolution and a lightweight encoder keep CPU training time reasonable. The
actual numbers from a real run are in README.md's training section.
"""
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from app.rle import rle_decode, IMG_HEIGHT, IMG_WIDTH

NUM_CLASSES=4
TRAIN_RESIZE=(128, 800) # (H, W), half native resolution, same aspect ratio


def load_annotations(csv_path: str) -> pd.DataFrame:
    """Group train.csv into one row per ImageId with a list of (class_id, rle) pairs."""
    df=pd.read_csv(csv_path)
    grouped=df.groupby("ImageId").apply(
        lambda g: list(zip(g["ClassId"], g["EncodedPixels"])), include_groups=False
    ).reset_index(name="annotations")
    return grouped


class SteelDefectDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, images_dir: str, resize: tuple[int,int]=TRAIN_RESIZE):
        self.annotations=annotations.reset_index(drop=True)
        self.images_dir=images_dir
        self.resize=resize

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row=self.annotations.iloc[idx]
        image_path=os.path.join(self.images_dir, row["ImageId"])

        image=cv2.imread(image_path, cv2.IMREAD_COLOR)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, (self.resize[1], self.resize[0])) # cv2.resize takes (W,H)

        mask=np.zeros((NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        for class_id,rle in row["annotations"]:
            mask[class_id-1]=rle_decode(rle, shape=(IMG_HEIGHT, IMG_WIDTH))
        mask_resized=np.zeros((NUM_CLASSES, self.resize[0], self.resize[1]), dtype=np.float32)
        for c in range(NUM_CLASSES):
            mask_resized[c]=cv2.resize(mask[c], (self.resize[1], self.resize[0]), interpolation=cv2.INTER_NEAREST)

        image=image.astype(np.float32)/255.0
        image=np.transpose(image, (2,0,1)) # HWC -> CHW

        return image, mask_resized
