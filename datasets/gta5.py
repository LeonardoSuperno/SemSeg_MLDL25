from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np
from typing import Optional, Tuple
from albumentations import Compose


class GTA5(Dataset):
    """
    Custom Dataset for the GTA5 dataset.
    Loads synthetic images and their corresponding labelTrainIds masks.
    """
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[Compose] = None):
        """
        Args:
            root_dir (str): Root directory of GTA5 dataset.
            transform (Optional[Compose]): Albumentations transform to apply.
        """
        super().__init__()
        self.transform = transform
        self.samples = []

        img_dir = os.path.join(root_dir, "images")
        lbl_dir = os.path.join(root_dir, "labels")

        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith(".png"):
                img_path = os.path.join(img_dir, fname)
                lbl_path = os.path.join(lbl_dir, fname)  # stesso nome, due cartelle diverse
                self.samples.append((img_path, lbl_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(lbl_path).convert("L"))  # label in grayscale (TrainIDs)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        return image, label
