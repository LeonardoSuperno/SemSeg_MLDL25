from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np
from typing import Optional, Tuple
from albumentations import Compose


class CityScapes(Dataset):
    """
    Custom Dataset for the Cityscapes dataset.
    Loads images and their corresponding labelTrainIds masks.
    """
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train', 
                 transform: Optional[Compose] = None):
        """
        Args:
            root_dir (str): Root directory of Cityscapes dataset.
            split (str): Dataset split, one of ['train', 'val', 'test'].
            transform (Optional[Compose]): Albumentations transform to apply.
        """
        super().__init__()
        self.transform = transform
        self.samples = []

        img_dir = os.path.join(root_dir, "leftImg8bit", split)
        lbl_dir = os.path.join(root_dir, "gtFine", split)

        for city in os.listdir(img_dir):
            img_city_path = os.path.join(img_dir, city)
            lbl_city_path = os.path.join(lbl_dir, city)
            for fname in os.listdir(img_city_path):
                if fname.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_city_path, fname)
                    lbl_name = fname.replace("_leftImg8bit", "_gtFine_labelTrainIds")
                    lbl_path = os.path.join(lbl_city_path, lbl_name)
                    self.samples.append((img_path, lbl_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(lbl_path).convert("L"))  # grayscale for labels

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        return image, label
