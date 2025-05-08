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
    
    RGB_TO_CLASS = {
        (128, 64, 128): 0,    # road
        (244, 35, 232): 1,    # sidewalk
        (70, 70, 70): 2,      # building
        (102, 102, 156): 3,   # wall
        (190, 153, 153): 4,   # fence
        (153, 153, 153): 5,   # pole
        (250, 170, 30): 6,    # traffic light
        (220, 220, 0): 7,     # traffic sign
        (107, 142, 35): 8,    # vegetation
        (152, 251, 152): 9,   # terrain
        (70, 130, 180): 10,   # sky
        (220, 20, 60): 11,    # person
        (255, 0, 0): 12,      # rider
        (0, 0, 142): 13,      # car
        (0, 0, 70): 14,       # truck
        (0, 60, 100): 15,     # bus
        (0, 80, 100): 16,     # train
        (0, 0, 230): 17,      # motorcycle
        (119, 11, 32): 18     # bicycle
    }

    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[Compose] = None):
        super().__init__()
        self.transform = transform
        self.samples = []

        img_dir = os.path.join(root_dir, "images")
        lbl_dir = os.path.join(root_dir, "labels")

        for fname in os.listdir(img_dir):
            if fname.endswith(".png"):
                img_path = os.path.join(img_dir, fname)
                lbl_path = os.path.join(lbl_dir, fname)
                self.samples.append((img_path, lbl_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        # label_rgb = np.array(Image.open(lbl_path).convert("RGB"))
        # label = self.convert_rgb_to_grey_scale(label_rgb)
        label = np.array(Image.open(lbl_path).convert("L"))

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        return image, label

    def convert_rgb_to_grey_scale(self, mask: np.ndarray) -> np.ndarray:
        h, w, _ = mask.shape
        label = np.zeros((h, w), dtype=np.uint8)

        for rgb, class_id in self.RGB_TO_CLASS.items():
            matches = np.all(mask == rgb, axis=-1)
            label[matches] = class_id

        return label
