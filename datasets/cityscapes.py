from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.labels_dir = os.path.join(root_dir, "gtFine", split)

        self.image_paths = []
        self.label_paths = []

        for city in os.listdir(self.images_dir):
            img_folder = os.path.join(self.images_dir, city)
            lbl_folder = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(img_folder):
                if file_name.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_folder, file_name)
                    label_name = file_name.replace("_leftImg8bit", "_gtFine_labelIds")
                    lbl_path = os.path.join(lbl_folder, label_name)

                    self.image_paths.append(img_path)
                    self.label_paths.append(lbl_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        image = np.array(image)
        label = np.array(label)

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        return image, label

    def __len__(self):
        return len(self.image_paths)
