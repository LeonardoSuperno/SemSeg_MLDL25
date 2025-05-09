import numpy as np
from PIL import Image
import PIL
from config import GTA, CITYSCAPES
import albumentations as A
import cv2

def get_color_to_id() -> dict:
    """
    Creates a dictionary mapping color representations to their corresponding IDs.

    Returns:
        dict: A dictionary where keys are color representations (RGB tuples) and values are IDs.
    """
    
    id_to_color = get_id_to_color()
    color_to_id = {color: id for id, color in id_to_color.items()}
    return color_to_id

def get_id_to_color() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding RGB color representations.

    Returns:
        dict: A dictionary where keys are class IDs (integers) and values are RGB tuples.
    """

    return {
        0: (128, 64, 128),    # road
        1: (244, 35, 232),    # sidewalk
        2: (70, 70, 70),      # building
        3: (102, 102, 156),   # wall
        4: (190, 153, 153),   # fence
        5: (153, 153, 153),   # pole
        6: (250, 170, 30),    # light
        7: (220, 220, 0),     # sign
        8: (107, 142, 35),    # vegetation
        9: (152, 251, 152),   # terrain
        10: (70, 130, 180),   # sky
        11: (220, 20, 60),    # person
        12: (255, 0, 0),      # rider
        13: (0, 0, 142),      # car
        14: (0, 0, 70),       # truck
        15: (0, 60, 100),     # bus
        16: (0, 80, 100),     # train
        17: (0, 0, 230),      # motorcycle
        18: (119, 11, 32),    # bicycle
    }

def get_id_to_label() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding semantic labels.

    Returns:
        dict: A dictionary where keys are class IDs (integers) and values are semantic labels (strings).
    """

    return {
        0: 'road',
        1: 'sidewalk',
        2: 'building',
        3: 'wall',
        4: 'fence',
        5: 'pole',
        6: 'light',
        7: 'sign',
        8: 'vegetation',
        9: 'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        255: 'unlabeled'
    }

def label_to_rgb(label:np.ndarray)->PIL.Image:
    """
    Converts a 2D numpy array of class IDs (labels) into an RGB image.

    Args:
        label (np.ndarray): 2D numpy array containing class IDs.
    Returns:
        PIL.Image.Image: RGB image where each pixel corresponds to a color based on class ID.
    """
    
    id_to_color = get_id_to_color()
    color_image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in id_to_color.items():
        color_image[label == class_id] = color
        
    # Set color to black for label 255
    color_image[label == 255] = (0, 0, 0)
    
    return Image.fromarray(color_image, 'RGB')

def get_augmented_data(augmentedType: str) -> A.Compose:
    
    augmentations = {
        'aug1': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(GTA['height'], GTA['width'])  
        ]),
        'aug2': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0), saturation=(0.7, 1.0), hue=0, p=0.5),
            A.Resize(GTA['height'], GTA['width'])  
        ]),
        'aug3': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0), saturation=(0.7, 1.0), hue=0, p=0.5),
            A.GaussianBlur(p=0.5, sigma_limit=(0.2, 0.6)),
            A.Resize(GTA['height'], GTA['width'])              
        ]),
        'aug4': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0), saturation=(0.7, 1.0), hue=0, p=0.5),
            A.GaussianBlur(p=0.5, sigma_limit=(0.2, 0.6)),
            A.RandomResizedCrop(size=(512,1024), ratio=(2,2)),
            A.Rotate(limit=(-10,10), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
            A.Resize(GTA['height'], GTA['width'])  
        ]),
    }
    
    # Return the specified augmentation pipeline if it exists
    if augmentedType in ['aug1', 'aug2', 'aug3', 'aug4']:
        return augmentations[augmentedType]
    else:
        print('Transformation accepted: [aug1, aug2, aug3, aug4]')
        
