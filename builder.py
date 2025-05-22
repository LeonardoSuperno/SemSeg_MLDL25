from models.discriminator.discriminator import FCDiscriminator
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Union
import albumentations as A
from utils.data_processing import get_augmented_data
#from models import BiSeNet, get_deeplab_v2, FCDiscriminator
from models.bisenet.build_bisenet import BiSeNet
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from config import CITYSCAPES, GTA, DEEPLABV2_PATH, CITYSCAPES_PATH, GTA5_PATH
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5

# Build model, optimizer and loss_fn
def build_model(model_name: str, 
             n_classes: int,
             device: str,
             parallelize: bool,
             optimizer_name: str, 
             lr: float,
             momentum: float,
             weight_decay: float,
             loss_fn_name: str,
             ignore_index: int,
             adversarial: bool) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.nn.Module]:
    """
    Set up components for semantic segmentation model training.

    Args:
    - model_name (str): Name of the segmentation model ('DeepLabV2' or 'BiSeNet').
    - n_classes (int): Number of classes in the dataset.
    - device (str): Device to run the model on ('cpu' or 'cuda').
    - parallelize (bool): Whether to use DataParallel for multi-GPU training.
    - optimizer_name (str): Name of the optimizer ('Adam' or 'SGD').
    - lr (float): Learning rate for the optimizer.
    - momentum (float): Momentum factor for SGD optimizer.
    - weight_decay (float): Weight decay (L2 penalty) for the optimizer.
    - loss_fn_name (str): Name of the loss function ('CrossEntropyLoss').
    - ignore_index (int): Index to ignore in loss computation.
    - adversarial (bool): Whether to include adversarial training components.

    Raises:
    - ValueError: If an invalid model_name, optimizer_name, or loss_fn_name is provided.

    Returns:
    - Tuple containing:
        - model (nn.Module): Segmentation model.
        - optimizer (torch.optim.Optimizer): Optimizer for the segmentation model.
        - loss_fn (nn.Module): Loss function for the segmentation model.
        - model_D (nn.Module or None): Discriminator model for adversarial training (if adversarial=True).
        - optimizer_D (torch.optim.Optimizer or None): Optimizer for the discriminator model (if adversarial=True).
        - loss_D (nn.Module or None): Loss function for the discriminator model (if adversarial=True).
    """
    
    model = None
    optimizer = None
    loss_fn = None
    model_D = None
    optimizer_D = None
    loss_D = None
    
    # Initialize segmentation model based on model_name
    if model_name == 'DeepLabV2':
        model = get_deeplab_v2(num_classes=n_classes, pretrain=True, pretrain_model_path=DEEPLABV2_PATH).to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
    elif model_name == 'BiSeNet':
        model = BiSeNet(num_classes=n_classes, context_path="resnet18").to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
    else:
        raise ValueError('Model accepted: [DeepLabV2, BiSeNet]')
            
    # Initialize optimizer based on optimizer_name
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer accepted: [Adam, SGD]')
        
    # Initialize loss function based on loss_fn_name
    if loss_fn_name == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError('Loss function accepted: [CrossEntropyLoss]')
    
    # Initialize adversarial components if adversarial is True
    if adversarial:
        model_D = FCDiscriminator(num_classes=n_classes).to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model_D = torch.nn.DataParallel(model_D).to(device)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-3, betas=(0.9, 0.99))
        loss_D = torch.nn.BCEWithLogitsLoss()
        
    return model, optimizer, loss_fn, model_D, optimizer_D, loss_D

def build_loaders(train_dataset_name: str, 
                val_dataset_name: str, 
                augmented: bool,
                augmentedType: str,
                batch_size: int,
                n_workers: int,
                adversarial: bool) -> Tuple[Union[DataLoader, Tuple[DataLoader, DataLoader]], DataLoader, int, int]:
    """
    Set up data loaders for training and validation datasets in semantic segmentation.

    Args:
    - train_dataset_name (str): Name of the training dataset ('CityScapes' or 'GTA5').
    - val_dataset_name (str): Name of the validation dataset ('CityScapes').
    - augmented (bool): Whether to use augmented data.
    - augmentedType (str): Type of augmentation to apply (specific to your implementation).
    - batch_size (int): Batch size for data loaders.
    - n_workers (int): Number of workers for data loading.
    - adversarial (bool): Whether to set up adversarial training data loaders.

    Raises:
    - ValueError: If an invalid train_dataset_name or val_dataset_name is provided.

    Returns:
    - Tuple containing:
        - train_loader (Union[DataLoader, Tuple[DataLoader, DataLoader]]): DataLoader(s) for the training dataset.
        - val_loader (DataLoader): DataLoader for the validation dataset.
        - data_height (int): Height of the dataset images.
        - data_width (int): Width of the dataset images.
    """

    transform_cityscapes = A.Compose([
        A.Resize(CITYSCAPES['height'], CITYSCAPES['width']),
    ])
    transform_gta5 = A.Compose([
        A.Resize(GTA['height'], GTA['width'])   
    ])

    train_loader = None
    val_loader = None
    data_height = None
    data_width = None
    
    if augmented:
        transform_gta5 = get_augmented_data(augmentedType)
    
    if adversarial:
        source_dataset = GTA5(root_dir=GTA5_PATH, transform=transform_gta5)
        target_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='train', transform=transform_cityscapes)

        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

        train_loader = (source_loader, target_loader)
    else:
        if train_dataset_name == 'CityScapes':
            train_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='train', transform=transform_cityscapes)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        elif train_dataset_name == 'GTA5':
            train_dataset = GTA5(root_dir=GTA5_PATH, transform=transform_gta5)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        else:
            raise ValueError('Train datasets accepted: [CityScapes, GTA5]')
        
    if val_dataset_name == 'CityScapes':
        val_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='val', transform=transform_cityscapes)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
        data_height = CITYSCAPES['height']
        data_width = CITYSCAPES['width']
    else:
        raise ValueError('Val datasets accepted: [CityScapes]')
    
    return train_loader, val_loader, data_height, data_width