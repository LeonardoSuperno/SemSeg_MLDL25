from models.discriminator.discriminator import FCDiscriminator
import numpy as np
import torch
import torch.nn as nn
from utils.losses import FocalLoss, DiceLoss
from torch.utils.data import DataLoader
from typing import Tuple, Union
import albumentations as A
from utils.data_processing import get_augmented_data
from models.bisenet.build_bisenet import BiSeNet
from models.bisenet.build_multy_bisenet import Multy_BiSeNet
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from config import CITYSCAPES, GTA, DEEPLABV2_PATH, CITYSCAPES_PATH, GTA5_PATH
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5


# Build segmentation model, optimizer, loss functions, and optionally adversarial components
def build_model(model_name: str, 
                n_classes: int,
                device: str,
                parallelize: bool,
                lr: float,
                loss_fn_name: str,
                ignore_index: int,
                adversarial: bool,
                multy_level: bool,
                extra_loss_name: str,
                feature: str) -> Tuple[
                    torch.nn.Module, torch.optim.Optimizer, torch.nn.Module,
                    Union[torch.nn.Module, Tuple[torch.nn.Module, torch.nn.Module]],
                    Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]],
                    torch.nn.Module, torch.nn.Module]:

    # Initialize model
    if model_name == 'DeepLabV2':
        model = get_deeplab_v2(num_classes=n_classes, pretrain=True, pretrain_model_path=DEEPLABV2_PATH).to(device)
    elif model_name == 'BiSeNet':
        if multy_level:
            model = Multy_BiSeNet(num_classes=n_classes, context_path="resnet18", feature=feature).to(device)
        else:
            model = BiSeNet(num_classes=n_classes, context_path="resnet18").to(device)
    else:
        raise ValueError('Model accepted: [DeepLabV2, BiSeNet]')

    # Enable model parallelization if possible
    if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize loss function
    if loss_fn_name == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError('Loss function accepted: [CrossEntropyLoss]')

    # Initialize adversarial training components
    if adversarial:
        # Additional auxiliary loss
        if extra_loss_name == 'FocalLoss':
            extra_loss_fn = FocalLoss(num_class=n_classes, ignore_label=ignore_index)
        elif extra_loss_name == 'DiceLoss':
            extra_loss_fn = DiceLoss(num_classes=n_classes, ignore_index=ignore_index)
        elif extra_loss_name == "None":
            extra_loss_fn = None
        else:
            raise ValueError('Extra loss function accepted: [FocalLoss, DiceLoss, None]')

        # Multi-level setup uses two discriminators
        if multy_level:
            model_D1 = FCDiscriminator(num_classes=n_classes).to(device)
            model_D2 = FCDiscriminator(num_classes=n_classes).to(device)
            if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
                model_D1 = torch.nn.DataParallel(model_D1).to(device)
                model_D2 = torch.nn.DataParallel(model_D2).to(device)
            model_D = (model_D1, model_D2)
            optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=1e-3, betas=(0.9, 0.99))
            optimizer_D2 = torch.optim.Adam(model_D2.parameters(), lr=1e-3, betas=(0.9, 0.99))
            optimizer_D = (optimizer_D1, optimizer_D2)
        else:
            # Single discriminator setup
            model_D = FCDiscriminator(num_classes=n_classes).to(device)
            if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
                model_D = torch.nn.DataParallel(model_D).to(device)
            optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-3, betas=(0.9, 0.99))

        loss_D = torch.nn.BCEWithLogitsLoss()
    else:
        model_D = None
        optimizer_D = None
        loss_D = None
        extra_loss_fn = None

    return model, optimizer, loss_fn, model_D, optimizer_D, loss_D, extra_loss_fn


# Build training and validation dataloaders
def build_loaders(train_dataset_name: str, 
                  val_dataset_name: str, 
                  augmented: bool,
                  augmentedType: str,
                  batch_size: int,
                  n_workers: int,
                  adversarial: bool) -> Tuple[
                      Union[DataLoader, Tuple[DataLoader, DataLoader]], DataLoader, int, int]:

    # Define transformations to resize images 
    transform_cityscapes = A.Compose([A.Resize(CITYSCAPES['height'], CITYSCAPES['width'])])
    transform_gta5 = A.Compose([A.Resize(GTA['height'], GTA['width'])])

    # Apply data augmentation if requested
    if augmented:
        transform_gta5 = get_augmented_data(augmentedType)

    # Adversarial training uses both source and target domains
    if adversarial:
        source_dataset = GTA5(root_dir=GTA5_PATH, transform=transform_gta5)
        target_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='train', transform=transform_cityscapes)
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        train_loader = (source_loader, target_loader)
    else:
        # Single domain training
        if train_dataset_name == 'CityScapes':
            train_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='train', transform=transform_cityscapes)
        elif train_dataset_name == 'GTA5':
            train_dataset = GTA5(root_dir=GTA5_PATH, transform=transform_gta5)
        else:
            raise ValueError('Train datasets accepted: [CityScapes, GTA5]')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    # Validation loader (only CityScapes supported)
    if val_dataset_name == 'CityScapes':
        val_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='val', transform=transform_cityscapes)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
        data_height = CITYSCAPES['height']
        data_width = CITYSCAPES['width']
    else:
        raise ValueError('Val datasets accepted: [CityScapes]')

    return train_loader, val_loader, data_height, data_width


# Build test dataloader for output saving
def build_test_loader(test_dataset_name: str,
                      batch_size: int,
                      n_workers: int) -> Tuple[DataLoader, int, int]:

    transform_cityscapes = A.Compose([A.Resize(CITYSCAPES['height'], CITYSCAPES['width'])])

    if test_dataset_name == 'CityScapes':
        test_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='val', transform=transform_cityscapes)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
        data_height = CITYSCAPES['height']
        data_width = CITYSCAPES['width']
    else:
        raise ValueError('Test datasets accepted: [CityScapes]')

    return test_loader, data_height, data_width
