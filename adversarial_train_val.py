import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List, Union
from utils.metrics import *
from utils.optimization import *
from utils.visualization import *
from utils.checkpoints import *
from config import GTA, CITYSCAPES
from itertools import cycle


def adversarial_train_val(model: torch.nn.Module, 
                           model_D: torch.nn.Module, 
                           loss_fn: torch.nn.Module, 
                           loss_D: torch.nn.Module, 
                           optimizer: torch.optim.Optimizer, 
                           optimizer_D: torch.optim.Optimizer, 
                           dataloaders: Tuple[DataLoader,DataLoader], 
                           device: str, 
                           n_classes: int = 19)-> Tuple[float, float, float]:
    """
    Perform a single adversarial training step for semantic segmentation.

    Args:
    - model (torch.nn.Module): Segmentation model.
    - model_D (torch.nn.Module): Discriminator model.
    - loss_fn (torch.nn.Module): Segmentation loss function.
    - loss_D (torch.nn.Module): Adversarial loss function for discriminator.
    - optimizer (torch.optim.Optimizer): Optimizer for segmentation model.
    - optimizer_D (torch.optim.Optimizer): Optimizer for discriminator model.
    - dataloaders (Tuple[DataLoader,DataLoader]): Source and target dataloaders for training data.
    - device (str): Device on which to run the models ('cuda' or 'cpu').
    - n_classes (int, optional): Number of classes for segmentation. Default is 19.

    Returns:
    - Tuple containing:
        - epoch_loss (float): Average segmentation loss for the epoch.
        - epoch_miou (float): Mean Intersection over Union (mIoU) for the epoch.
        - epoch_iou (np.ndarray): Array of per-class IoU values for the epoch.
    """

    model_G = model.to(device)
    optimizer_G = optimizer
    ce_loss = loss_fn
    bce_loss = loss_D

    # labels for adversarial training
    source_label = 0
    target_label = 1
    
    interp_source = nn.Upsample(size=(GTA['height'], GTA['width']), mode='bilinear')
    interp_target = nn.Upsample(size=(CITYSCAPES['height'], CITYSCAPES['width']), mode='bilinear')
    
    lambda_adv = 0.001
    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(n_classes)
    
    iterations = 0
    
    model_G.train()
    model_D.train()
    
    source_loader, target_loader = dataloaders
    train_loader = zip(source_loader, cycle(target_loader)) # make target_loader cycle in order to match length of source_loader
    
    
    for (source_data, source_labels), (target_data, _) in train_loader:
        
        iterations+=1

        source_data, source_labels = source_data.to(device), source_labels.to(device)
        target_data = target_data.to(device)
        
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        #TRAIN GENERATOR
        
        #Train Segmentation (only source)
        for param in model_D.parameters():
            param.requires_grad = False # The Discriminator parameters don't receive the gradients, so the weights remain fixed
        
        output_source = model_G(source_data)
        output_source = interp_source(output_source) # apply upsample

        segmentation_loss = ce_loss(output_source, source_labels)
        segmentation_loss.backward()

        #Train to fool the Discriminator (source + target)
        output_target = model_G(target_data)
        output_target = interp_target(output_target) # apply upsample
        
        prediction_target = torch.nn.functional.softmax(output_target)
        discriminator_output_target = model_D(prediction_target)
        discriminator_label_source = torch.FloatTensor(discriminator_output_target.data.size()).fill_(source_label).cuda() # 0 = source domain
        
        adversarial_loss = bce_loss(discriminator_output_target, discriminator_label_source) # The generator try to fool the discriminator
        discriminator_loss = lambda_adv * adversarial_loss
        discriminator_loss.backward() # Only for the Generator
        
        
        #TRAIN DISCRIMINATOR
        
        #Train with source
        for param in model_D.parameters():
            param.requires_grad = True
            
        output_source = output_source.detach()
        
        prediction_source = torch.nn.functional.softmax(output_source)
        discriminator_output_source = model_D(prediction_source)
        discriminator_label_source = torch.FloatTensor(discriminator_output_source.data.size()).fill_(source_label).cuda() # 0 = source domain
        discriminator_loss_source = bce_loss(discriminator_output_source, discriminator_label_source)
        discriminator_loss_source.backward()

        #Train with target
        output_target = output_target.detach()
        
        prediction_target = torch.nn.functional.softmax(output_target)
        discriminator_output_target = model_D(prediction_target)
        discriminator_label_target = torch.FloatTensor(discriminator_output_target.data.size()).fill_(target_label).cuda() # 1 = target domain
        
        discriminator_loss_target = bce_loss(discriminator_output_target, discriminator_label_target)
        discriminator_loss_target.backward()
        
        optimizer_G.step()
        optimizer_D.step()
        
        total_loss += segmentation_loss.item()
        
        prediction_source = torch.argmax(torch.softmax(output_source, dim=1), dim=1)
        hist = fast_hist(source_labels.cpu().numpy(), prediction_source.cpu().numpy(), n_classes)
        running_iou = np.array(per_class_iou(hist)).flatten()
        total_miou += running_iou.sum()
        total_iou += running_iou

        
    epoch_loss = total_loss / iterations
    epoch_miou = total_miou / (iterations * n_classes)
    epoch_iou = total_iou / iterations
    
    return epoch_loss, epoch_miou, epoch_iou

