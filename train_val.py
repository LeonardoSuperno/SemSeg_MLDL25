# TODO: Define here your training and validation loops.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from models import BiSeNet, get_deeplab_v2
from typing import Tuple, List, Union
from utils.metrics import *
from utils.optimization import *
from utils.visualization import *
from utils.checkpoints import *

def train_val(model: torch.nn.Module, 
          model_D: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, 
          optimizer_D: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module, 
          loss_D: torch.nn.Module, 
          train_loader: Union[DataLoader, Tuple[DataLoader,DataLoader]],  
          val_loader: DataLoader, 
          epochs: int, 
          device: str, 
          output_root: str,
          checkpoint_root: str,
          project_step: str,
          verbose: bool,
          n_classes: int = 19,
          power: float = 0.9,
          adversarial: bool = False) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Train a semantic segmentation model with optional adversarial training.

    Args:
        model (torch.nn.Module): Semantic segmentation model.
        model_D (torch.nn.Module): Discriminator model for adversarial training.
        optimizer (torch.optim.Optimizer): Optimizer for the segmentation model.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        loss_fn (torch.nn.Module): Loss function for segmentation.
        loss_D (torch.nn.Module): Loss function for adversarial training.
        train_loader (Union[DataLoader, Tuple[DataLoader,DataLoader]]): DataLoader(s) for the training dataset.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        device (str): Device on which to run computations ('cuda' or 'cpu').
        checkpoint_root (str): Root directory to save checkpoints.
        project_step (str): Name/id of the project or step.
        verbose (bool): Whether to print verbose training statistics.
        n_classes (int, optional): Number of classes for segmentation. Defaults to 19.
        power (float, optional): Power parameter for learning rate scheduler. Defaults to 0.9.
        adversarial (bool, optional): Whether to use adversarial training. Defaults to False.
Returns:
        Tuple containing lists of:
        - train_loss_list (List[float]): List of training losses per epoch.
        - val_loss_list (List[float]): List of validation losses per epoch.
        - train_miou_list (List[float]): List of training mIoU per epoch.
        - val_miou_list (List[float]): List of validation mIoU per epoch.
        - train_iou (List[float]): List of per-class IoU for training per epoch.
        - val_iou (List[float]): List of per-class IoU for validation per epoch.
    """
    # Load or initialize checkpoint
    no_checkpoint, start_epoch, train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou = load_checkpoint(checkpoint_root=checkpoint_root, project_step=project_step, adversarial=adversarial, model=model, model_D=model_D, optimizer=optimizer, optimizer_D=optimizer_D)
        
    if no_checkpoint:
        train_loss_list, train_miou_list = [], []
        val_loss_list, val_miou_list = [], []
        start_epoch = 0

    for epoch in range(start_epoch, epochs):

        # Train
        total_loss = 0
        total_miou = 0
        total_iou = np.zeros(n_classes)
        
        model.train()
        
        
        #for image, label in train_loader:
        for i, (image, label) in enumerate(train_loader):  # SOLO PER DEBUG ELIMINARE
            if i >= 5:
                break
            image, label = image.to(device), label.type(torch.LongTensor).to(device)
        
            output = model(image)
            loss = loss_fn(output, label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)

            hist = fast_hist(label.cpu().numpy(), prediction.cpu().numpy(), n_classes)
            running_iou = np.array(per_class_iou(hist)).flatten() 
            total_miou += running_iou.sum()
            total_iou += running_iou
        
        train_loss = total_loss / len(train_loader)
        train_miou = total_miou / (len(train_loader)* n_classes)
        train_iou = total_iou / len(train_loader)

        # Validation

        total_loss = 0
        total_miou = 0
        total_iou = np.zeros(n_classes)
        
        model.eval()

        with torch.inference_mode(): # which is analogous to torch.no_grad
            # for image, label in val_loader:
            for i, (image, label) in enumerate(val_loader):  # SOLO PER DEBUG ELIMINARE
                if i >= 5:
                    break
                image, label = image.to(device), label.type(torch.LongTensor).to(device)
                
                output = model(image)
                loss = loss_fn(output, label)
                total_loss += loss.item()
                
                prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
                
                hist = fast_hist(label.cpu().numpy(), prediction.cpu().numpy(), n_classes)
                running_iou = np.array(per_class_iou(hist)).flatten()
                total_miou += running_iou.sum()
                total_iou += running_iou
        
        val_loss = total_loss / len(val_loader)
        val_miou = total_miou / (len(val_loader)* n_classes)
        val_iou = total_iou / len(val_loader)

        # Append metrics to lists
        train_loss_list.append(train_loss) 
        train_miou_list.append(train_miou) 
        val_loss_list.append(val_loss)
        val_miou_list.append(val_miou)

        # Print statistics if verbose 
        print_stats(epoch=epoch, 
                    train_loss=train_loss,
                    val_loss=val_loss, 
                    train_miou=train_miou, 
                    val_miou=val_miou, 
                    verbose=verbose)

        # Adjust learning rate
        poly_lr_scheduler(optimizer=optimizer,
                          init_lr=optimizer.param_groups[0]['lr'],
                          iter=epoch, 
                          max_iter=epochs,
                          power=power)
        
        # Save checkpoint after each epoch
        save_checkpoint(output_root=output_root, 
                        project_step=project_step,
                        adversarial=adversarial,
                        model=model, 
                        model_D=model_D,
                        optimizer=optimizer, 
                        optimizer_D=optimizer_D, 
                        epoch=epoch,
                        train_loss_list=train_loss_list, 
                        train_miou_list=train_miou_list,
                        train_iou=train_iou,
                        val_loss_list=val_loss_list,
                        val_miou_list=val_miou_list,
                        val_iou=val_iou,
                        verbose=verbose)


    return train_loss_list, val_loss_list, train_miou_list, val_miou_list, train_iou, val_iou