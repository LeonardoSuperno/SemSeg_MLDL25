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
          optimizer: torch.optim.Optimizer, 
          optimizer_D: torch.optim.Optimizer, 
          ce_loss: torch.nn.Module, 
          bce_loss: torch.nn.Module, 
          dataloaders: Tuple[DataLoader,DataLoader], 
          val_loader: DataLoader, 
          epochs: int, 
          device: str, 
          output_root: str,
          checkpoint_root: str,
          verbose: bool,
          n_classes: int = 19,
          power: float = 0.9,
          adversarial: bool = True,
          multi_level: bool = False) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    
    # Load or initialize checkpoint
    no_checkpoint, start_epoch, train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou = load_checkpoint(checkpoint_root=checkpoint_root, adversarial=adversarial, model=model, model_D=model_D, optimizer=optimizer, optimizer_D=optimizer_D)
        
    if no_checkpoint:
        train_loss_list, train_miou_list = [], []
        val_loss_list, val_miou_list = [], []
        start_epoch = 0
    
    for epoch in (range(start_epoch, epochs)):

        # labels for adversarial training
        adversarial_source_label = 0
        adversarial_target_label = 1
        
        interp_source = nn.Upsample(size=(GTA['height'], GTA['width']), mode='bilinear')
        interp_target = nn.Upsample(size=(CITYSCAPES['height'], CITYSCAPES['width']), mode='bilinear')
        
        lambda_adv = 0.001
        total_loss = 0
        total_miou = 0
        total_iou = np.zeros(n_classes)
        
        iterations = 0
        
        model.train()
        model_D.train()
        
        source_loader, target_loader = dataloaders        
        
        for (source_data, source_label), (target_data, _) in zip(source_loader, cycle(target_loader)):
            
            iterations+=1

            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)
            
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            #TRAIN GENERATOR
            
            #Train Segmentation (only source)
            for param in model_D.parameters():
                param.requires_grad = False # The Discriminator parameters don't receive the gradients, so the weights remain fixed
            
            source_output = interp_source(model(source_data))

            segmentation_loss = ce_loss(source_output, source_label)
            segmentation_loss.backward()

            #Train to segment train images like source images
            output_target = interp_target(model(target_data))
            
            prediction_target = torch.nn.functional.softmax(output_target)
            discriminator_output_target = model_D(prediction_target)
            discriminator_label_source = torch.FloatTensor(discriminator_output_target.data.size()).fill_(adversarial_source_label).cuda() # 0 = source domain
            
            adversarial_loss = bce_loss(discriminator_output_target, discriminator_label_source) 
            discriminator_loss = lambda_adv * adversarial_loss
            discriminator_loss.backward() # Only for the Generator
            
            
            #TRAIN DISCRIMINATOR
            
            #Train with source
            for param in model_D.parameters():
                param.requires_grad = True
                
            source_output = source_output.detach()
            
            prediction_source = torch.nn.functional.softmax(source_output)
            discriminator_output_source = model_D(prediction_source)
            discriminator_label_source = torch.FloatTensor(discriminator_output_source.data.size()).fill_(adversarial_source_label).cuda()
            discriminator_loss_source = bce_loss(discriminator_output_source, discriminator_label_source)
            discriminator_loss_source.backward()

            #Train with target
            output_target = output_target.detach()
            
            prediction_target = torch.nn.functional.softmax(output_target)
            discriminator_output_target = model_D(prediction_target)
            discriminator_label_target = torch.FloatTensor(discriminator_output_target.data.size()).fill_(adversarial_target_label).cuda()
            
            discriminator_loss_target = bce_loss(discriminator_output_target, discriminator_label_target)
            discriminator_loss_target.backward()
            
            optimizer.step()
            optimizer_D.step()
            
            total_loss += segmentation_loss.item()
            
            prediction_source = torch.argmax(torch.softmax(source_output, dim=1), dim=1)
            hist = fast_hist(source_label.cpu().numpy(), prediction_source.cpu().numpy(), n_classes)
            running_iou = np.array(per_class_iou(hist)).flatten()
            total_miou += running_iou.sum()
            total_iou += running_iou

            
        train_loss = total_loss / iterations
        train_miou = total_miou / (iterations * n_classes)
        train_iou = total_iou / iterations

        # Validation

        total_loss = 0
        total_miou = 0
        total_iou = np.zeros(n_classes)
        
        model.eval()

        with torch.inference_mode(): # which is analogous to torch.no_grad
            for image, label in val_loader:        
                image, label = image.to(device), label.type(torch.LongTensor).to(device)
                
                output = model(image)
                loss = ce_loss(output, label)
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
                          lr_decay_iter=1, 
                          max_iter=epochs,
                          power=power)
        
        # Save checkpoint after each epoch
        save_checkpoint(output_root=output_root, 
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
                        verbose=verbose,
                        multi_level=multi_level)


    return train_loss_list, val_loss_list, train_miou_list, val_miou_list, train_iou, val_iou
    

