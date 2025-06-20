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


def multi_adversarial_train_val(model: torch.nn.Module, 
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
          multi_level: bool = True) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    
    # Load or initialize checkpoint
    no_checkpoint, start_epoch, train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou = load_checkpoint(checkpoint_root=checkpoint_root, adversarial=adversarial, model=model, model_D=model_D, optimizer=optimizer, optimizer_D=optimizer_D, multi_level=multi_level)
        
    if no_checkpoint:
        train_loss_list, train_miou_list = [], []
        val_loss_list, val_miou_list = [], []
        start_epoch = 0

    model_D1, model_D2 = model_D
    optimizer_D1, optimizer_D2 = optimizer_D
    
    for epoch in (range(start_epoch, epochs)):

        # labels for adversarial training
        adversarial_source_label = 0
        adversarial_target_label = 1
        
        interp_source = nn.Upsample(size=(GTA['height'], GTA['width']), mode='bilinear')
        interp_target = nn.Upsample(size=(CITYSCAPES['height'], CITYSCAPES['width']), mode='bilinear')
        
        lambda_seg = 0.1
        lambda_adv1 = 0.0002
        lambda_adv2 = 0.001        
        total_loss = 0
        total_miou = 0
        total_iou = np.zeros(n_classes)
        
        iterations = 0
        
        model.train()
        model_D1.train()
        model_D2.train()
        
        source_loader, target_loader = dataloaders        
        
        for (source_data, source_label), (target_data, _) in zip(source_loader, cycle(target_loader)):
            
            iterations+=1

            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)
            
            optimizer.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()

            #TRAIN GENERATOR
            
            #Train Segmentation (only source)
            for param in model_D1.parameters():
                param.requires_grad = False # The Discriminator parameters don't receive the gradients, so the weights remain fixed
            for param in model_D2.parameters():
                param.requires_grad = False # The Discriminator parameters don't receive the gradients, so the weights remain fixed


            source_output1, source_output2 = model(source_data)

            source_output1 = interp_source(source_output1)
            source_output2 = interp_source(source_output2)

            segmentation_loss1 = ce_loss(source_output1, source_label)
            segmentation_loss2 = ce_loss(source_output2, source_label)

            segmentation_loss = segmentation_loss2 + lambda_seg * segmentation_loss1

            # loss = loss / args.iter_size NORMALIZZAZIONE???
            segmentation_loss.backward()


            #Train to segment train images like source images
            output_target1, output_target2 = model(target_data)

            output_target1 = interp_target(output_target1)
            output_target2 = interp_target(output_target2)
            
            prediction_target1 = torch.nn.functional.softmax(output_target1)
            prediction_target2 = torch.nn.functional.softmax(output_target2)

            discriminator_output_target1 = model_D1(prediction_target1)
            discriminator_output_target2 = model_D2(prediction_target2)
            discriminator_label_source = torch.FloatTensor(discriminator_output_target1.data.size()).fill_(adversarial_source_label).cuda() # 0 = source domain
            
            adversarial_loss1 = bce_loss(discriminator_output_target1, discriminator_label_source) 
            adversarial_loss2 = bce_loss(discriminator_output_target2, discriminator_label_source) 

            discriminator_loss = lambda_adv1 * adversarial_loss1 + lambda_adv2 * adversarial_loss2
        
            discriminator_loss.backward() # Only for the Generator
            
            
            #TRAIN DISCRIMINATOR
            
            #Train with source
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True
                
            source_output1 = source_output1.detach()
            source_output2 = source_output2.detach()
            
            prediction_source1 = torch.nn.functional.softmax(source_output1)
            prediction_source2 = torch.nn.functional.softmax(source_output2)

            discriminator_output_source1 = model_D1(prediction_source1)
            discriminator_output_source2 = model_D2(prediction_source2)

            discriminator_label_source = torch.FloatTensor(discriminator_output_source1.data.size()).fill_(adversarial_source_label).cuda()
    
            discriminator_loss_source1 = bce_loss(discriminator_output_source1, discriminator_label_source)
            discriminator_loss_source2 = bce_loss(discriminator_output_source2, discriminator_label_source)

            discriminator_loss_source1 = discriminator_loss_source1 / 2
            discriminator_loss_source2 = discriminator_loss_source2 / 2

            discriminator_loss_source1.backward()
            discriminator_loss_source2.backward()

            #Train with target
            output_target1 = output_target1.detach()
            output_target2 = output_target2.detach()
            
            prediction_target1 = torch.nn.functional.softmax(output_target1)
            prediction_target2 = torch.nn.functional.softmax(output_target2)
            
            discriminator_output_target1 = model_D1(prediction_target1)
            discriminator_output_target2 = model_D2(prediction_target2)

            discriminator_label_target = torch.FloatTensor(discriminator_output_target1.data.size()).fill_(adversarial_target_label).cuda()
            
            discriminator_loss_target1 = bce_loss(discriminator_output_target1, discriminator_label_target)
            discriminator_loss_target2 = bce_loss(discriminator_output_target2, discriminator_label_target)

            discriminator_loss_target1 = discriminator_loss_target1 / 2
            discriminator_loss_target2 = discriminator_loss_target2 / 2
            
            discriminator_loss_target1.backward()
            discriminator_loss_target2.backward()
            
            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()
            
            total_loss += segmentation_loss.item()
            
            prediction_source = torch.argmax(torch.softmax(source_output2, dim=1), dim=1)
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
                
                _, output = model(image)
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
    

