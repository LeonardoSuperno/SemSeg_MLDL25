import os
import torch
from typing import List, Dict, Tuple, Optional
from utils.data_processing import get_id_to_label
from config import OUTPUT_ROOT

def save_results(model_results: List[List[float]], 
                 filename: str,
                 project_step: str,
                 model_params_flops: Dict[str, float],
                 model_latency_fps: Dict[str, float]) -> None:
    
    # Construct the checkpoint path
    checkpoint_path = f'{OUTPUT_ROOT}/{project_step}'
    
    # Create the directory if it does not exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Open the file for writing
    with open(f"{checkpoint_path}/{filename}.txt", 'w') as file:
        # Write model parameters and FLOPS
        file.write(f"Parameters: {model_params_flops['Parameters']}\n")
        file.write(f"FLOPS: {model_params_flops['FLOPS']}\n")
        
        # Write latency information
        file.write("Latency:\n")
        file.write(f"\tmean: {model_latency_fps['mean_latency']}\n")
        file.write(f"\tstd: {model_latency_fps['std_latency']}\n")
        
        # Write FPS information
        file.write("FPS:\n")
        file.write(f"\tmean: {model_latency_fps['mean_fps']}\n")
        file.write(f"\tstd: {model_latency_fps['std_fps']}\n")
        
        # Write loss information
        file.write("Loss:\n")
        file.write(f"\ttrain: {model_results[0][-1]}\n")
        file.write(f"\tval: {model_results[1][-1]}\n")
        
        # Write mIoU information
        file.write("mIoU:\n")
        file.write(f"\ttrain: {model_results[2][-1]}\n")
        file.write(f"\tval: {model_results[3][-1]}\n")
        
        # Write training IoU for each class
        file.write("Training IoU for class:\n")
        for i, iou in enumerate(model_results[4]):
            file.write(f"{get_id_to_label()[i]}: {iou}\n")
        
        # Write validation IoU for each class
        file.write("\nValidation IoU for class:\n")
        for i, iou in enumerate(model_results[5]):
            file.write(f"{get_id_to_label()[i]}: {iou}\n")

def save_checkpoint(output_root: str, 
                    adversarial: bool,
                    model: torch.nn.Module, 
                    model_D: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    optimizer_D: torch.optim.Optimizer, 
                    epoch: int,
                    train_loss_list: List[float], 
                    train_miou_list: List[float],
                    train_iou: List[float],
                    val_loss_list: List[float],
                    val_miou_list: List[float],
                    val_iou: List[float],
                    verbose: bool,
                    multi_level: bool = False)->None:
    
    # Construct the path for the checkpoint file
    checkpoint_path = f'{output_root}/checkpoint{epoch}.pth'

    
    
    # Save the state of the training process, including model parameters, optimizers, and performance metrics
    if adversarial:
        if multi_level:
            torch.save({
            'model': model.state_dict(),
            'model_D1': model_D[0].state_dict(),
            'model_D2': model_D[1].state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_D1': optimizer_D[0].state_dict(),
            'optimizer_D2': optimizer_D[1].state_dict(),
            'epoch': epoch + 1,
            'train_loss_list': train_loss_list,
            'train_miou_list': train_miou_list,
            'train_iou': train_iou,
            'val_loss_list': val_loss_list,
            'val_miou_list': val_miou_list,
            'val_iou': val_iou
        }, checkpoint_path)
        else:
            torch.save({
            'model': model.state_dict(),
            'model_D': model_D.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'epoch': epoch + 1,
            'train_loss_list': train_loss_list,
            'train_miou_list': train_miou_list,
            'train_iou': train_iou,
            'val_loss_list': val_loss_list,
            'val_miou_list': val_miou_list,
            'val_iou': val_iou
        }, checkpoint_path)
    else:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss_list': train_loss_list,
            'train_miou_list': train_miou_list,
            'train_iou': train_iou,
            'val_loss_list': val_loss_list,
            'val_miou_list': val_miou_list,
            'val_iou': val_iou
        }, checkpoint_path)
    
    # If verbose is True, print a confirmation message
    if verbose == True:
        print(f"Checkpoint saved in {checkpoint_path}")
    
def load_checkpoint(checkpoint_root: str, 
                    adversarial: bool,
                    model: torch.nn.Module, 
                    model_D: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    optimizer_D: torch.optim.Optimizer,
                    multi_level:bool = False) -> Tuple[bool, Optional[int], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    
    
    # Check if the checkpoint file exists
    if checkpoint_root != None and os.path.exists(checkpoint_root):

        # Construct the path to the checkpoint file
        checkpoint_path = f'{checkpoint_root}/checkpoint.pth'

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Load the state dictionaries into the model, auxiliary model, and optimizers
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if adversarial:
            if multi_level:
                model_D[0].load_state_dict(checkpoint['model_D1'])
                model_D[1].load_state_dict(checkpoint['model_D2'])
                optimizer_D[0].load_state_dict(checkpoint['optimizer_D1'])
                optimizer_D[1].load_state_dict(checkpoint['optimizer_D2'])
            else:
                model_D.load_state_dict(checkpoint['model_D'])
                optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        
        # Extract training state information
        start_epoch = checkpoint['epoch']
        train_loss_list = checkpoint['train_loss_list']
        train_miou_list = checkpoint['train_miou_list']
        train_iou = checkpoint['train_iou']
        val_loss_list = checkpoint['val_loss_list']
        val_miou_list = checkpoint['val_miou_list']
        val_iou = checkpoint['val_iou']
        
        # Print a message indicating the checkpoint was found and loaded
        print(f"Checkpoint found. Resuming from epoch {start_epoch}.")
        
        # Return the state indicating that training can resume from the checkpoint
        return (False, start_epoch, train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou)
    
    else:
        
        print(f"No checkpoint found. Starting from scratch.")
        
        # Return the state indicating that training should start from scratch
        return (True, None, None, None, None, None, None, None)