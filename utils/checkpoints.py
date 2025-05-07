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
    """
    Saves the model results to a text file.

    Args:
        model_results (List[List[float]]): A list containing model results.
            - model_results[0]: List of training losses.
            - model_results[1]: List of validation losses.
            - model_results[2]: List of training mIoU scores.
            - model_results[3]: List of validation mIoU scores.
            - model_results[4]: List of training IoU scores for each class.
            - model_results[5]: List of validation IoU scores for each class.
        filename (str): The name of the file to save the results in.
        project_step (str): The current project step, used for directory naming.
        model_params_flops (Dict[str, float]): Dictionary containing model parameters and FLOPS.
            - 'Parameters': Number of parameters.
            - 'FLOPS': Floating Point Operations per Second.
        model_latency_fps (Dict[str, float]): Dictionary containing model latency and FPS information.
            - 'mean_latency': Mean latency.
            - 'std_latency': Standard deviation of latency.
            - 'mean_fps': Mean FPS.
            - 'std_fps': Standard deviation of FPS.
    """
    
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
                    project_step: str, 
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
                    verbose: bool)->None:
    """
    Saves the current state of the training process to a checkpoint file.

    Args:
        checkpoint_root (str): The root directory where the checkpoint will be saved.
        project_step (str): The current project step or phase, used for naming the checkpoint file.
        adversarial (bool): Whether to use adversarial training.
        model (torch.nn.Module): The main model whose state is to be saved.
        model_D (torch.nn.Module): The auxiliary or discriminator model whose state is to be saved.
        optimizer (torch.optim.Optimizer): The optimizer for the main model.
        optimizer_D (torch.optim.Optimizer): The optimizer for the auxiliary/discriminator model.
        epoch (int): The current epoch number.
        train_loss_list (List[float]): List of training losses over epochs.
        train_miou_list (List[float]): List of training mean Intersection over Union (mIoU) scores over epochs.
        train_iou (List[float]): List of training IoU scores for each class.
        val_loss_list (List[float]): List of validation losses over epochs.
        val_miou_list (List[float]): List of validation mIoU scores over epochs.
        val_iou (List[float]): List of validation IoU scores for each class.
        verbose (bool): If True, prints a message confirming the checkpoint has been saved.

    Returns:
        None
    """
    # Construct the path for the checkpoint file
    checkpoint_path = f'{output_root}/checkpoint{epoch}.pth'

    
    
    # Save the state of the training process, including model parameters, optimizers, and performance metrics
    if adversarial:
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
                    project_step: str, 
                    adversarial: bool,
                    model: torch.nn.Module, 
                    model_D: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    optimizer_D: torch.optim.Optimizer) -> Tuple[bool, Optional[int], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """
    Loads the checkpoint from the specified directory and restores the model, optimizer, and training state.

    Args:
        checkpoint_root (str): The root directory where the checkpoint is stored.
        project_step (str): The current project step or phase, used for constructing the checkpoint file path.
        adversarial (bool): Whether to use adversarial training.
        model (torch.nn.Module): The main model to load the state dictionary into.
        model_D (torch.nn.Module): The auxiliary or discriminator model to load the state dictionary into.
        optimizer (torch.optim.Optimizer): The optimizer for the main model to load the state dictionary into.
        optimizer_D (torch.optim.Optimizer): The optimizer for the auxiliary/discriminator model to load the state dictionary into.

    Returns:
        Tuple[bool, Optional[int], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
            - bool: Indicates whether to start training from scratch (True) or resume from a checkpoint (False).
            - Optional[int]: The epoch to resume from, if a checkpoint is found.
            - Optional[List[float]]: List of training losses over epochs, if a checkpoint is found.
            - Optional[List[float]]: List of training mean Intersection over Union (mIoU) scores over epochs, if a checkpoint is found.
            - Optional[List[float]]: List of training IoU scores for each class, if a checkpoint is found.
            - Optional[List[float]]: List of validation losses over epochs, if a checkpoint is found.
            - Optional[List[float]]: List of validation mIoU scores over epochs, if a checkpoint is found.
            - Optional[List[float]]: List of validation IoU scores for each class, if a checkpoint is found.
    """

    
    
    # Check if the checkpoint file exists
    if checkpoint_root != None:

        # Construct the path to the checkpoint file
        checkpoint_path = f'{checkpoint_root}/checkpoint.pth'

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Load the state dictionaries into the model, auxiliary model, and optimizers
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if adversarial:
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