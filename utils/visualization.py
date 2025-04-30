import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random
from collections import OrderedDict
from typing import Tuple
from utils.data_processing import get_id_to_label, label_to_rgb
from datasets.cityscapes import CityScapes
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from models.bisenet.build_bisenet import BiSeNet
from config import OUTPUT_ROOT, CITYSCAPES_PATH, DEEPLABV2_PATH

def print_stats(epoch:int, 
                train_loss:float,
                val_loss:float, 
                train_miou:float, 
                val_miou:float, 
                verbose:bool)->None:
    """
    Print training and validation statistics if verbose is True.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss value.
        val_loss (float): Validation loss value.
        train_miou (float): Training mean IoU value.
        val_miou (float): Validation mean IoU value.
        verbose (bool): Flag to control verbosity. If False, no output is printed.

    Returns:
        None
    """
    if verbose:
        print(f'Epoch: {epoch}')
        print(f'\tTrain Loss: {train_loss}, Validation Loss: {val_loss}')
        print(f'\tTrain mIoU: {train_miou}, Validation mIoU: {val_miou}')
    
def plot_loss(model_results:list, 
              model_name:str, 
              project_step:str, 
              train_dataset:str, 
              validation_dataset:str)->None:
    """
    Plot and save the training and validation loss curves.

    Args:
        model_results (list): List of model results containing training and validation losses.
        model_name (str): Name of the model.
        project_step (str): Project step or phase.
        train_dataset (str): Name of the training dataset.
        validation_dataset (str): Name of the validation dataset.

    Returns:
        None
    """
    
    epochs = range(len(model_results[0]))
    train_losses = model_results[0]
    validation_losses = model_results[1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  
    ax.set_title(f'Train vs. Validation Loss for {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.plot(epochs, train_losses, 'o-', color='tab:blue', label=f"train loss ({train_dataset})", linewidth=2, markersize=5)
    ax.plot(epochs, validation_losses, '^-', color='tab:red', label=f"validation loss ({validation_dataset})", linewidth=2, markersize=5)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()
    
    # Save the plot
    checkpoint_path = f'{OUTPUT_ROOT}/{project_step}'
    if os.path.exists(checkpoint_path):
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_loss.png", format='png')
    else:
        os.makedirs(checkpoint_path)
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_loss.png", format='png')
    
def plot_miou(model_results:list, 
              model_name:str, 
              project_step:str, 
              train_dataset:str, 
              validation_dataset:str) -> None:
    """
    Plot and save the training and validation mIoU curves.

    Args:
        model_results (list): List of model results containing training and validation mIoU values.
        model_name (str): Name of the model.
        project_step (str): Project step or phase.
        train_dataset (str): Name of the training dataset.
        validation_dataset (str): Name of the validation dataset.

    Returns:
        None
    """
    epochs = range(len(model_results[2]))
    train_mIoU = model_results[2]
    validation_mIoU = model_results[3]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  
    ax.set_title(f'Train vs. Validation mIoU for {model_name} over Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Mean Intersection over Union (mIoU)', fontsize=14)
    ax.plot(epochs, train_mIoU, 'o-', color='tab:blue', label=f"train mIoU ({train_dataset})", linewidth=2, markersize=5)
    ax.plot(epochs, validation_mIoU, '^-', color='tab:red', label=f"validation mIoU ({validation_dataset})", linewidth=2, markersize=5)
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()

    # Save the plot
    checkpoint_path = f'{OUTPUT_ROOT}/{project_step}'
    if os.path.exists(checkpoint_path):
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_miou.png", format='png')
    else:
        os.makedirs(checkpoint_path)
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_miou.png", format='png')

def plot_iou(model_results:list, 
              model_name:str, 
              project_step:str, 
              train_dataset:str, 
              validation_dataset:str) -> None:
    """
    Plot and save the IoU (Intersection over Union) for each class across training and validation phases.

    Args:
        model_results (list): List of model results containing training and validation IoU values for each class.
                              It should contain two lists:
                              - model_results[4]: List of training IoU values for each class.
                              - model_results[5]: List of validation IoU values for each class.
        model_name (str): Name of the model.
        project_step (str): Project step or phase.
        train_dataset (str): Name of the training dataset.
        validation_dataset (str): Name of the validation dataset.

    Returns:
        None
    """
    num_classes = 19
    class_names = [get_id_to_label()[i] for i in range(num_classes)]
    train_iou = [model_results[4][i] for i in range(num_classes)]
    val_iou = [model_results[5][i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  
    bar_width = 0.35
    index = np.arange(num_classes)

    ax.bar(index, train_iou, bar_width, label=f'train IoU ({train_dataset})', color='tab:blue', alpha=0.7)
    ax.bar(index + bar_width, val_iou, bar_width, label=f'validation IoU ({validation_dataset})', color='tab:red', alpha=0.7)

    ax.set_xlabel('Classes', fontsize=14)
    ax.set_ylabel('IoU', fontsize=14)
    ax.set_title(f'Training and Validation IoU for Each Class ({model_name})', fontsize=16, fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linewidth=0.5, axis='y')

    plt.tight_layout()
    plt.show()

    checkpoint_path = f'{OUTPUT_ROOT}/{project_step}'
    if os.path.exists(checkpoint_path):
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_iou.png", format='png')
    else:
        os.makedirs(checkpoint_path)
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_iou.png", format='png')

def plot_segmented_images(model_roots: list,
                          model_types: list[Tuple],
                          n_images: int = 5,
                          device: str = "cpu") -> None:
    """Visualizes the segmentation results of multiple models on multiple random Cityscapes validation images.

    Args:
        model_roots (list): List of paths to the model checkpoints.
        model_types (list): List of model types (e.g., 'DeepLabV2' or 'BiSeNet').
        n_images (int): Number of random images to visualize.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    # Load the Cityscapes validation dataset
    cityscapes_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='val')

    # Select n_images random images from the validation set
    selected_indices = random.sample(range(len(cityscapes_dataset)), n_images)
    selected_images = [cityscapes_dataset[i][0] for i in selected_indices]
    ground_truths = [cityscapes_dataset[i][1] for i in selected_indices]

    # Initialize the models and load their checkpoints
    models = []
    for model_root, model_type in zip(model_roots, model_types):
        checkpoint = torch.load(model_root, map_location=torch.device(device))
        
        if model_type[0] == 'DeepLabV2':
            model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path=DEEPLABV2_PATH).to(device)
        else:
            model = BiSeNet(num_classes=19, context_path="resnet18").to(device)
        
        try:
            model.load_state_dict(checkpoint["model"])
            print(f"{model_type[0]} model loaded successfully.")
        except RuntimeError as e:
            print(f"Error: Failed to load {model_type[0]} model state dictionary with error: {e}")
            print(f"Attempting to adjust the state dictionary for {model_type[0]}...")
            new_state_dict = OrderedDict()
            
            for k, v in checkpoint['model'].items():
                if k.startswith("module"):
                    name = k[7:]  # remove "module." prefix
                else:
                    name = k
                
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print(f"Adjusted state dictionary for {model_type[0]} loaded successfully.")
        model.eval()
        models.append(model)
    
    # Generate the segmented images for each selected image and model
    outputs = []
    with torch.no_grad():
        for image in selected_images:
            model_outputs = []
            for model in models:
                output = model(image.unsqueeze(0))
                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
                output = np.squeeze(output)
                segmented_image = label_to_rgb(output)
                model_outputs.append(segmented_image)
            outputs.append(model_outputs)

    # Convert the original and ground truth images to numpy arrays for plotting
    selected_images_np = [(image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for image in selected_images]
    ground_truths_np = [label_to_rgb(np.squeeze(gt.unsqueeze(0))) for gt in ground_truths]
    
    # Plot the images
    _, axes = plt.subplots(n_images, len(models) + 2, figsize=(23, 2 * n_images))
    
    for row in range(n_images):
        axes[row, 0].imshow(selected_images_np[row])
        axes[0, 0].set_title("Target Image", fontsize=16, fontweight='bold')
        axes[row, 0].axis("off")

        axes[row, 1].imshow(ground_truths_np[row])
        axes[0, 1].set_title("Ground Truth", fontsize=16, fontweight='bold')
        axes[row, 1].axis("off")

        for col, (output, model_type) in enumerate(zip(outputs[row], model_types), start=2):
            axes[row, col].imshow(output)
            axes[0, col].set_title(model_type[1], fontsize=16, fontweight='bold')
            axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.show()