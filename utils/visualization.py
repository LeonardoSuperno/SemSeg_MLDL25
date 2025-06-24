import matplotlib.pyplot as plt
import numpy as np
import os
from utils.data_processing import get_id_to_label
from config import OUTPUT_ROOT

# Print training and validation statistics per epoch
def print_stats(epoch:int, train_loss:float, val_loss:float, train_miou:float, val_miou:float, verbose:bool)->None:
    if verbose:
        print(f'Epoch: {epoch}')
        print(f'\tTrain Loss: {train_loss}, Validation Loss: {val_loss}')
        print(f'\tTrain mIoU: {train_miou}, Validation mIoU: {val_miou}')

# Plot and save training and validation loss
def plot_loss(model_results:list, model_name:str, project_step:str, train_dataset:str, validation_dataset:str)->None:
    epochs = range(len(model_results[0]))
    train_losses = model_results[0]
    validation_losses = model_results[1]

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
    os.makedirs(checkpoint_path, exist_ok=True)
    fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_loss.png", format='png')

# Plot and save training and validation mean IoU
def plot_miou(model_results:list, model_name:str, project_step:str, train_dataset:str, validation_dataset:str) -> None:
    epochs = range(len(model_results[2]))
    train_mIoU = model_results[2]
    validation_mIoU = model_results[3]

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
    os.makedirs(checkpoint_path, exist_ok=True)
    fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_miou.png", format='png')

# Plot and save per-class IoU bars for training and validation
def plot_iou(model_results:list, model_name:str, project_step:str, train_dataset:str, validation_dataset:str) -> None:
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

    # Save the plot
    checkpoint_path = f'{OUTPUT_ROOT}/{project_step}'
    os.makedirs(checkpoint_path, exist_ok=True)
    fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_iou.png", format='png')