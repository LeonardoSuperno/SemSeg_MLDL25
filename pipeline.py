from typing import Tuple, List, Union
from models import BiSeNet, get_deeplab_v2
from config import *
from builder import *
from train import train
from evaluation import evaluate_model


def pipeline (model_name: str, 
              train_dataset_name: str, 
              val_dataset_name: str,
              n_classes:int,
              epochs: int,
              augmented: bool,
              augmentedType:str,
              optimizer_name: str,
              lr:float,
              momentum:float,
              weight_decay:float,
              loss_fn_name: str,
              ignore_index:int,
              batch_size: int,
              n_workers: int,
              device:str,
              parallelize:bool,
              project_step:str,
              verbose: bool,
              checkpoint_root:str,
              power:float,
              evalIterations:int,
              adversarial:bool
              )->None:
    """
    Main pipeline function to orchestrate the training and evaluation of a deep learning model.

    Args:
        model_name (str): Name of the deep learning model architecture.
        train_dataset_name (str): Name of the training dataset.
        val_dataset_name (str): Name of the validation dataset.
        n_classes (int): Number of classes in the dataset.
        epochs (int): Number of epochs for training.
        augmented (bool): Whether to use data augmentation during training.
        augmentedType (str): Type of data augmentation to apply.
        optimizer_name (str): Name of the optimizer to use.
        lr (float): Learning rate for the optimizer.
        momentum (float): Momentum factor for optimizers like SGD.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        loss_fn_name (str): Name of the loss function.
        ignore_index (int): Index to ignore in the loss function (e.g., for padding).
        batch_size (int): Batch size for training and validation data loaders.
        n_workers (int): Number of workers for data loading.
        device (str): Device to run the model on ('cuda' or 'cpu').
        parallelize (bool): Whether to use GPU parallelization.
        project_step (str): Name or identifier of the current project step or experiment.
        verbose (bool): Whether to print detailed logs during training.
        checkpoint_root (str): Root directory to save checkpoints and results.
        power (float): Power parameter for polynomial learning rate scheduler.
        evalIterations (int): Number of iterations for evaluating model latency and FPS.
        adversarial (bool): Whether to use adversarial training.

    Returns:
        None
    """
    model, optimizer, loss_fn, model_D, optimizer_D, loss_D = build_model(model_name, 
                                                                       n_classes,
                                                                       device,
                                                                       parallelize,
                                                                       optimizer_name, 
                                                                       lr,
                                                                       momentum,
                                                                       weight_decay,
                                                                       loss_fn_name,
                                                                       ignore_index,
                                                                       adversarial)

    # get loader
    train_loader, val_loader, data_height, data_width = build_loaders(train_dataset_name, 
                                                                    val_dataset_name, 
                                                                    augmented,
                                                                    augmentedType,
                                                                    batch_size,
                                                                    n_workers,
                                                                    adversarial)
    
    model_results = train(model=model,
                          model_D = model_D,
                          optimizer=optimizer, 
                          optimizer_D = optimizer_D,
                          loss_fn = loss_fn, 
                          loss_D = loss_D,
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          epochs=epochs, 
                          device=device, 
                          checkpoint_root=checkpoint_root,
                          project_step=project_step,
                          verbose=verbose,
                          n_classes=n_classes,
                          power=power,
                          adversarial=adversarial)

    # evaluation -> in utils in computations
    model_params_flops = compute_flops(model=model, 
                                       height=data_height, 
                                       width=data_width)
    
    model_latency_fps = compute_latency_and_fps(model=model,
                                                height=data_height, 
                                                width=data_width, 
                                                iterations=evalIterations, 
                                                device=device)
    
    # visualization -> in utils in visualization
    plot_loss(model_results, 
              model_name, 
              project_step, 
              train_dataset_name, 
              val_dataset_name)
    
    plot_miou(model_results, 
              model_name, 
              project_step, 
              train_dataset_name, 
              val_dataset_name)
    
    plot_iou(model_results, 
             model_name, 
             project_step, 
             train_dataset_name, 
             val_dataset_name)
    
    # save results -> in utlis in checkpoint
    save_results(model_results, 
                 filename=f"{model_name}_metrics_{project_step}", 
                 project_step=project_step,
                 model_params_flops=model_params_flops,
                 model_latency_fps=model_latency_fps)
