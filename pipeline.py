from typing import Tuple, List, Union
from config import *
from builder import *
from train_val import train_val
from adversarial_train_val import adversarial_train_val
from multi_adversarial_train_val import multi_adversarial_train_val
from utils.metrics import *
from utils.optimization import *
from utils.visualization import *
from utils.checkpoints import *

def pipeline(model_name: str, 
             train_dataset_name: str, 
             val_dataset_name: str,
             n_classes: int,
             epochs: int,
             augmented: bool,
             augmentedType: str,
             multi_level: bool,
             feature: str,
             lr: float,
             loss_fn_name: str,
             ignore_index: int,
             batch_size: int,
             n_workers: int,
             device: str,
             parallelize: bool,
             project_step: str,
             verbose: bool,
             output_root: str,
             checkpoint_root: str,
             power: float,
             evalIterations: int,
             adversarial: bool,
             lambda_ce: float,
             lambda_extra : float,
             extra_loss_name: str) -> None:
    
    # Build data loaders and retrieve input image dimensions
    train_loader, val_loader, data_height, data_width = build_loaders(
        train_dataset_name, val_dataset_name, augmented, augmentedType,
        batch_size, n_workers, adversarial
    )
    
    # Build model(s), optimizer(s), and loss function(s)
    model, optimizer, loss_fn, model_D, optimizer_D, loss_D, extra_loss_fn = build_model(
        model_name, n_classes, device, parallelize, lr, loss_fn_name,
        ignore_index, adversarial, multi_level, extra_loss_name=extra_loss_name, feature=feature
    )

    # Choose training loop based on adversarial and multi-level flags
    if adversarial:
        if multi_level:
            # Multi-level adversarial training
            model_results = multi_adversarial_train_val(
                model=model, model_D=model_D, optimizer=optimizer, optimizer_D=optimizer_D,
                ce_loss=loss_fn, bce_loss=loss_D, dataloaders=train_loader, val_loader=val_loader,
                epochs=epochs, device=device, output_root=output_root, checkpoint_root=checkpoint_root,
                verbose=verbose, n_classes=n_classes, power=power,
                adversarial=adversarial, multi_level=multi_level
            )
        else:
            # Single-level adversarial training with extra loss
            model_results = adversarial_train_val(
                model=model, model_D=model_D, optimizer=optimizer, optimizer_D=optimizer_D,
                ce_loss=loss_fn, bce_loss=loss_D, dataloaders=train_loader, val_loader=val_loader,
                epochs=epochs, device=device, output_root=output_root, checkpoint_root=checkpoint_root,
                verbose=verbose, lambda_ce=lambda_ce, lambda_extra=lambda_extra,
                extra_loss_fn=extra_loss_fn, n_classes=n_classes, power=power,
                adversarial=adversarial
            )
    else:
        # Standard training (no adversarial setup)
        model_results = train_val(
            model=model, optimizer=optimizer, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            epochs=epochs, device=device, output_root=output_root,
            checkpoint_root=checkpoint_root, verbose=verbose,
            n_classes=n_classes, power=power
        )

    # Compute FLOPs and parameter count
    model_params_flops = compute_flops(
        model=model, height=data_height, width=data_width
    )
    
    # Measure model latency and FPS
    model_latency_fps = compute_latency_and_fps(
        model=model, height=data_height, width=data_width,
        iterations=evalIterations, device=device
    )
    
    # Generate and save loss, mIoU, and per-class IoU plots
    plot_loss(model_results, model_name, project_step, train_dataset_name, val_dataset_name)
    plot_miou(model_results, model_name, project_step, train_dataset_name, val_dataset_name)
    plot_iou(model_results, model_name, project_step, train_dataset_name, val_dataset_name)
    
    # Save all final metrics and performance results
    save_results(
        model_results,
        filename=f"{model_name}_metrics_{project_step}",
        project_step=project_step,
        model_params_flops=model_params_flops,
        model_latency_fps=model_latency_fps
    )
