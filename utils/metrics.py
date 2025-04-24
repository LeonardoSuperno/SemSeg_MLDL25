from typing import Dict, Tuple
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time

def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def compute_flops(model: torch.nn.Module, 
                  height: int = 512, 
                  width: int = 1024) -> Dict[str, str]:
    """
    Computes the number of floating point operations (FLOPs) and parameters of a model.

    Args:
        model (torch.nn.Module): The neural network model to analyze.
        height (int, optional): The height of the input image. Defaults to 512.
        width (int, optional): The width of the input image. Defaults to 1024.

    Returns:
        Dict[str, str]: A dictionary containing the number of parameters and FLOPs of the model.
    """
    
    # Create a dummy input tensor with the specified dimensions
    image = torch.zeros((1, 3, height, width))

    # Perform FLOP analysis on the model with the dummy input
    flops = FlopCountAnalysis(model.cpu(), image)
    
    # Generate a formatted table with the FLOP count results
    table = flop_count_table(flops)
    
    # Extract the number of parameters and FLOPs from the table
    n_param_table = table.split('\n')[2].split('|')[2].strip()
    flops_table = table.split('\n')[2].split('|')[3].strip()

    # Return the extracted values as a dictionary
    return {'Parameters': n_param_table,
            'FLOPS': flops_table
            }


def compute_latency_and_fps(model: torch.nn.Module, 
                            height: int = 512, 
                            width: int = 1024, 
                            iterations: int = 1000, 
                            device: str = 'cuda') ->  Dict[str, float]:
    """
    Computes the mean latency, standard deviation of latency, mean FPS, and standard deviation of FPS for a given model.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        height (int, optional): The height of the input image. Defaults to 512.
        width (int, optional): The width of the input image. Defaults to 1024.
        iterations (int, optional): Number of iterations to measure latency and FPS. Defaults to 1000.
        device (str, optional): Device to run inference ('cpu' or 'cuda'). Defaults to 'cuda'.

    Returns:
        Dict[str, float]: Dictionary containing model latency and FPS information.
    """
    
    latencies = []
    fps_records = []
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for _ in range(iterations):
            # Create a dummy input tensor with the specified dimensions and move it to the device
            image = torch.zeros((1, 3, height, width)).to(device)
            
            # Measure the start time of the inference
            start_time = time.time()
            
            # Perform inference with the model
            model(image)
            
            # Measure the end time of the inference
            end_time = time.time() 
            
            # Calculate the latency in seconds and append it to the list
            latency = end_time - start_time
            latencies.append(latency)
            
            # Calculate the FPS and append it to the list
            fps_records.append(1 / latency)
    
    # Calculate mean and standard deviation of latency
    mean_latency = np.mean(latencies)*1000
    std_latency = np.std(latencies)*1000
    
    # Calculate mean and standard deviation of FPS
    mean_fps = np.mean(fps_records)
    std_fps = np.std(fps_records)

    return {'mean_latency': mean_latency,
            'std_latency': std_latency,
            'mean_fps': mean_fps,
            'std_fps':std_fps}
