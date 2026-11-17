"""
usage.py

This script provides functionality to monitor system resource utilization, including memory, CPU, and GPU usage.

Dependencies:
-------------
- psutil: For retrieving system and process-related information such as CPU and memory usage.
- GPUtil: For monitoring NVIDIA GPU usage.

Author:
-------
Edwing Ulin

Version:
--------
v1.0.1
"""

import psutil
import GPUtil
import torch

def get_memory_usage_percentage() -> float:
    """
    Calculate the memory usage percentage of the current process relative to total system memory.

    Returns:
    --------
    float: The percentage of total system memory used by the current process.
    """
    process = psutil.Process()
    total_memory = psutil.virtual_memory().total #Total system memory in bytes
    used_memory = process.memory_info().rss # Resident Set Size in bytes

    return (used_memory / total_memory) * 100

def get_cpu_usage_percentage() -> float:
    """
    Calculate the CPU usage percentage for the entire system.

    Returns:
    --------
    float: The percentage of CPU usage averaged over a 1-second interval.
    """
    return psutil.cpu_percent(interval=1) #Calculated CPU usage over 1 second

def get_gpu_usage_percentage() -> tuple[float, float]:
    """
    Get the GPU memory usage and utilization percentage.

    Returns:
    --------
    tuple[float, float]: (GPU memory usage percentage, GPU utilization percentage).
                         Returns (0.0, 0.0) if no GPU is available.
    """

    gpus = GPUtil.getGPUs()

    if not gpus:
        return 0.0, 0.0

    # Assuming the first available GPU (index 0) is being used

    gpu = gpus[0]
    gpu_memory_percentage = gpu.memoryUtil * 100  # Memory usage percentage
    gpu_utilization_percentage = gpu.load * 100  # GPU utilization percentage

    return gpu_memory_percentage, gpu_utilization_percentage

def get_system_usage():
    """
    Retrieve system resource usage statistics including CPU, memory, and GPU.

    Returns:
    --------
    tuple[str, str]: Formatted strings containing CPU, memory, and GPU usage details.
    """

    cpu_usage = get_cpu_usage_percentage()
    memory_usage = get_memory_usage_percentage()
    gpu_memory_usage, gpu_utilization = get_gpu_usage_percentage()

    cpu_mem_stats = f"CPU Usage: {cpu_usage:.2f}% | Memory Usage: {memory_usage:.2f}%"
    gpu_stats = f"GPU Usage: {gpu_memory_usage:.2f}% | GPU Utilization: {gpu_utilization:.2f}%"

    return cpu_mem_stats, gpu_stats

def estimate_memory_usage(batch_size: int, target_size: tuple[int, int, int], dtype=torch.float32) -> float:
    """
    Estimate GPU memory usage for given parameters.
    
    Parameters
    ----------
    batch_size : int
        Batch size
    target_size : tuple
        Target volume size (D, H, W)
    dtype : torch.dtype
        Data type
        
    Returns
    -------
    float
        Estimated memory in GB
    """
    
    element_size = 4 if dtype == torch.float32 else 2  # bytes per element
    volume_elements = batch_size * 1 * target_size[0] * target_size[1] * target_size[2]  # B x C x D x H x W
    volume_memory = volume_elements * element_size
    
    # Account for both clean and noisy volumes
    total_memory = volume_memory * 2
    
    # Convert to GB and add some overhead
    memory_gb = (total_memory / (1024**3)) * 1.2  # 20% overhead
    
    return memory_gb


if __name__ == '__main__':

    cpu_mem_usage, gpu_usage = get_system_usage()
    print(cpu_mem_usage)
    print(gpu_usage)


