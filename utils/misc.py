"""
misc.py

This module contains various utility functions used throughout the project, including:
- Experiment management (finding the latest experiment)
- Medical imaging file handling (NIfTI reading)
- Tensor statistics computation
- Image visualization and logging with TensorBoard
- File processing utilities

Dependencies:
-------------
- `nibabel`: Handles medical imaging files (NIfTI format).
- `yacs`: Manages configuration settings.
- `torch`: Deep learning framework.
- `numpy`: Numerical computations.
- `re`: Regular expressions for file name pattern matching.
- `matplotlib.pyplot`: Visualization of medical images.
- `pathlib.Path`: Simplifies file path manipulations.

Author:
-------
Edwing Ulin

Version:
--------
v1.0.1
"""

import os
import io
from PIL import Image

import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # Added missing import
import torch
import torch.nn.functional as F

import data.loader
import yacs.config


def update_num_steps(
    dataloader: data.loader.DataLoader, cfg: yacs.config.CfgNode
):
    """
    Update the number of steps.

    Parameters
    ----------
    dataloader : FastSurferCNN.data_loader.loader.DataLoader
        [MISSING].
    cfg : yacs.config.CfgNode
        [MISSING].
    """
    cfg.TRAIN.NUM_STEPS = len(dataloader)

def find_latest_experiment(path: str) -> int:
    """
    Find the latest experiment ID by scanning a directory.

    Args:
    -----
        path (str): Path to the experiments directory.

    Returns:
    --------
        int: The highest-numbered experiment ID found. Returns 0 if no valid experiment is found.
    """
    list_of_experiments = os.listdir(path)
    experiment_ids = []

    for exp in list_of_experiments:
        try:
            experiment_ids.append(int(exp))
        except ValueError:
            continue

    return max(experiment_ids, default=0)


def plot_nifti_batch_comparison(
    gt_tensor: torch.Tensor, 
    denoised_tensor: torch.Tensor, 
    writer: any, 
    tag: str, 
    step: int,
    file_save_name: str = None
) -> None:
    """
    Visualiza un batch completo. 
    Espera tensores con dimensión de Batch: [B, C, D, H, W] o [B, D, H, W]
    """
    gt_np = gt_tensor.cpu().numpy()
    pred_np = denoised_tensor.cpu().numpy()
    
    if gt_np.ndim == 5:
        gt_np = gt_np[:, 0, ...]
        pred_np = pred_np[:, 0, ...]

    B = gt_np.shape[0]

    HU_MIN = -1000.0
    HU_MAX = 1500.0
    
    gt_hu = (gt_np * (HU_MAX - HU_MIN)) + HU_MIN
    pred_hu = (pred_np * (HU_MAX - HU_MIN)) + HU_MIN
    diff_hu = np.abs(gt_hu - pred_hu)

    cols = [
        "Axial GT", "Axial pCT", "Axial Diff",
        "Coronal GT", "Coronal pCT", "Coronal Diff",
        "Sagittal GT", "Sagittal pCT", "Sagittal Diff"
    ]

    fig, axes = plt.subplots(nrows=B, ncols=9, figsize=(22, 2.5 * B))
    
    fig.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.08, top=0.88)

    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    im_ct = None
    im_diff = None

    for b in range(B):
        gt = gt_hu[b]
        pred = pred_hu[b]
        diff = diff_hu[b]

        d, h, w = gt.shape

        slices = [
            (gt[d//2, :, :], pred[d//2, :, :], diff[d//2, :, :]), # Axial
            (gt[:, h//2, :], pred[:, h//2, :], diff[:, h//2, :]), # Coronal
            (gt[:, :, w//2], pred[:, :, w//2], diff[:, :, w//2])  # Sagital
        ]

        plot_data = []
        for s_gt, s_pred, s_diff in slices:
            plot_data.extend([
                (s_gt, 'gray', HU_MIN, HU_MAX), 
                (s_pred, 'gray', HU_MIN, HU_MAX), 
                (s_diff, 'hot', 0, 800)
            ])

        for c in range(9):
            ax = axes[b, c]
            img, cmap, vmin, vmax = plot_data[c]
            
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax.axis('off')

            if b == 0:
                ax.set_title(cols[c], fontsize=13, fontweight='bold')
                
            if c == 0:
                ax.text(-0.1, 0.5, f"Sample {b}", transform=ax.transAxes, 
                        fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)

            if b == B - 1:
                if c == 0: im_ct = im 
                if c == 2: im_diff = im

    cbar_ax1 = fig.add_axes([0.20, 0.03, 0.25, 0.015]) 
    cbar_ax2 = fig.add_axes([0.55, 0.03, 0.25, 0.015])

    cbar1 = fig.colorbar(im_ct, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('CT Intensity (HU)', fontsize=12, fontweight='bold')

    cbar2 = fig.colorbar(im_diff, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Absolute Error (HU)', fontsize=12, fontweight='bold')

    plt.suptitle(f"Batch Synthesis Comparison - Epoch/Step {step}", fontsize=20, fontweight='bold', y=0.96)

    if file_save_name:
        fig.savefig(file_save_name, facecolor='white')
    
    if writer is not None:
        writer.add_figure(tag, fig, global_step=step)


    
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_batch_inputs(batch: dict, step: int, tag: str, writer=None, max_samples: int = 10, save_path: str = None):
    """
    Visualiza las entradas brutas (PET, MRI, TOPO) y el Target (CT) de un batch.
    Muestra el corte central axial de cada volumen.
    """
    pet = batch["pet"].cpu().numpy()
    mri = batch["mri"].cpu().numpy()
    target = batch["target"].cpu().numpy()
    mask = batch["batch_mask"].cpu().numpy()
    
    if pet.ndim == 5: pet = pet[:, 0, ...]
    if mri.ndim == 5: mri = mri[:, 0, ...]
    if target.ndim == 5: target = target[:, 0, ...]
    
    if mask.ndim == 5: mask = mask[:, 0, ...]
    elif topo.ndim == 4: topo = topo[:, 0, ...]

    B = min(pet.shape[0], max_samples)
    
    fig, axes = plt.subplots(nrows=B, ncols=4, figsize=(16, 3 * B))
    fig.subplots_adjust(wspace=0.1, hspace=0.3, top=0.95)
    
    if B == 1: axes = np.expand_dims(axes, axis=0)
    
    for b in range(B):
        d_pet = pet[b].shape[0]
        d_mri = mri[b].shape[0]
        d_tgt = target[b].shape[0]
        d_mask = mask[b].shape[0]
        
        s_pet = pet[b, d_pet//2, :, :]
        s_mri = mri[b, d_mri//2, :, :]
        s_tgt = target[b, d_tgt//2, :, :]
        s_mask = mask[b, d_mask//2, :, :]


        im0 = axes[b, 0].imshow(s_mri, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[b, 0].axis('off')
        plt.colorbar(im0, ax=axes[b, 0], fraction=0.046, pad=0.04)

        im1 = axes[b, 1].imshow(s_pet, cmap='hot', vmin=0, vmax=1, origin='lower')
        axes[b, 1].axis('off')
        plt.colorbar(im1, ax=axes[b, 1], fraction=0.046, pad=0.04)

        im2 = axes[b, 2].imshow(s_mask, cmap='tab20', origin='lower', vmin=0, vmax=5)
        axes[b, 2].axis('off')
        plt.colorbar(im2, ax=axes[b, 2], fraction=0.046, pad=0.04)

        im3 = axes[b, 3].imshow(s_tgt, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[b, 3].axis('off')
        plt.colorbar(im3, ax=axes[b, 3], fraction=0.046, pad=0.04)

        if b == 0:
            axes[b, 0].set_title("MRI (Input)", fontweight='bold', fontsize=12)
            axes[b, 1].set_title("PET (Input)", fontweight='bold', fontsize=12)
            axes[b, 2].set_title("Mask (Input)", fontweight='bold', fontsize=12)
            axes[b, 3].set_title("Target CT (Input)", fontweight='bold', fontsize=12)

        axes[b, 0].text(-0.1, 0.5, f"Sample {b}", transform=axes[b, 0].transAxes, 
                        fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
        
    plt.suptitle(f"Batch Inputs Visualization - Step {step}", fontsize=18, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, facecolor='white')
    
    if writer is not None:
        writer.add_figure(tag, fig, global_step=step)
        
    plt.close(fig)    