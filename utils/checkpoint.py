"""
checkpoint.py

This script provides utilities to create, save, and load training checkpoints
for deep learning models. Adapted for the Multimodal pseudo-CT generation project.

Dependencies:
- logging: Standard Python logging library.
- glob: Unix style pathname pattern expansion.
- os: Standard Python library for operating system interactions.

Owner:
Edwing Ulin

Version:
v1.1.0
"""

import glob
import os
from typing import Optional, Union, MutableSequence, Tuple, Any

import torch
import yacs.config
from utils.logger import logger_gen # Ajustado al nombre de tu logger

LOGGER = logger_gen(__name__)

def create_checkpoint_dir(expr_dir: Union[str, os.PathLike], expr_num: str) -> str:
    """
    Create the checkpoint directory if it does not exist.
    """
    checkpoint_dir = os.path.join(expr_dir, "checkpoints", str(expr_num))
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(
    log_dir: str,
    resume_experiment: Union[str, int, None] = None
) -> Optional[MutableSequence[str]]:
    """
    Find checkpoint paths from the experiment directory.
    Returns the paths sorted by modification time (latest last).
    """
    if resume_experiment in ["Default", None, ""]:
        return None

    checkpoint_path = os.path.join(log_dir, "checkpoints", str(resume_experiment))
    prior_model_paths = sorted(
        glob.glob(os.path.join(checkpoint_path, "Epoch_*")), key=os.path.getmtime
    )
    return prior_model_paths if prior_model_paths else None


def load_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    fine_tune: bool = False,
    drop_output_layer: bool = False
) -> Tuple[int, float]:
    """
    Load a checkpoint to restore model, optimizer, and scheduler states.

    Returns:
    --------
    tuple: (Next epoch to run, Best MAE metric achieved)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if drop_output_layer:
        _remove_output_layers(checkpoint)

    # Unwrap DataParallel if necessary
    model_state = model.module if hasattr(model, "module") else model
    model_state.load_state_dict(checkpoint["model_state"], strict=not drop_output_layer)

    if not fine_tune:
        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    # Fallback a float('inf') si es un checkpoint antiguo que no tenía best_metric
    best_metric = checkpoint.get("best_metric", float('inf'))
    
    return checkpoint["epoch"], best_metric


def _remove_output_layers(checkpoint: dict):
    """
    Remove final output layers for fine-tuning a pre-trained synthesis network.
    Useful if changing the number of output channels later.
    """
    # Cambia estos keys dependiendo de la capa final de tu UNet/Transformer
    output_keys = ["final_conv.weight", "final_conv.bias"]
    for key in output_keys:
        checkpoint["model_state"].pop(key, None)


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    best_metric: float,
    num_gpus: int,
    cfg: yacs.config.CfgNode,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    best: bool = False
) -> None:
    """
    Save the current state of training to a checkpoint.
    """
    save_name = f"Epoch_{epoch:05d}_training_state.pkl"
    
    # Check for DataParallel safely
    saving_model = model.module if hasattr(model, "module") else model
    
    checkpoint = {
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": cfg.dump(),
    }

    if scheduler:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    torch.save(checkpoint, os.path.join(checkpoint_dir, save_name))

    if best:
        _save_best_checkpoint(checkpoint_dir, checkpoint)


def _save_best_checkpoint(checkpoint_dir: str, checkpoint: dict):
    """Save a copy of the best checkpoint."""
    best_ckpt_path = os.path.join(checkpoint_dir, "Best_training_state.pkl")
    _remove_ckpt(best_ckpt_path)
    torch.save(checkpoint, best_ckpt_path)


def _remove_ckpt(ckpt: str):
    """Safely remove a checkpoint file."""
    try:
        os.remove(ckpt)
    except FileNotFoundError:
        pass