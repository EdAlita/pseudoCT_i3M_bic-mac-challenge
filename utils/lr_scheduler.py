"""
lr_scheduler.py

This script provides the learning rate scheduler to use during model training.
It supports several scheduler types based on configuration, including step LR,
cosine annealing with warm restarts, and multi-step LR.

Dependencies:
- torch.optim: Optimizer and scheduler utilities.
- yacs.config: Configuration management for training settings.

Owner:
Edwing Ulin

Version:
v1.0.0
"""

from typing import Union
import torch.optim
import torch.optim.lr_scheduler as scheduler
import yacs.config
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: yacs.config.CfgNode
) -> Union[None, StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau]:
    """
    Returns the learning rate scheduler based on configuration settings.

    Args:
    -----
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        cfg (yacs.config.CfgNode): Configuration settings.

    Returns:
    --------
        Union[None, StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau]:
            The configured learning rate scheduler or None if no scheduler is required.

    Raises:
    -------
        ValueError: If the specified scheduler type in the config is unsupported.
    """
    scheduler_type = cfg.OPTIMIZER.LR_SCHEDULER

    if scheduler_type == "step_lr":
        return scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg.OPTIMIZER.STEP_SIZE,
            gamma=cfg.OPTIMIZER.GAMMA
        )
    elif scheduler_type == "cosineWarmRestarts":
        epochs_per_cycle = cfg.OPTIMIZER.T_ZERO
        steps_per_cycle = epochs_per_cycle * cfg.TRAIN.BATCHES_PER_EPOCH
        
        return scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=steps_per_cycle,
            T_mult=cfg.OPTIMIZER.T_MULT,
            eta_min=cfg.OPTIMIZER.ETA_MIN
        )
    elif scheduler_type == "multiStep":
        return scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.OPTIMIZER.MILESTONE,
            gamma=cfg.OPTIMIZER.GAMMA
        )
    elif scheduler_type == "reduceLROnPlateau":
        return scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=cfg.OPTIMIZER.GAMMA,
            patience=cfg.OPTIMIZER.PATIENCE,
            min_lr=cfg.OPTIMIZER.ETA_MIN,
            verbose=True
        )
    elif scheduler_type == "polynomialDecay":
        total_steps = cfg.TRAIN.NUM_EPOCHS * cfg.TRAIN.BATCHES_PER_EPOCH
        power = cfg.OPTIMIZER.POLY_POWER
        eta_min_ratio = cfg.OPTIMIZER.ETA_MIN / cfg.OPTIMIZER.BASE_LR

        def poly_lambda(current_step: int) -> float:
            progress = min(current_step / total_steps, 1.0)
            decay = (1.0 - progress) ** power
            return eta_min_ratio + (1.0 - eta_min_ratio) * decay

        return scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=poly_lambda
        )

    elif scheduler_type in ["NoScheduler", None]:
        return None
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")