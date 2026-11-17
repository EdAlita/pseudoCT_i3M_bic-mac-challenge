"""
optimizer.py

This module provides functionality to configure and retrieve the AdamW or SGD 
optimizer for training 3D multimodal deep learning models based on the 
user-defined configuration.

Dependencies:
    - yacs.config: Manages hierarchical configuration settings.
    - torch: Supplies model parameter access and optimizer implementations.
    - networks: Contains the model definitions such as MultimodalPCT.
    - typing: Enables type hinting for function arguments and return types.

Owner:
    Edwing Ulin

Version:
    v1.0.3
"""

from typing import Union
import torch
import yacs.config
from models.networks import MultimodalPCT
from utils.logger import logger_gen

logger = logger_gen(__name__)


def get_param_groups(model, cfg):
    """
    Grupos de parámetros para fine-tuning (B3): congela los de FREEZE_PREFIXES y aplica
    BASE_LR * LR_MULT_VALUES[i] a los de LR_MULT_PREFIXES[i]. Sin prefijos devuelve un
    único grupo con todos los parámetros (comportamiento original).
    """
    freeze = tuple(cfg.OPTIMIZER.FREEZE_PREFIXES)
    mults = list(zip(cfg.OPTIMIZER.LR_MULT_PREFIXES, cfg.OPTIMIZER.LR_MULT_VALUES))
    if len(cfg.OPTIMIZER.LR_MULT_PREFIXES) != len(cfg.OPTIMIZER.LR_MULT_VALUES):
        raise ValueError("LR_MULT_PREFIXES y LR_MULT_VALUES deben tener la misma longitud")
    if not freeze and not mults:
        return [{"params": [p for p in model.parameters()], "lr": cfg.OPTIMIZER.BASE_LR}]

    groups = {1.0: []}      # el grupo base primero: train.py registra param_groups[0]['lr']
    for _, m in mults:
        groups.setdefault(m, [])
    n_frozen = 0
    for name, p in model.named_parameters():
        name = name.replace("module.", "", 1)
        if freeze and name.startswith(freeze):
            p.requires_grad_(False)
            n_frozen += p.numel()
            continue
        mult = next((m for prefix, m in mults if name.startswith(prefix)), 1.0)
        groups[mult].append(p)

    summary = ", ".join(f"x{m}: {sum(p.numel() for p in ps):,}" for m, ps in groups.items())
    logger.info(f"Grupos de LR (BASE_LR {cfg.OPTIMIZER.BASE_LR}): {summary}; congelados: {n_frozen:,}")
    return [{"params": ps, "lr": cfg.OPTIMIZER.BASE_LR * m} for m, ps in groups.items() if ps]

def get_optimizer(
        model: Union[MultimodalPCT, torch.nn.DataParallel],
        cfg: yacs.config.CfgNode
) -> torch.optim.Optimizer:
    """
    Get the optimizer based on the configuration provided.

    Parameters
    ----------
    model : Union[MultimodalPCT, torch.nn.DataParallel]
        The model whose parameters will be optimized.
    cfg : yacs.config.CfgNode
        Configuration Node containing optimizer settings.

    Returns
    -------
    torch.optim.Optimizer
        The initialized optimizer (AdamW o SGD).

    Raises
    ------
    NotImplementedError
        If the specified optimizer method in the configuration is not
        'adamW' o 'sgd'.
    """
    
    method = cfg.OPTIMIZER.OPTIMIZING_METHOD.lower()

    if method == "adamw":
        return torch.optim.AdamW(
            get_param_groups(model, cfg),
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=(0.9, getattr(cfg.OPTIMIZER, "BETA2", 0.999)),
            eps=1e-08,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif method == "sgd":
        return torch.optim.SGD(
            get_param_groups(model, cfg),
            lr=cfg.OPTIMIZER.BASE_LR,
            momentum=getattr(cfg.OPTIMIZER, "MOMENTUM", 0.99),
            nesterov=getattr(cfg.OPTIMIZER, "NESTEROV", True),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{cfg.OPTIMIZER.OPTIMIZING_METHOD}' is not supported. "
            "Solo 'adamW' y 'sgd' están implementados para este proyecto."
        )