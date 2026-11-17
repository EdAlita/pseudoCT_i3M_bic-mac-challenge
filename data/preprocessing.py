"""
preprocessing.py

Funciones reutilizables de normalización y desnormalización para PET, MRI y CT.
Aseguran consistencia entre training, validación e inferencia.

Owner: Edwing Ulin
Version: v1.0.0
"""

import numpy as np
import torch
import yacs.config


def normalize_pet(pet_patch: np.ndarray) -> np.ndarray:
    """
    Normaliza PET usando min-max local del patch.
    
    Args:
        pet_patch: Array de PET (H, W, D).
        
    Returns:
        Array normalizado a rango [0, 1].
    """
    pet_min, pet_max = pet_patch.min(), pet_patch.max()
    if pet_max > 0:
        return (pet_patch - pet_min) / (pet_max - pet_min)
    return pet_patch


def normalize_mri(mri_patch: np.ndarray) -> np.ndarray:
    """
    Normaliza el MRI usando min-max por canal de forma independiente.
    Soporta tensores multicanal (C, D, H, W) como secuencias Dixon (In/Out Phase).
    """
    if mri_patch.ndim == 3:
        mri_min, mri_max = mri_patch.min(), mri_patch.max()
        if (mri_max - mri_min) > 1e-8:
            return (mri_patch - mri_min) / (mri_max - mri_min)
        return mri_patch - mri_min

    mri_min = mri_patch.min(axis=(1, 2, 3), keepdims=True)
    mri_max = mri_patch.max(axis=(1, 2, 3), keepdims=True)
    
    denom = mri_max - mri_min
    
    normalized = np.zeros_like(mri_patch)
    valid_channels = denom > 1e-8
    
    normalized = np.where(valid_channels, (mri_patch - mri_min) / (denom + 1e-8), mri_patch - mri_min)
    
    return normalized


def normalize_ct(ct_patch: np.ndarray, cfg: yacs.config.CfgNode) -> np.ndarray:
    """
    Normaliza CT usando rango configurado (CT_MIN, CT_MAX).
    
    Args:
        ct_patch: Array de CT (H, W, D).
        cfg: Configuración YACS con DATA.CT_MIN y DATA.CT_MAX.
        
    Returns:
        Array normalizado a rango [0, 1].
    """
    return (ct_patch - cfg.DATA.CT_MIN) / (cfg.DATA.CT_MAX - cfg.DATA.CT_MIN)


def denormalize_ct(ct_normalized: np.ndarray, cfg: yacs.config.CfgNode) -> np.ndarray:
    """
    Desnormaliza CT desde rango [0, 1] a valores originales (HU).
    
    Args:
        ct_normalized: Array normalizado en rango [0, 1].
        cfg: Configuración YACS con DATA.CT_MIN y DATA.CT_MAX.
        
    Returns:
        Array en valores originales (HU).
    """
    return ct_normalized * (cfg.DATA.CT_MAX - cfg.DATA.CT_MIN) + cfg.DATA.CT_MIN


def compute_body_bbox(pet: np.ndarray, rel_thr: float = 0.05, min_count: int = 50) -> tuple:
    """
    Bounding box del cuerpo a partir del NAC-PET (disponible en train e inferencia).

    Args:
        pet: Volumen NAC-PET completo (3D).
        rel_thr: Umbral relativo al percentil 99 de los vóxeles positivos.
        min_count: Mínimo de vóxeles de cuerpo por corte para contar ese índice.

    Returns:
        Tupla (lo, hi) de arrays por eje, en índices de vóxel; hi es exclusivo.
    """
    sub = pet[::2, ::2, ::2]
    positive = sub[sub > 0]
    thr = rel_thr * np.percentile(positive, 99) if positive.size else 0.0
    body = pet > thr

    lo, hi = [], []
    for ax in range(3):
        counts = body.sum(axis=tuple(a for a in range(3) if a != ax))
        idx = np.where(counts > min_count)[0]
        if idx.size == 0:
            lo.append(0)
            hi.append(pet.shape[ax])
        else:
            lo.append(int(idx[0]))
            hi.append(int(idx[-1]) + 1)
    return np.array(lo), np.array(hi)


def patch_coords(start, patch_size, shape, bbox=None) -> np.ndarray:
    """
    Coordenadas normalizadas de un parche para el FiLM.

    Args:
        start: Índice inicial del parche por eje.
        patch_size: Tamaño del parche por eje.
        shape: Tamaño del volumen completo por eje.
        bbox: None -> modo "grid" (v12): inicio / tamaño de la malla.
              (lo, hi) -> modo "body": centro del parche relativo al bbox del cuerpo.

    Returns:
        Array float32 de 3 coordenadas.
    """
    start = np.asarray(start, dtype=np.float32)
    if bbox is None:
        return start / np.asarray(shape, dtype=np.float32)
    lo, hi = bbox
    center = start + np.asarray(patch_size, dtype=np.float32) / 2
    coords = (center - lo) / np.maximum(hi - lo, 1)
    return np.clip(coords, -0.5, 1.5).astype(np.float32)


def normalize_patch_batch(
    pet_patch: np.ndarray,
    mri_patch: np.ndarray,
    ct_patch: np.ndarray,
    cfg: yacs.config.CfgNode
) -> tuple:
    """
    Normaliza un batch de parches (PET, MRI, CT) de forma consistente.
    
    Args:
        pet_patch: Array PET (D, H, W).
        mri_patch: Array MRI (D, H, W).
        ct_patch: Array CT (D, H, W).
        cfg: Configuración YACS.
        
    Returns:
        Tupla (pet_norm, mri_norm, ct_norm) normalizados.
    """
    pet_norm = normalize_pet(pet_patch)
    mri_norm = normalize_mri(mri_patch)
    ct_norm = normalize_ct(ct_patch, cfg)
    
    return pet_norm, mri_norm, ct_norm


def normalize_tensor_batch(
    pet_batch: torch.Tensor,
    mri_batch: torch.Tensor,
    device=None
) -> tuple:
    """
    Normaliza batch de tensores en GPU (min-max local).
    Optimizado para velocidad, sin transferencias CPU↔GPU.
    
    Args:
        pet_batch: Tensor (B, C, D, H, W) en GPU.
        mri_batch: Tensor (B, C, D, H, W) en GPU.
        device: Device (auto-detectado de los tensores).
        
    Returns:
        Tupla (pet_norm, mri_norm) normalizados en GPU.
    """
    if device is None:
        device = pet_batch.device
    
    pet_norm = torch.zeros_like(pet_batch)
    mri_norm = torch.zeros_like(mri_batch)
    
    for i in range(pet_batch.shape[0]):
        pet_min = pet_batch[i].min()
        pet_max = pet_batch[i].max()
        if pet_max > pet_min:
            pet_norm[i] = (pet_batch[i] - pet_min) / (pet_max - pet_min + 1e-7)
        else:
            pet_norm[i] = pet_batch[i]
        
        mri_min = mri_batch[i].min()
        mri_max = mri_batch[i].max()
        if mri_max > mri_min:
            mri_norm[i] = (mri_batch[i] - mri_min) / (mri_max - mri_min + 1e-7)
        else:
            mri_norm[i] = mri_batch[i]
    
    return pet_norm, mri_norm


def get_normalized_subject(
    pet_patch: np.ndarray,
    mri_patch: np.ndarray,
    ct_patch: np.ndarray,
    mask_patch: np.ndarray,
    cfg: yacs.config.CfgNode,
    transforms=None
):
    """
    Crea un subject TorchIO normalizado con transformaciones opcionales.
    Reutilizable en dataset.py e inference.py.
    
    Args:
        pet_patch: Array PET (D, H, W).
        mri_patch: Array MRI (D, H, W).
        ct_patch: Array CT (D, H, W).
        mask_patch: Array máscara (D, H, W).
        cfg: Configuración YACS.
        transforms: Transformaciones TorchIO opcionales.
        
    Returns:
        Subject TorchIO normalizado.
    """
    import torchio as tio
    
    pet_norm, mri_norm, ct_norm = normalize_patch_batch(
        pet_patch, mri_patch, ct_patch, cfg
    )
    
    subject = tio.Subject(
        pet=tio.ScalarImage(tensor=torch.from_numpy(pet_norm).unsqueeze(0).float()),
        mri=tio.ScalarImage(tensor=torch.from_numpy(mri_norm).unsqueeze(0).float()),
        target=tio.ScalarImage(tensor=torch.from_numpy(ct_norm).unsqueeze(0).float()),
        batch_mask=tio.LabelMap(tensor=torch.from_numpy(mask_patch).unsqueeze(0).long()),
    )
    
    if transforms:
        subject = transforms(subject)
    
    return subject
