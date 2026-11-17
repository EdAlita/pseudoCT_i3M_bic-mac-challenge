"""
inferencia_docker.py

Script de inferencia para el Bic-Mac Challenge — modo contenedor/evaluación.
Utiliza "Sliding Window Inference" y un Ensemble de folds (promedio secuencial,
sin vmap, para máxima estabilidad).

Rutas FIJAS exigidas por el entorno de evaluación:
  - Lee de:   /data/features/  (montado solo lectura)
  - Escribe: /data/output/ct.nii.gz

Pesos y config van HORNEADOS en la imagen (ver Dockerfile) — el contenedor
corre con --network none, así que nada se descarga en runtime.

Owner: Edwing Ulin
Version: v1.0.0 (docker, stable/no-vmap)
"""

import os
import time
from pathlib import Path

import torch
import torchio as tio
import numpy as np
import nibabel as nib
from tqdm import tqdm

from models.networks import build_model
from config.defaults import get_cfg_defaults
from utils.logger import logger_gen
from data.preprocessing import denormalize_ct, normalize_tensor_batch

logger = logger_gen("inferencia_docker")

FEATURES_DIR = "/data/features"
OUTPUT_PATH = "/data/output/ct.nii.gz"

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config/config.yaml")
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "/app/checkpoints")
VERSION_PATTERN = os.environ.get("MODEL_VERSION", None)
DEVICE_STR = os.environ.get("DEVICE", "cuda")


INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "8"))
_stride_env = os.environ.get("INFERENCE_PATCH_STRIDE")
INFERENCE_PATCH_STRIDE = (
    [int(x) for x in _stride_env.split(",")] if _stride_env else None
)
_max_models_env = os.environ.get("MAX_ENSEMBLE_MODELS")
MAX_ENSEMBLE_MODELS = int(_max_models_env) if _max_models_env else None


def load_ensemble_models(cfg, checkpoints_dir, device, version_pattern=None):
    """Carga los mejores modelos generados por la validación cruzada."""
    models = []

    if version_pattern:
        glob_pattern = f"*{version_pattern}*/Best_training_state.pkl"
    else:
        glob_pattern = "**/Best_training_state.pkl"

    ckpt_paths = list(Path(checkpoints_dir).glob(glob_pattern))

    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No se encontraron checkpoints en {checkpoints_dir}")

    if MAX_ENSEMBLE_MODELS is not None:
        ckpt_paths = ckpt_paths[:MAX_ENSEMBLE_MODELS]

    logger.info(f"Se encontraron {len(ckpt_paths)} modelos para el Ensemble.")

    for ckpt_path in ckpt_paths:
        model = build_model(cfg).to(device)

        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

        state_dict = checkpoint["model_state"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(new_state_dict)
        model.eval()
        models.append(model)

    return models


def inference_sliding_window(cfg, features_dir, models, device, output_path):
    """Realiza la inferencia parche por parche y reconstruye el volumen."""
    logger.info(f"Procesando features en: {features_dir}")

    pet_path = os.path.join(features_dir, "nacpet.nii.gz")
    mri_in_path = os.path.join(features_dir, "mri_combined_in_phase.nii.gz")
    mri_out_path = os.path.join(features_dir, "mri_combined_out_phase.nii.gz")
    topo_path = os.path.join(features_dir, "topogram.nii.gz")

    for p in (pet_path, mri_in_path, mri_out_path, topo_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Falta el archivo requerido: {p}")

    pet_nii = nib.load(pet_path)
    mri_in_nii = nib.load(mri_in_path)
    mri_out_nii = nib.load(mri_out_path)
    topo_nii = nib.load(topo_path)

    pet_data = np.squeeze(pet_nii.get_fdata(dtype=np.float32))
    mri_in_data = np.squeeze(mri_in_nii.get_fdata(dtype=np.float32))
    mri_out_data = np.squeeze(mri_out_nii.get_fdata(dtype=np.float32))
    topo_data = np.squeeze(topo_nii.get_fdata(dtype=np.float32))

    reference_shape = pet_data.shape

    mri_combined = np.stack([mri_in_data, mri_out_data], axis=0)

    subject = tio.Subject(
        pet=tio.ScalarImage(tensor=torch.from_numpy(pet_data).unsqueeze(0).float()),
        mri=tio.ScalarImage(tensor=torch.from_numpy(mri_combined).float())
    )

    topo_tensor = torch.from_numpy(topo_data).unsqueeze(0).unsqueeze(0).float().to(device)

    patch_size = cfg.TRAIN.PATCH_SIZE
    patch_overlap = INFERENCE_PATCH_STRIDE if INFERENCE_PATCH_STRIDE else cfg.TRAIN.PATCH_STRIDE

    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        padding_mode='reflect'
    )

    patch_loader = torch.utils.data.DataLoader(
        grid_sampler,
        batch_size=INFERENCE_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')

    full_shape = subject['pet'].shape[1:]

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', enabled=cfg.TRAIN.AMP):
            for patch_batch in tqdm(patch_loader, desc="Generando parches"):
                locations = patch_batch[tio.LOCATION]
                pet_patches = patch_batch['pet'][tio.DATA].float().to(device)
                mri_patches = patch_batch['mri'][tio.DATA].float().to(device)

                batch_size_current = pet_patches.shape[0]

                pet_patches_norm, mri_patches_norm = normalize_tensor_batch(
                    pet_patches, mri_patches, device=device
                )

                coords_list = []
                for i in range(batch_size_current):
                    z, y, x = locations[i][:3]
                    coords_list.append([z / full_shape[0], y / full_shape[1], x / full_shape[2]])
                coords_tensor = torch.tensor(coords_list, dtype=torch.float32).to(device)

                topo_batch = topo_tensor.expand(batch_size_current, -1, -1, -1)

                # Ensemble secuencial simple: un forward pass por modelo.
                ensemble_preds = torch.zeros_like(pet_patches_norm)
                for model in models:
                    pred = model(pet_patches_norm, mri_patches_norm, topo_batch, coords_tensor)
                    ensemble_preds += pred

                ensemble_preds /= len(models)

                aggregator.add_batch(ensemble_preds, locations)

    output_tensor = aggregator.get_output_tensor()

    output_np = output_tensor.squeeze(0).cpu().numpy()
    output_denorm = denormalize_ct(output_np, cfg)

    if output_denorm.shape != reference_shape:
        raise ValueError(
            f"Shape de salida {output_denorm.shape} no coincide con "
            f"nacpet.nii.gz {reference_shape}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_nifti = nib.Nifti1Image(output_denorm, pet_nii.affine, pet_nii.header)
    nib.save(output_nifti, output_path)

    logger.info(f"pseudo-CT guardado con éxito en: {output_path}")


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device(DEVICE_STR)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(CONFIG_PATH)

    ensemble_models = load_ensemble_models(
        cfg,
        CHECKPOINTS_DIR,
        device,
        version_pattern=VERSION_PATTERN
    )

    inference_sliding_window(
        cfg=cfg,
        features_dir=FEATURES_DIR,
        models=ensemble_models,
        device=device,
        output_path=OUTPUT_PATH
    )

    elapsed = time.time() - start_time
    logger.info(f"Tiempo total de inferencia: {elapsed:.1f}s")