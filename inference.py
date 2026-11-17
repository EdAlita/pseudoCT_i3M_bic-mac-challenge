"""
inference.py

Script de inferencia para el Bic-Mac Challenge.
Utiliza "Sliding Window Inference" para reconstruir volúmenes masivos (ej. 512x512x531)
y hace un "Ensemble" promediando las predicciones de los 5 folds.

Procesa TODAS las carpetas de pacientes dentro de un directorio raíz. Por cada
paciente crea, dentro de la carpeta de salida (carpeta de la inferencia), una
subcarpeta con el nombre del caso y guarda el resultado como "ct.nii".

Versión ESTABLE: ensemble secuencial simple (sin vmap), sin prefetch en
background threads. Más lenta que la versión "optimizada", pero sin riesgo
de deadlocks fork+threads ni de fallos de vmap con checkpointing.

Owner: Edwing Ulin
Version: v1.2.0 (stable)
"""

import os
import argparse
from pathlib import Path
import torch
import torchio as tio
import numpy as np
import yacs.config
import nibabel as nib
from tqdm import tqdm

from models.networks import build_model
from config.defaults import get_cfg_defaults
from utils.logger import logger_gen
from data.preprocessing import denormalize_ct, normalize_tensor_batch, compute_body_bbox, patch_coords

logger = logger_gen("inference_ensemble")


def load_ensemble_models(cfg, checkpoints_dir, device, version_pattern=None):
    """Carga los mejores modelos generados por la validación cruzada filtrados por versión."""
    models = []

    if version_pattern:
        glob_pattern = f"*{version_pattern}*/Best_training_state.pkl"
    else:
        glob_pattern = "**/Best_training_state.pkl"

    ckpt_paths = list(Path(checkpoints_dir).glob(glob_pattern))

    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No se encontraron checkpoints en {checkpoints_dir}")

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


def is_valid_patient_dir(patient_dir):
    """
    Verifica que la carpeta del paciente tenga la modalidad mínima requerida
    (NAC-PET). El MRI y el topograma son OPCIONALES: si faltan (ej. cohortes
    pediátricas de UMC Utrecht, sin MRI ni scout), se usan ceros, lo que requiere
    un modelo entrenado con MRI dropout y TOPO dropout.
    """
    required_files = [
        os.path.join(patient_dir, "features", "nacpet.nii.gz"),
    ]
    return all(os.path.isfile(f) for f in required_files)


def has_mri(patient_dir):
    """Verifica si las dos fases del MRI Dixon están disponibles para el paciente."""
    mri_files = [
        os.path.join(patient_dir, "features", "mri_combined_in_phase.nii.gz"),
        os.path.join(patient_dir, "features", "mri_combined_out_phase.nii.gz"),
    ]
    return all(os.path.isfile(f) for f in mri_files)


def inference_sliding_window(cfg, patient_dir, models, device, output_path, clamp_outside_body=False):
    """Realiza la inferencia parche por parche y reconstruye el volumen."""
    logger.info(f"Procesando paciente: {patient_dir}")

    pet_path = os.path.join(patient_dir, "features", "nacpet.nii.gz")
    mri_in_path = os.path.join(patient_dir, "features", "mri_combined_in_phase.nii.gz")
    mri_out_path = os.path.join(patient_dir, "features", "mri_combined_out_phase.nii.gz")
    topo_path = os.path.join(patient_dir, "features", "topogram.nii.gz")

    pet_nii = nib.load(pet_path)
    pet_data = np.squeeze(pet_nii.get_fdata())

    if os.path.isfile(topo_path):
        topo_data = np.squeeze(nib.load(topo_path).get_fdata())
    else:
        logger.warning(
            f"Topograma no disponible para {patient_dir}. Usando topograma vacío "
            f"— requiere un checkpoint entrenado con TOPO dropout."
        )
        topo_data = np.zeros(cfg.DATA.TOPO_SHAPE, dtype=np.float32)

    affine = pet_nii.affine

    if has_mri(patient_dir):
        mri_in_data = np.squeeze(nib.load(mri_in_path).get_fdata())
        mri_out_data = np.squeeze(nib.load(mri_out_path).get_fdata())
        mri_combined = np.stack([mri_in_data, mri_out_data], axis=0)
    else:
        logger.warning(
            f"MRI no disponible para {patient_dir}. Usando modo PET-only "
            f"(MRI en cero) — requiere un checkpoint entrenado con MRI dropout."
        )
        mri_combined = np.zeros((2, *pet_data.shape), dtype=np.float32)

    subject = tio.Subject(
        pet=tio.ScalarImage(tensor=torch.from_numpy(pet_data).unsqueeze(0).float()),
        mri=tio.ScalarImage(tensor=torch.from_numpy(mri_combined).float())
    )

    topo_tensor = torch.from_numpy(topo_data).unsqueeze(0).unsqueeze(0).float().to(device)

    patch_size = cfg.TRAIN.PATCH_SIZE
    patch_overlap = cfg.TRAIN.PATCH_STRIDE

    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        padding_mode='reflect'
    )

    # num_workers=0: sin multiprocessing, sin riesgo de deadlock fork+threads.
    patch_loader = torch.utils.data.DataLoader(
        grid_sampler,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')

    full_shape = subject['pet'].shape[1:]
    # Mismas coordenadas que en dataset.py: bbox del cuerpo desde el NAC-PET en modo "body".
    bbox = compute_body_bbox(pet_data) if cfg.DATA.COORDS_MODE == "body" else None

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

                coords_tensor = torch.from_numpy(np.stack([
                    patch_coords(locations[i][:3].numpy(), patch_size, full_shape, bbox)
                    for i in range(batch_size_current)
                ])).float().to(device)

                topo_batch = topo_tensor.expand(batch_size_current, -1, -1, -1)

                # Ensemble secuencial simple: un forward pass por modelo, sin vmap.
                ensemble_preds = torch.zeros_like(pet_patches_norm)
                for model in models:
                    pred = model(pet_patches_norm, mri_patches_norm, topo_batch, coords_tensor)
                    ensemble_preds += pred

                ensemble_preds /= len(models)

                aggregator.add_batch(ensemble_preds, locations)

    output_tensor = aggregator.get_output_tensor()

    output_np = output_tensor.squeeze(0).cpu().numpy()
    output_denorm = denormalize_ct(output_np, cfg)

    if clamp_outside_body:
        # Post-proceso (B2b): aire fuera de la máscara de cuerpo del NAC-PET, dilatada
        # 4 vóxeles. En Utrecht quita la mayoría de "fantasmas" fuera del cuerpo sin
        # cambiar el MAE del cuerpo (+0.9 HU). Import aquí: necesita scipy.
        from scipy import ndimage as ndi
        from utils.body_coords import body_mask_pet, upsample_mask
        keep = upsample_mask(ndi.binary_dilation(body_mask_pet(pet_data), iterations=2), output_denorm.shape)
        output_denorm = np.where(keep, output_denorm, -1000.0).astype(output_denorm.dtype)

    output_denorm_tensor = torch.from_numpy(output_denorm).unsqueeze(0)

    predicted_image = tio.ScalarImage(tensor=output_denorm_tensor, affine=affine)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predicted_image.save(output_path)

    logger.info(f"pseudo-CT guardado con éxito en: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inferencia Ensemble - Bic-Mac Challenge")

    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Ruta al .yaml"
    )
    parser.add_argument(
        "--checkpoints_dir", type=str, required=True,
        help="Carpeta 'experiments/checkpoints/'"
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Versión específica a usar (ej. 'v6_torso')"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Carpeta raíz que contiene todas las carpetas de pacientes a inferir (ej. dev/)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Carpeta de la inferencia donde se guardarán los resultados "
             "(ej. outputs/inference_v6). Dentro se crea una subcarpeta por "
             "cada caso con el archivo ct.nii"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Dispositivo a usar (cuda o cpu)")
    parser.add_argument(
        "--clamp_outside_body", action="store_true",
        help="Poner a -1000 HU lo que queda fuera del cuerpo según el NAC-PET (recomendado en Utrecht)")

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)

    device = torch.device(args.device)

    ensemble_models = load_ensemble_models(
        cfg,
        args.checkpoints_dir,
        device,
        version_pattern=args.version
    )

    input_root = Path(args.input_dir)
    patient_dirs = sorted(
        [d for d in input_root.iterdir() if d.is_dir() and is_valid_patient_dir(d)]
    )

    if len(patient_dirs) == 0:
        raise FileNotFoundError(
            f"No se encontraron carpetas de pacientes válidas en {args.input_dir}"
        )

    logger.info(f"Se encontraron {len(patient_dirs)} pacientes para inferir.")

    failed_cases = []
    for patient_dir in tqdm(patient_dirs, desc="Procesando pacientes"):
        case_name = patient_dir.name
        output_path = os.path.join(args.output_dir, case_name, "ct.nii")

        try:
            inference_sliding_window(
                cfg=cfg,
                patient_dir=str(patient_dir),
                models=ensemble_models,
                device=device,
                output_path=output_path,
                clamp_outside_body=args.clamp_outside_body,
            )
        except Exception as e:
            logger.error(f"Fallo al procesar el caso '{case_name}': {e}")
            failed_cases.append(case_name)
            continue

    if failed_cases:
        logger.info(
            f"Inferencia finalizada con {len(failed_cases)} caso(s) fallido(s): "
            f"{', '.join(failed_cases)}"
        )
    else:
        logger.info("Inferencia finalizada con éxito para todos los casos.")