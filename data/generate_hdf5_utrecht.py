"""
generate_hdf5_utrecht.py

Genera HDF5 para la cohorte pediátrica de UMC Utrecht (NAC-PET + CT, sin MRI,
sin scout y sin segmentación de órganos) a partir de la salida de
data/prepare_utrecht.py, que ya está en el espacio de datos de BigMAC
(malla del CT, 1.5234375 x 1.5234375 x 2.0 mm, LAS). Aquí NO se remuestrea:
se carga con as_closest_canonical igual que generate_hdf5.py (BigMAC), para
reutilizar dataset.py / train.py / inference.py.

Diferencias frente a generate_hdf5.py (BigMAC):
  - El número de cortes es variable (Utrecht es total-body); dataset.py e
    inference.py no dependen de los 531 cortes de BigMAC.
  - No se guarda MRI ni topograma: el grupo lleva attrs has_mri=False y
    has_topo=False, y dataset.py los trata como modalidades "dropped".
  - El bbox corporal (desde NAC-PET) se guarda en attrs body_lo / body_hi
    para DATA.COORDS_MODE = "body".
  - La máscara de pesos (clases 0/2/3/5/6) se deriva de umbrales de HU sobre el
    CT remuestreado; no hay clase 4 (órganos críticos) sin segmentación.
  - Particiones a nivel de PACIENTE (un paciente puede tener varios exámenes):
    test.h5 apartado + fold_k.h5 con GroupKFold.

Owner: Edwing Ulin
Version: v2.1.0
"""

import argparse
import random
import time
import traceback
import gc
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, label as cc_label
from sklearn.model_selection import GroupKFold

from utils.logger import logger_gen
from data.preprocessing import compute_body_bbox

LOGGER = logger_gen('generate_hdf5_utrecht')

# Espaciado de BigMAC (medido en /mnt/d/bic-mac-data/train/*/ct-label/ct.nii.gz)
BIGMAC_SPACING = (1.5234375, 1.5234375, 2.0)

# Lado (vóxeles, ~7.6 mm) de la apertura en el plano axial que separa el cuerpo de la camilla.
BODY_OPENING_VOXELS = 5


def load_canonical(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Carga un NIfTI en RAS (como generate_hdf5.py) y devuelve (datos float32, affine)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(dtype=np.float32), img.affine


class H5pyUtrechtGenerator:
    def __init__(self, params: Dict):
        self.root_path = Path(params["dataset_path"])
        self.output_dir = Path(params["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = params.get("seed", 42)
        self.num_workers = params.get("num_workers", 4)

        self.pet_rel_path = params["pet_rel_path"]
        self.ct_rel_path = params["ct_rel_path"]

        # Umbrales HU para la máscara sintética (sin organ_seg disponible).
        # Aproximados; ajustar si el body-mask resultante se ve mal en QA visual.
        self.hu_body_min = -300        # separa cuerpo de aire externo
        self.hu_air_max = -500         # aire interno (pulmón / intestino)
        self.hu_bone_min = 200         # hueso estándar (clase 5)
        self.hu_dense_bone_min = 600   # hueso denso / cortical-vertebral (clase 6)

        all_subjects = sorted(
            d.name for d in self.root_path.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
        random.seed(self.seed)
        random.shuffle(all_subjects)
        self.subjects = all_subjects
        LOGGER.info(f"Se encontraron {len(self.subjects)} exámenes de Utrecht. Shuffle inicial completado.")

    def _build_mask(self, ct: np.ndarray, pet: np.ndarray) -> np.ndarray:
        """
        Deriva una máscara de pesos (0=fondo, 2=cuerpo, 3=aire interno,
        5=hueso estándar, 6=hueso denso) a partir de umbrales de HU sobre el
        CT ya resampleado. No hay clase 4 (órganos críticos) por falta de
        segmentación de órganos.

        Selección del cuerpo (antes de rellenar huecos):
          - Apertura morfológica en el plano axial (BODY_OPENING_VOXELS): borra
            estructuras finas como las láminas de la camilla y rompe el contacto
            cuerpo-camilla, que si no forman una sola componente.
          - Componentes conexas de lo que queda. En niños pequeños la camilla puede
            ser mayor que el cuerpo, así que no basta con la mayor: entre las
            componentes grandes (>= 5 % de la mayor) se elige la de más actividad
            en el NAC-PET (la camilla y los inmovilizadores no captan).
          - Se recupera el borde perdido en la apertura dilatando la componente lo
            mismo, sin salir del umbral original.
          - Solo después se rellenan huecos por corte axial (eje 2 en RAS), para
            meter los pulmones en el cuerpo: en 3D conectan con el aire exterior
            por la vía aérea. Si se rellenara antes, el interior hueco de la
            camilla la haría crecer y podría ganar al cuerpo.
        """
        body_rough = ct > self.hu_body_min
        inplane = np.ones((BODY_OPENING_VOXELS, BODY_OPENING_VOXELS, 1), dtype=bool)
        labeled, n = cc_label(binary_opening(body_rough, structure=inplane))
        if n == 0:
            LOGGER.warning("No se detectó ningún componente de cuerpo en el CT.")
            return np.zeros_like(ct, dtype=np.uint8)

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignorar el fondo
        activity = np.bincount(labeled.ravel(), weights=np.clip(pet, 0, None).ravel(), minlength=sizes.size)
        candidates = np.where(sizes >= 0.05 * sizes.max())[0]
        body_label = candidates[np.argmax(activity[candidates])]
        if body_label != sizes.argmax():
            LOGGER.info(f"Cuerpo = componente con más actividad PET ({sizes[body_label]} vóx), "
                        f"no la mayor ({sizes.max()} vóx; probablemente la camilla)")

        body_raw = binary_dilation(labeled == body_label, structure=inplane) & body_rough
        del labeled
        body_mask = np.zeros_like(body_raw)
        for k in range(ct.shape[2]):
            body_mask[:, :, k] = binary_fill_holes(body_raw[:, :, k])

        mask = np.zeros_like(ct, dtype=np.uint8)
        mask[body_mask] = 2
        mask[body_mask & (ct < self.hu_air_max)] = 3
        mask[body_mask & (ct > self.hu_bone_min) & (ct <= self.hu_dense_bone_min)] = 5
        mask[body_mask & (ct > self.hu_dense_bone_min)] = 6
        return mask

    def _load_subject_data(self, sub_id: str) -> Optional[Tuple[str, list]]:
        try:
            sub_path = self.root_path / sub_id
            pet, pet_affine = load_canonical(sub_path / self.pet_rel_path)
            ct, ct_affine = load_canonical(sub_path / self.ct_rel_path)

            # prepare_utrecht.py escribe PET y CT en la misma malla de BigMAC: se verifica.
            if pet.shape != ct.shape or not np.allclose(pet_affine, ct_affine, atol=1e-3):
                raise ValueError(f"PET {pet.shape} y CT {ct.shape} no comparten malla: ¿se generaron con prepare_utrecht.py?")
            spacing = np.abs(np.diag(ct_affine)[:3])
            if not np.allclose(spacing, BIGMAC_SPACING, atol=1e-3):
                raise ValueError(f"Spacing {spacing} distinto del de BigMAC {BIGMAC_SPACING}")

            ct = np.clip(ct, -1024, 3071)
            mask = self._build_mask(ct, pet)
            lo, hi = compute_body_bbox(pet)

            return sub_id, [pet, ct, mask, lo, hi]
        except Exception as e:
            LOGGER.error(f"Error cargando {sub_id}: {e}")
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _write_h5(self, h5_name: Path, subjects: List[str]) -> None:
        LOGGER.info(f"== Generando {h5_name} ({len(subjects)} exámenes) con {self.num_workers} hilos ==")
        start_time = time.time()

        with h5py.File(h5_name, "w") as hf:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_sub = {
                    executor.submit(self._load_subject_data, sub_id): sub_id
                    for sub_id in subjects
                }

                for future in as_completed(future_to_sub):
                    sub_id = future_to_sub[future]
                    LOGGER.info(f"Procesando resultados de {sub_id}...")
                    result = future.result()
                    if result:
                        sub_id, arrays = result
                        pet, ct, mask, lo, hi = arrays

                        group = hf.create_group(sub_id)
                        group.create_dataset("pet", data=pet, compression="gzip", compression_opts=4)
                        group.create_dataset("ct", data=ct, compression="gzip", compression_opts=4)
                        group.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
                        group.attrs["has_mri"] = False
                        group.attrs["has_topo"] = False
                        group.attrs["body_lo"] = lo
                        group.attrs["body_hi"] = hi

                        LOGGER.info(f"Sujeto {sub_id} guardado correctamente.")
                        del pet, ct, mask, arrays, result
                        gc.collect()

        LOGGER.info(f"{h5_name.name} terminado en {time.time() - start_time:.2f}s")

    def run_kfold_gen(self, n_splits: int = 5, test_frac: float = 0.2,
                      test_patients: Optional[List[str]] = None) -> None:
        # sub-PUMP266612-20241217 -> PUMP266612: los exámenes de un mismo paciente van juntos
        groups = np.array([s.split("-")[1] for s in self.subjects])
        subjects = np.array(self.subjects)

        if test_patients:
            test_set = set(test_patients)
        else:
            patients = sorted(set(groups))
            random.Random(self.seed).shuffle(patients)
            test_set = set(patients[:max(1, round(test_frac * len(patients)))])

        is_test = np.isin(groups, list(test_set))
        LOGGER.info(f"Test apartado ({len(test_set)} pacientes): {sorted(test_set)}")
        self._write_h5(self.output_dir / "test.h5", subjects[is_test].tolist())

        cv_subjects, cv_groups = subjects[~is_test], groups[~is_test]
        splitter = GroupKFold(n_splits=n_splits)
        for fold_idx, (_, val_idx) in enumerate(splitter.split(cv_subjects, groups=cv_groups)):
            LOGGER.info(f"Fold {fold_idx}: pacientes {sorted(set(cv_groups[val_idx]))}")
            self._write_h5(self.output_dir / f"fold_{fold_idx}.h5", cv_subjects[val_idx].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "HDF5 Generator para la cohorte pediátrica de UMC Utrecht (sin MRI ni scout)"
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Carpeta con los sub-* generados por data/prepare_utrecht.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pet_rel_path", type=str, default="features/nacpet.nii.gz",
                        help="Ruta relativa al NAC-PET dentro de cada carpeta sub-*")
    parser.add_argument("--ct_rel_path", type=str, default="ct-label/ct.nii.gz",
                        help="Ruta relativa al CT dentro de cada carpeta sub-*")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Fracción de PACIENTES apartados en test.h5")
    parser.add_argument("--test_patients", type=str, nargs="*", default=None,
                        help="IDs de paciente para test (ej. PUMP108861 PUMP266612); "
                             "si se dan, ignoran --test_frac")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    generator = H5pyUtrechtGenerator({
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "num_workers": args.workers,
        "pet_rel_path": args.pet_rel_path,
        "ct_rel_path": args.ct_rel_path,
    })
    generator.run_kfold_gen(n_splits=args.folds, test_frac=args.test_frac,
                            test_patients=args.test_patients)
