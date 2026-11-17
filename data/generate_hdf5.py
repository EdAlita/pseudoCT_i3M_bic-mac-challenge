"""
generate_hdf5.py

Genera 5 archivos HDF5 independientes. Cada archivo representa un "chunk" o fold
del dataset total. Durante el entrenamiento se cargarán 4 para train y 1 para val.
Incluye soporte para multithreading para acelerar la carga de datos.

Owner: Edwing Ulin
Version: v1.5.0
"""

import argparse
import time
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold
import traceback
import gc

# Asegúrate de que el nombre coincida con tu archivo en utils/
from utils.logger import logger_gen

LOGGER = logger_gen('generate_hdf5')

class H5pyFoldGenerator:
    def __init__(self, params: Dict):
        self.root_path = Path(params["dataset_path"])
        self.output_dir = Path(params["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = params.get("seed", 42)
        self.num_workers = params.get("num_workers", 4)
        
        self.pet_rel_path = "features/nacpet.nii.gz"
        self.mri_in_path = "features/mri_combined_in_phase.nii.gz"
        self.mri_out_path = "features/mri_combined_out_phase.nii.gz"
        self.topo_rel_path = "features/topogram.nii.gz"
        self.ct_rel_path = "ct-label/ct.nii.gz"
        
        self.ct_organ_path = "ct-label/organ_seg.nii.gz"
        self.ct_body_path = "ct-label/body_seg.nii.gz"

        self.vertebrae_cervical = list(range(44, 51))  # C7 to C1
        self.vertebrae_thoracic = list(range(32, 44))  # T12 to T1
        self.vertebrae_lumbar = list(range(27, 32))    # L5 to L1
        self.vertebrae_sacrum = [25, 26]               # Sacrum, S1
        self.vertebrae_all = self.vertebrae_cervical + self.vertebrae_thoracic + \
                             self.vertebrae_lumbar + self.vertebrae_sacrum
        


        self.ribs = list(range(92, 116))               # rib_left_1 to rib_right_12
        self.sternum = [116]
        self.costal_cartilages = [117]
        self.other_bones = [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91] # Humerus, Scapula, etc.

        self.standard_bones = self.ribs + self.sternum + self.costal_cartilages + self.other_bones
        self.gi_tract = [18, 19, 20, 21, 22, 23]
        self.other_organs = [8, 9, 24, 25]
        self.critical_organs = [1, 2, 3, 4, 5, 6, 7] + self.gi_tract + self.other_organs # Spleen, Kidneys, GB, Liver, Stomach, Pancreas
        self.internal_air = [10, 11, 12, 13, 14]     # Lung lobes
        self.vessels = list(range(52, 69))           # Aorta, Vena Cava, Iliacs, etc.
        self.hearth = [50, 51]
        self.muscles = list(range(80, 90))  + self.hearth         # Gluteus, Iliopsoas, Autochthon

        all_subjects = sorted([d.name for d in self.root_path.iterdir() if d.is_dir() and d.name.startswith("sub-")])
        random.seed(self.seed)
        random.shuffle(all_subjects) 
        
        self.subjects = all_subjects
        LOGGER.info(f"Se encontraron {len(self.subjects)} sujetos. Shuffle inicial completado.")

    def _load_subject_data(self, sub_id: str) -> Optional[Tuple[str, List[np.ndarray]]]:
        """
        Carga y procesa los datos de un sujeto. 
        Retorna el ID y una lista de arrays si tiene éxito.
        """
        try:
            sub_path = self.root_path / sub_id
            
            pet = nib.as_closest_canonical(nib.load(sub_path / self.pet_rel_path)).get_fdata(dtype=np.float32)
            mri_in = nib.as_closest_canonical(nib.load(sub_path / self.mri_in_path)).get_fdata(dtype=np.float32)
            mri_out = nib.as_closest_canonical(nib.load(sub_path / self.mri_out_path)).get_fdata(dtype=np.float32)
            ct = nib.as_closest_canonical(nib.load(sub_path / self.ct_rel_path)).get_fdata(dtype=np.float32)
            topo = nib.as_closest_canonical(nib.load(sub_path / self.topo_rel_path)).get_fdata(dtype=np.float32)

            mri_combined = np.stack([mri_in, mri_out], axis=0)

            ct_organ = nib.as_closest_canonical(nib.load(sub_path / self.ct_organ_path)).get_fdata()
            ct_body = nib.as_closest_canonical(nib.load(sub_path / self.ct_body_path)).get_fdata()

            mask = np.zeros_like(ct_organ, dtype=np.uint8)

            mask[ct_body > 0] = 2

            mask[np.isin(ct_organ, self.muscles)] = 4
            mask[np.isin(ct_organ, self.vessels)] = 4

            mask[np.isin(ct_organ, self.internal_air)] = 3      # Clase 3: Aire Interno
            mask[np.isin(ct_organ, self.critical_organs)] = 4   # Clase 4: Órganos Críticos
            mask[np.isin(ct_organ, self.standard_bones)] = 5    # Clase 5: Huesos Estándar
            mask[np.isin(ct_organ, self.vertebrae_all)] = 6  # Clase 6: Huesos Densos (Vértebras/Sacro)

            data = [pet, mri_combined, ct, topo, mask]
            
            return sub_id, data
        
        except Exception as e:
            LOGGER.error(f"Error cargando {sub_id}: {e}")
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            return None

    def run_kfold_gen(self, n_splits: int = 5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        subject_arr = np.array(self.subjects)

        for fold_idx, (_, val_idx) in enumerate(kf.split(subject_arr)):
            current_fold_subjects = subject_arr[val_idx].tolist()
            random.shuffle(current_fold_subjects)
            
            h5_name = self.output_dir / f"fold_{fold_idx}.h5"
            LOGGER.info(f"== Generando {h5_name} ({len(current_fold_subjects)} sujetos) con {self.num_workers} hilos ==")
            
            start_fold_time = time.time()
            
            with h5py.File(h5_name, "w") as hf:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_sub = {executor.submit(self._load_subject_data, sub_id): sub_id for sub_id in current_fold_subjects}
                    
                    for future in as_completed(future_to_sub):
                        sub_id = future_to_sub[future] # Recuperamos el ID
                        LOGGER.info(f"Procesando resultados de {sub_id}...")
                        result = future.result()
                        if result:
                            sub_id, arrays = result
                            pet, mri_comb, ct, topo, mask = arrays
                            
                            group = hf.create_group(sub_id)
                            group.create_dataset("pet", data=pet, compression="gzip", compression_opts=4)
                            group.create_dataset("mri", data=mri_comb, compression="gzip", compression_opts=4)
                            group.create_dataset("ct", data=ct, compression="gzip", compression_opts=4)
                            group.create_dataset("topo", data=topo, compression="gzip", compression_opts=4)
                            group.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
                            
                            LOGGER.info(f"Sujeto {sub_id} guardado correctamente.")

                            del pet, mri_comb, ct, topo, mask
                            del arrays
                            del result
                            gc.collect()

            LOGGER.info(f"Fold {fold_idx} terminado en {time.time() - start_fold_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("HDF5 K-Fold Generator (Multithreaded)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Ruta a 'train'")
    parser.add_argument("--output_dir", type=str, required=True, help="Carpeta de salida para los .h5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="Número de hilos para carga paralela")

    args = parser.parse_args()

    generator = H5pyFoldGenerator({
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "num_workers": args.workers
    })
    generator.run_kfold_gen(n_splits=args.folds)