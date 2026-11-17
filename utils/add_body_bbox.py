"""
add_body_bbox.py

Precalcula el bounding box corporal (desde NAC-PET) de cada sujeto y lo guarda
como atributos body_lo / body_hi en su grupo HDF5. Lo necesita
DATA.COORDS_MODE = "body" (dataset.py lee los atributos sin cargar volúmenes).

Abre los HDF5 en modo r+: solo añade atributos, no toca los datasets.

Uso:
    python -m utils.add_body_bbox --hdf5_dir /mnt/c/Users/edyA0/Proyectos/data_multimask

Owner: Edwing Ulin
Version: v1.0.0
"""

import argparse
import glob

import h5py

from data.preprocessing import compute_body_bbox
from utils.logger import logger_gen

LOGGER = logger_gen('add_body_bbox')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Añade body_lo/body_hi a cada sujeto de los HDF5")
    parser.add_argument("--hdf5_dir", type=str, required=True,
                        help="Carpeta con los .h5 (se procesan todos los *.h5)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recalcular aunque el sujeto ya tenga body_lo/body_hi")
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{args.hdf5_dir}/*.h5"))
    if not paths:
        raise FileNotFoundError(f"No hay archivos .h5 en {args.hdf5_dir}")

    for path in paths:
        with h5py.File(path, "r+") as hf:
            for sub_id in hf.keys():
                group = hf[sub_id]
                if "body_lo" in group.attrs and not args.overwrite:
                    LOGGER.info(f"{path} {sub_id}: ya tiene bbox, se omite")
                    continue
                lo, hi = compute_body_bbox(group["pet"][:])
                group.attrs["body_lo"] = lo
                group.attrs["body_hi"] = hi
                LOGGER.info(f"{path} {sub_id}: lo={lo.tolist()} hi={hi.tolist()} shape={group['pet'].shape}")
