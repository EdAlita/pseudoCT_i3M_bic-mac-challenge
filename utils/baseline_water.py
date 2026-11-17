"""
utils/baseline_water.py

Baseline sin aprendizaje para el paper: pseudo-CT "agua" a partir del NAC-PET solo.
Dentro del contorno del cuerpo detectado en el NAC-PET (utils/body_coords.body_mask_pet,
el mismo que da las coordenadas del modelo) se pone WATER_HU; fuera, aire (-1000 HU).
Es la corrección de atenuación clásica por contorno uniforme, sin pulmones ni hueso.

Escribe filas con label "water" en el mismo CSV y formato que utils/evaluate_models.py
(mismas regiones y métricas), y guarda el pseudo-CT en <save_dir>/water/pet_only/<sub>/ct.nii.gz
para que make_presentation_figures pueda tratarlo como un modelo más.

Ejemplo:
    python -m utils.baseline_water

Owner: Edwing Ulin
"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from utils.body_coords import body_mask_pet, upsample_mask
from utils.evaluate_models import case_metrics, case_regions, save_prediction

WATER_HU = 0.0


def main():
    p = argparse.ArgumentParser("Baseline de agua desde el contorno del NAC-PET")
    p.add_argument("--h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    p.add_argument("--out_csv", default="/mnt/d/Data/007_pseudoCT/eval/utrecht_B4.csv")
    p.add_argument("--save_dir", default="/mnt/d/Data/007_pseudoCT/eval/utrecht_pred")
    p.add_argument("--ref_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_bigmac_space")
    p.add_argument("--label", default="water")
    p.add_argument("--bone_hu", type=float, default=300.0)
    args = p.parse_args()

    header = next(csv.reader(open(args.out_csv)))
    save_args = SimpleNamespace(ref_dir=args.ref_dir, save_dir=args.save_dir)
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, header)
        for h5 in sorted(Path(args.h5_dir).glob("*.h5")):
            with h5py.File(h5, "r") as hf:
                for sub in hf:
                    g = hf[sub]
                    pet, ct, mask = g["pet"][:], g["ct"][:], g["mask"][:]
                    pet_mask = body_mask_pet(pet)
                    body_pet = upsample_mask(pet_mask, ct.shape)
                    pred = np.where(body_pet, WATER_HU, -1000.0).astype(np.float32)
                    row = {"label": args.label, "expr": "water_from_nacpet_contour", "fold": "baseline",
                           "h5": h5.name, "subject": sub, "regime": "pet_only", "bbox": "attrs"}
                    row.update(case_metrics(pred, ct, mask, args.bone_hu, case_regions(mask, pet_mask)))
                    writer.writerow({k: row.get(k, "") for k in header})
                    f.flush()
                    save_prediction(pred, sub, "pet_only", args.label, save_args)
                    print(f"{sub}: body MAE {row['mae_body']:.1f} HU, aire {row['mae_air']:.0f}, "
                          f"hueso {row['mae_bone']:.0f}", flush=True)


if __name__ == "__main__":
    main()
