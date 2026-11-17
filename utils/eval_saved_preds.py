"""
utils/eval_saved_preds.py

Reevalúa pseudo-CT ya guardados (utrecht_pred/<label>/pet_only/<sub>/ct.nii.gz) contra los
HDF5 de --h5_dir, sin GPU. Sirve para medir todos los modelos con las máscaras anatómicas de
TotalSegmentator (utrecht_hdf5_ts), las mismas que definen las clases en BigMAC, y así comparar
las métricas por tejido entre cohortes. Mismas métricas y regiones que utils/evaluate_models.py.

Las filas se añaden a --out_csv (por defecto utrecht_B4_ts.csv); "fold" se copia de
utrecht_B4.csv para cada (label, sujeto).

Ejemplo:
    python -m utils.eval_saved_preds --labels water v18 v23

Owner: Edwing Ulin
"""

import argparse
import csv
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from utils.body_coords import body_mask_pet
from utils.evaluate_models import case_metrics, case_regions

FIELDS = ["label", "expr", "fold", "h5", "subject", "regime", "bbox", "mae_body", "me_body", "mae_air", "mae_soft",
          "mae_bone", "dice_bone", "mae_core", "mae_thorax", "mae_rest", "ghost_frac", "mae_body_clamp",
          "ghost_frac_clamp"]


def main():
    ap = argparse.ArgumentParser("Reevaluar pseudo-CT guardados con otras máscaras")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5_ts")
    ap.add_argument("--pred_dir", default="/mnt/d/Data/007_pseudoCT/eval/utrecht_pred")
    ap.add_argument("--b4_csv", default="/mnt/d/Data/007_pseudoCT/eval/utrecht_B4.csv")
    ap.add_argument("--out_csv", default="/mnt/d/Data/007_pseudoCT/eval/utrecht_B4_ts.csv")
    ap.add_argument("--bone_hu", type=float, default=300.0)
    args = ap.parse_args()

    origin = {(r["label"], r["subject"]): r for r in csv.DictReader(open(args.b4_csv)) if r["regime"] == "pet_only"}
    new = not Path(args.out_csv).is_file()
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for h5 in sorted(Path(args.h5_dir).glob("*.h5")):
            with h5py.File(h5, "r") as hf:
                for sub in hf:
                    preds = {l: Path(args.pred_dir) / l / "pet_only" / sub / "ct.nii.gz" for l in args.labels}
                    preds = {l: p for l, p in preds.items() if p.is_file()}
                    if not preds:
                        continue
                    g = hf[sub]
                    ct, mask = g["ct"][:], g["mask"][:]
                    regions = case_regions(mask, body_mask_pet(g["pet"][:]))
                    for label, p in preds.items():
                        pred = np.asarray(nib.as_closest_canonical(nib.load(str(p))).dataobj, dtype=np.float32)
                        o = origin.get((label, sub), {})
                        row = {"label": label, "expr": o.get("expr", label), "fold": o.get("fold", ""),
                               "h5": h5.name, "subject": sub, "regime": "pet_only", "bbox": "attrs"}
                        row.update(case_metrics(pred, ct, mask, args.bone_hu, regions))
                        w.writerow(row)
                        f.flush()
                        print(f"{label} {sub}: aire {row['mae_air']:.0f}, blando {row['mae_soft']:.0f}, "
                              f"hueso {row['mae_bone']:.0f} HU", flush=True)


if __name__ == "__main__":
    main()
