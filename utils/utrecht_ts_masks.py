"""
utils/utrecht_ts_masks.py

Máscaras anatómicas para Utrecht a partir de TotalSegmentator (tarea "total", --ml), con el
MISMO mapeo de etiquetas que las máscaras de BigMAC (data/generate_hdf5.py; BigMAC trae
organ_seg con la numeración de TotalSegmentator v2). Hasta v25, Utrecht usaba umbrales de HU
(data/generate_hdf5_utrecht.py), así que las clases no significaban lo mismo en las dos
cohortes:

    clase   BigMAC / este script            Utrecht por umbrales
    2       cuerpo                          cuerpo
    3       lóbulos pulmonares              aire < -500 HU (incluye gas intestinal)
    4       órganos, músculos, vasos        (no existe)
    5       huesos completos (costillas...) >= 200 HU
    6       vértebras y sacro               >= 600 HU

El cuerpo (clase > 0) se toma de la máscara de umbrales actual (sin camilla). Las etiquetas
de TotalSegmentator fuera de ese cuerpo se ignoran.

Uso:
    python -m utils.utrecht_ts_masks                 # informe de discrepancias (no escribe nada)
    python -m utils.utrecht_ts_masks --write         # copia los HDF5 a utrecht_hdf5_ts con la máscara nueva

Owner: Edwing Ulin
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

# Mismas listas que data/generate_hdf5.H5pyFoldGenerator (numeración de TotalSegmentator v2).
VERTEBRAE = list(range(44, 51)) + list(range(32, 44)) + list(range(27, 32)) + [25, 26]
STANDARD_BONES = list(range(92, 116)) + [116, 117] + [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91]
CRITICAL_ORGANS = [1, 2, 3, 4, 5, 6, 7] + [18, 19, 20, 21, 22, 23] + [8, 9, 24, 25]
LUNG_LOBES = [10, 11, 12, 13, 14]
VESSELS = list(range(52, 69))
MUSCLES = list(range(80, 90)) + [50, 51]


def anatomical_mask(body, seg):
    """Mismo orden de asignación que BigMAC: músculos/vasos 4, pulmón 3, órganos 4, hueso 5, vértebras 6."""
    mask = np.zeros(seg.shape, dtype=np.uint8)
    mask[body] = 2
    for labels, cls in ((MUSCLES, 4), (VESSELS, 4), (LUNG_LOBES, 3), (CRITICAL_ORGANS, 4),
                        (STANDARD_BONES, 5), (VERTEBRAE, 6)):
        mask[np.isin(seg, labels) & body] = cls
    return mask


def load_seg(seg_dir, sub):
    p = Path(seg_dir) / f"{sub}.nii.gz"
    return np.asarray(nib.as_closest_canonical(nib.load(str(p))).dataobj).astype(np.int16) if p.is_file() else None


def read_ages(path):
    return {r["sub_id"]: int(re.match(r"(\d+)Y", r["age"]).group(1)) for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser("Máscaras de Utrecht desde TotalSegmentator (mapeo de BigMAC)")
    ap.add_argument("--h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    ap.add_argument("--seg_dir", default="/mnt/d/Data/007_pseudoCT/eval/totalseg")
    ap.add_argument("--out_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5_ts")
    ap.add_argument("--report", default="/mnt/d/Data/007_pseudoCT/eval/utrecht_mask_agreement.csv")
    ap.add_argument("--participants", default="/mnt/d/Data/007_pseudoCT/utrecht_bigmac_space/participants.csv")
    ap.add_argument("--write", action="store_true", help="Escribir los HDF5 nuevos en --out_dir")
    args = ap.parse_args()
    ages = read_ages(args.participants)
    out_dir = Path(args.out_dir)
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for h5 in sorted(Path(args.h5_dir).glob("*.h5")):
        dst = out_dir / h5.name
        if args.write:
            shutil.copy2(h5, dst)
            print(f"copiado {h5.name}", flush=True)
        with h5py.File(dst if args.write else h5, "r+" if args.write else "r") as hf:
            for sub in hf:
                g = hf[sub]
                old = g["mask"][:]
                seg = load_seg(args.seg_dir, sub)
                if seg is None or seg.shape != old.shape:
                    raise SystemExit(f"{sub}: falta la segmentación o no coincide la malla")
                body = old > 0
                new = anatomical_mask(body, seg)
                ct = g["ct"][:]
                ts_bone = new >= 5
                ts_lung = new == 3
                thr_bone = old >= 5
                thr_air = old == 3
                rows.append({
                    "subject": sub, "age": ages.get(sub, ""), "split": h5.stem,
                    # Hueso anatómico (TotalSegmentator) que el umbral de 200 HU deja fuera de las clases de hueso.
                    "bone_missed_by_threshold": float((ts_bone & ~thr_bone).sum() / max(ts_bone.sum(), 1)),
                    # Vóxeles de hueso por umbral que TotalSegmentator no llama hueso (calcificaciones, contraste...).
                    "threshold_bone_outside_ts": float((thr_bone & ~ts_bone).sum() / max(thr_bone.sum(), 1)),
                    "ts_bone_median_hu": float(np.median(ct[ts_bone])) if ts_bone.any() else np.nan,
                    "lung_lobes_ml": float(ts_lung.sum() * 1.5234375 ** 2 * 2.0 / 1000),
                    "air_class_ml": float(thr_air.sum() * 1.5234375 ** 2 * 2.0 / 1000),
                    "air_class_outside_lungs": float((thr_air & ~ts_lung).sum() / max(thr_air.sum(), 1)),
                    "lung_lobes_not_air_class": float((ts_lung & ~thr_air).sum() / max(ts_lung.sum(), 1)),
                    "organ_frac": float((new == 4).mean() / max(body.mean(), 1e-9)),
                })
                r = rows[-1]
                print(f"{sub} ({r['age']} a): hueso fuera del umbral {100 * r['bone_missed_by_threshold']:.0f} %, "
                      f"HU mediana hueso {r['ts_bone_median_hu']:.0f}; aire fuera de pulmón "
                      f"{100 * r['air_class_outside_lungs']:.0f} %, pulmón no-aire {100 * r['lung_lobes_not_air_class']:.0f} %",
                      flush=True)
                if args.write:
                    attrs = dict(g["mask"].attrs)
                    chunks = g["mask"].chunks
                    del g["mask"]
                    d = g.create_dataset("mask", data=new, chunks=chunks, compression="gzip", compression_opts=4)
                    for k, v in attrs.items():
                        d.attrs[k] = v
                    g.attrs["mask_source"] = "totalsegmentator_bigmac_mapping"
    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"informe: {args.report}")


if __name__ == "__main__":
    main()
