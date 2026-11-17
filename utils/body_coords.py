"""
utils/body_coords.py

Diagnóstico de las coordenadas "body" (FiLM) entre BigMAC y Utrecht (plan B2b).

El modelo aprendió en BigMAC qué anatomía hay en cada coordenada relativa al bbox del
NAC-PET. En Utrecht ese bbox sale ~2x más ancho de delante a atrás (halo de scatter del
NAC-PET) y el FOV es de cuerpo entero, así que los pulmones caen en z ≈ 0.60-0.92 en vez
de 0.49-0.69. Este módulo construye bboxes alternativos para Utrecht que reproducen la
relación bbox/anatomía de BigMAC:

    attrs    el bbox guardado en el HDF5 (lo que se usa hoy)
    pet_xy   x, y desde una máscara de cuerpo del NAC-PET (umbral por corte, sin CT) +
             márgenes de BigMAC; z como attrs. Es el único modo desplegable.
    ct_xy    x, y desde el cuerpo del CT (oráculo) + márgenes de BigMAC; z como attrs
    ct_xyz   ct_xy + z reescalado para que los pulmones caigan donde en BigMAC (oráculo)

Calibración (una vez, ~15 min; lee los 75 sujetos de BigMAC):
    python -m utils.body_coords --h5_dir /mnt/c/Users/edyA0/Proyectos/data_multimask \
        --out /mnt/d/Data/007_pseudoCT/eval/coord_calib_bigmac.json

Owner: Edwing Ulin
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

BBOX_MODES = ["attrs", "pet_xy", "ct_xy", "ct_xyz"]

PET_STEP = 2            # la máscara del PET se calcula a resolución 1/2
PET_FRAC = 0.3          # umbral: fracción del percentil 95 de cada corte axial
EXTENT_MIN_COUNT = 50   # vóxeles (resolución completa) para que un índice cuente como cuerpo
LUNG_AREA_FRAC = 0.25   # cortes con aire interno > 25 % del máximo = losa pulmonar


def body_mask_pet(pet, frac=PET_FRAC, sigma=2.0, q=95):
    """Máscara de cuerpo a resolución 1/PET_STEP desde el NAC-PET (sin CT)."""
    sub = ndi.gaussian_filter(pet[::PET_STEP, ::PET_STEP, ::PET_STEP].astype(np.float32), sigma)
    # Umbral por corte axial: el cerebro o la vejiga no fijan un umbral global.
    ref = np.percentile(sub.reshape(-1, sub.shape[2]), q, axis=0)
    ref = ndi.maximum_filter1d(ref, 15)
    m = ndi.binary_opening(sub > frac * ref[None, None, :], iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    return np.stack([ndi.binary_fill_holes(m[:, :, z]) for z in range(m.shape[2])], axis=2)


def upsample_mask(m, shape):
    """Máscara a resolución 1/PET_STEP -> resolución completa (vecino más cercano)."""
    for ax in range(3):
        m = np.repeat(m, PET_STEP, axis=ax)
    return m[: shape[0], : shape[1], : shape[2]]


def mask_extent(m, step=1, min_count=EXTENT_MIN_COUNT):
    """(lo, hi) por eje en índices de resolución completa; hi exclusivo (como compute_body_bbox)."""
    thr = max(min_count / step ** 2, 1)
    lo, hi = [], []
    for ax in range(3):
        idx = np.where(m.sum(axis=tuple(a for a in range(3) if a != ax)) > thr)[0]
        if idx.size == 0:
            lo.append(0)
            hi.append(m.shape[ax] * step)
        else:
            lo.append(int(idx[0]) * step)
            hi.append((int(idx[-1]) + 1) * step)
    return np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)


def lung_z_range(mask, contiguous=False):
    """
    Losa pulmonar (z_lo, z_hi exclusivo) desde la clase 3 (aire interno) de la máscara del CT.

    contiguous=False (original): del primer al último corte con aire > 25 % del máximo. En 5
        de 25 exámenes de Utrecht (entre ellos los de 2 y 4 años) incluye gas intestinal y la
        losa mide 24-45 cm. Se mantiene por defecto para que mae_thorax sea comparable entre
        las evaluaciones ya hechas.
    contiguous=True: solo el tramo continuo alrededor del corte con más aire (los pulmones).
    """
    area = (mask == 3).sum(axis=(0, 1))
    if area.max() == 0:
        return None
    thr = LUNG_AREA_FRAC * area.max()
    if not contiguous:
        idx = np.where(area > thr)[0]
        return int(idx[0]), int(idx[-1]) + 1
    lo = hi = int(np.argmax(area))
    while lo > 0 and area[lo - 1] > thr:
        lo -= 1
    while hi < len(area) - 1 and area[hi + 1] > thr:
        hi += 1
    return lo, hi + 1


def margins(bbox, extent):
    """Márgenes del bbox respecto a la extensión del cuerpo, relativos a su longitud (por eje)."""
    (b_lo, b_hi), (e_lo, e_hi) = bbox, extent
    length = np.maximum(e_hi - e_lo, 1)
    return (e_lo - b_lo) / length, (b_hi - e_hi) / length


def apply_margins(extent, m_lo, m_hi):
    e_lo, e_hi = extent
    length = np.maximum(e_hi - e_lo, 1)
    return e_lo - np.asarray(m_lo) * length, e_hi + np.asarray(m_hi) * length


def build_bbox(mode, attrs_bbox, mask, pet_mask, calib):
    """bbox (lo, hi) del modo pedido; los ejes que el modo no cambia se toman de attrs_bbox."""
    lo, hi = (np.asarray(v, dtype=np.float64).copy() for v in attrs_bbox)
    if mode == "attrs":
        return lo, hi
    if mode == "pet_xy":
        ext, cal = mask_extent(pet_mask, PET_STEP), calib["pet"]
    else:
        ext, cal = mask_extent(mask > 0), calib["ct"]
    new_lo, new_hi = apply_margins(ext, cal["margin_lo"], cal["margin_hi"])
    lo[:2], hi[:2] = new_lo[:2], new_hi[:2]
    if mode == "ct_xyz":
        lung = lung_z_range(mask)
        if lung is not None:
            a, b = calib["lung_z_rel"]
            length = (lung[1] - lung[0]) / (b - a)
            lo[2] = lung[0] - a * length
            hi[2] = lo[2] + length
    return lo, hi


def calibrate(h5_files):
    """Márgenes medios bbox/cuerpo (CT y PET) y posición relativa de los pulmones en BigMAC."""
    ct_m, pet_m, lungs = [], [], []
    for f in h5_files:
        with h5py.File(f, "r") as hf:
            for sub_id in hf:
                g = hf[sub_id]
                bbox = (np.asarray(g.attrs["body_lo"], float), np.asarray(g.attrs["body_hi"], float))
                mask = g["mask"][:]
                ct_m.append(margins(bbox, mask_extent(mask > 0)))
                pet_m.append(margins(bbox, mask_extent(body_mask_pet(g["pet"][:]), PET_STEP)))
                lung = lung_z_range(mask)
                if lung is not None:
                    span = bbox[1][2] - bbox[0][2]
                    lungs.append(((lung[0] - bbox[0][2]) / span, (lung[1] - bbox[0][2]) / span))
                print(f"{sub_id}: ct margins lo {np.round(ct_m[-1][0], 2)} hi {np.round(ct_m[-1][1], 2)}, "
                      f"lung z rel {np.round(lungs[-1], 3) if lung else '-'}", flush=True)

    def summary(ms):
        return {"margin_lo": np.mean([m[0] for m in ms], 0).tolist(),
                "margin_hi": np.mean([m[1] for m in ms], 0).tolist()}
    lungs = np.array(lungs)
    return {"n": len(ct_m), "ct": summary(ct_m), "pet": summary(pet_m),
            "lung_z_rel": lungs.mean(0).tolist(), "lung_z_rel_std": lungs.std(0).tolist()}


def main():
    parser = argparse.ArgumentParser("Calibración de coordenadas body en BigMAC")
    parser.add_argument("--h5_dir", required=True, help="Carpeta con los fold_*.h5 de BigMAC")
    parser.add_argument("--out", required=True, help="JSON de salida")
    args = parser.parse_args()
    calib = calibrate(sorted(Path(args.h5_dir).glob("fold_*.h5")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
