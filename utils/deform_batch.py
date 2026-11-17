"""
utils/deform_batch.py

Tarea T3 en lote: genera la cohorte "BigMAC pediátrica" deformando adultos de BigMAC hacia
niños de los folds de Utrecht (nunca test) con utils/deform_to_pediatric.deform_pair.

Prioridad a los niños pequeños (los que peor predice v21):
    < YOUNG_AGE años  -> los N_YOUNG adultos más pequeños de BigMAC (ancho de tórax)
    >= YOUNG_AGE años -> N_OLD adultos, rotando entre los N_POOL más pequeños
Las parejas de niños pequeños van primero; el script se puede interrumpir y reanudar
(las parejas ya escritas se saltan).

Salida (mismo formato que los HDF5 de entrenamiento):
    <out_dir>/fold_k.h5   un grupo "<adulto>_to_<niño>" por pareja, en el fold k DEL NIÑO.
                          El modelo del fold k no debe entrenar con fold_k.h5 (copia la forma
                          de sus pacientes de validación).
        pet, ct (float32), mri (2 canales, float32), mask (uint8)
        attrs: has_mri=True, has_topo=False, body_lo/body_hi (desde el PET, como el resto),
               adult, child, child_age, target_split, adult_split, dice_*, jac_fold_frac, qa_ok
    <out_dir>/deform_qa.csv   métricas de cada pareja
    <out_dir>/qa/<pareja>.png corte coronal de control

Uso:
    nohup python -m utils.deform_batch > logs/deform_batch.log 2>&1 &

Owner: Edwing Ulin
"""

import argparse
import csv
import time
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk

from data.preprocessing import compute_body_bbox
from utils.deform_to_pediatric import deform_pair, figure, load_subject, qa_title

CHUNKS = (32, 32, 34)


def plan_pairs(cohort_csv, young_age, n_young, n_old, n_pool):
    rows = list(csv.DictReader(open(cohort_csv)))
    adults = sorted((r for r in rows if r["cohort"] == "BigMAC"), key=lambda r: float(r["chest_width_cm"]))
    children = sorted((r for r in rows if r["cohort"] == "Utrecht" and r["split"] != "test"),
                      key=lambda r: float(r["age"]))
    pairs = []
    young = [c for c in children if float(c["age"]) < young_age]
    old = [c for c in children if float(c["age"]) >= young_age]
    # Niños pequeños: agrupado por adulto para leer cada adulto una sola vez.
    for a in adults[:n_young]:
        for c in young:
            pairs.append((a["subject"], c["subject"], float(c["age"])))
    pool = adults[:n_pool]
    for i, c in enumerate(old):
        for j in range(n_old):
            pairs.append((pool[(i * n_old + j) % len(pool)]["subject"], c["subject"], float(c["age"])))
    return pairs


def write_pair(h5_path, name, warped, attrs):
    with h5py.File(h5_path, "a") as hf:
        g = hf.create_group(name)
        for k in ("pet", "ct"):
            g.create_dataset(k, data=warped[k].astype(np.float32), chunks=CHUNKS, compression="gzip")
        g.create_dataset("mri", data=warped["mri"].astype(np.float32), chunks=(1,) + CHUNKS, compression="gzip")
        g.create_dataset("mask", data=warped["mask"].astype(np.uint8), chunks=CHUNKS, compression="gzip")
        for k, v in attrs.items():
            g.attrs[k] = v


def exists(h5_path, name):
    if not h5_path.is_file():
        return False
    with h5py.File(h5_path, "r") as hf:
        return name in hf


def main():
    p = argparse.ArgumentParser("Generación en lote de la cohorte BigMAC pediátrica (T3)")
    p.add_argument("--cohort_csv", default="/mnt/d/Data/007_pseudoCT/eval/figuras/cohortes.csv")
    p.add_argument("--bigmac_h5_dir", default="/mnt/c/Users/edyA0/Proyectos/data_multimask")
    p.add_argument("--utrecht_h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    p.add_argument("--out_dir", default="/mnt/d/Data/007_pseudoCT/bigmac_ped_hdf5")
    p.add_argument("--young_age", type=float, default=13)
    p.add_argument("--n_young", type=int, default=10, help="Adultos por niño pequeño")
    p.add_argument("--n_old", type=int, default=3, help="Adultos por niño >= young_age")
    p.add_argument("--n_pool", type=int, default=20, help="Adultos más pequeños entre los que rotar")
    p.add_argument("--threads", type=int, default=8, help="Hilos de SimpleITK (la GPU entrena a la vez)")
    p.add_argument("--max_pairs", type=int, default=None, help="Solo las N primeras parejas (prueba)")
    # Mismos parámetros de registro que el prototipo.
    p.add_argument("--mesh", type=int, nargs=3, default=[6, 6, 8])
    p.add_argument("--iters_affine", type=int, default=150)
    p.add_argument("--iters_bspline", type=int, default=60)
    p.add_argument("--method", default="adaptive")
    p.add_argument("--max_fold", type=float, default=0.001)
    p.add_argument("--demons_sigma", type=float, default=1.5)
    args = p.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)

    out = Path(args.out_dir)
    (out / "qa").mkdir(parents=True, exist_ok=True)
    pairs = plan_pairs(args.cohort_csv, args.young_age, args.n_young, args.n_old, args.n_pool)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    n_young = sum(age < args.young_age for _, _, age in pairs)
    print(f"{len(pairs)} parejas ({n_young} con niños < {args.young_age:.0f} años) -> {out}", flush=True)

    qa_csv = out / "deform_qa.csv"
    fields = ["pair", "adult", "child", "child_age", "target_split", "method", "dice_body", "dice_lung",
              "dice_bone", "mae_ct_body", "jac_fold_frac", "qa_ok", "seconds"]
    new_csv = not qa_csv.is_file()
    cached_adult = (None, None, None)
    t_start = time.time()
    for i, (adult_id, child_id, age) in enumerate(pairs, 1):
        name = f"{adult_id}_to_{child_id}"
        child, child_split = load_subject(args.utrecht_h5_dir, child_id, ["pet", "ct", "mask"])
        h5_path = out / f"{child_split}.h5"
        if exists(h5_path, name):
            print(f"[{i}/{len(pairs)}] {name}: ya existe, se salta", flush=True)
            continue
        if cached_adult[0] != adult_id:
            adult, adult_split = load_subject(args.bigmac_h5_dir, adult_id, ["pet", "mri", "ct", "mask"])
            cached_adult = (adult_id, adult, adult_split)
        _, adult, adult_split = cached_adult

        try:
            warped, report, child_crop = deform_pair(adult, child, args)
        except Exception as e:      # una pareja que falla no debe parar el lote
            print(f"[{i}/{len(pairs)}] {name}: FALLÓ ({e})", flush=True)
            continue
        st = report["stages"]["no_rigido"]
        qa_ok = bool(st["dice_body"] > 0.9 and st["jac_fold_frac"] < 0.001)
        lo, hi = compute_body_bbox(warped["pet"])
        write_pair(h5_path, name, warped, {
            "has_mri": True, "has_topo": False, "body_lo": lo, "body_hi": hi,
            "adult": adult_id, "adult_split": adult_split, "child": child_id, "child_age": age,
            "target_split": child_split, "qa_ok": qa_ok, "method": report["method"],
            **{k: float(v) for k, v in st.items()},
        })
        figure(out / "qa" / f"{name}.png", child_crop, adult, warped, qa_title(adult_id, child_id, report))

        row = {"pair": name, "adult": adult_id, "child": child_id, "child_age": age, "target_split": child_split,
               "qa_ok": qa_ok, "seconds": report["seconds"], "method": report["method"], **{k: round(st[k], 4) for k in fields if k in st}}
        with open(qa_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_csv:
                w.writeheader()
                new_csv = False
            w.writerow(row)
        eta = (time.time() - t_start) / i * (len(pairs) - i) / 60
        print(f"[{i}/{len(pairs)}] {name} ({age:.0f} años, {child_split}): Dice cuerpo {st['dice_body']:.2f}, "
              f"pulmón {st['dice_lung']:.2f}, hueso {st['dice_bone']:.2f}, MAE {st['mae_ct_body']:.0f} HU, "
              f"plegado {100 * st['jac_fold_frac']:.2f} % [{report['method']}] {'OK' if qa_ok else 'REVISAR'} "
              f"({report['seconds']} s; quedan ~{eta:.0f} min)", flush=True)
    print(f"Terminado en {(time.time() - t_start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
