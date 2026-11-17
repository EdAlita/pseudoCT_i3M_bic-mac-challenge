"""
utils/evaluate_models.py

Evaluación volumétrica (sliding window) de checkpoints sobre HDF5 con CT de referencia,
separada por régimen de modalidades:

    full      PET + MRI + topograma
    no_topo   PET + MRI           (topograma en cero)
    no_mri    PET + topograma     (MRI en cero)
    pet_only  solo PET            (MRI y topograma en cero; condición de Utrecht)

Todos los regímenes se calculan en una sola pasada sobre los parches. La normalización
por parche, el anulado de modalidades y las coordenadas son los mismos que en
data/dataset.py, para que la métrica refleje lo que el modelo vio en entrenamiento.
Una modalidad que el sujeto no tiene (Utrecht: has_mri / has_topo = False) solo
permite los regímenes que la anulan.

Modos:
    --oof     Out-of-fold (BigMAC): el modelo del fold k se evalúa sobre fold_k.h5,
              que no vio en entrenamiento (sí se usó para elegir su mejor checkpoint).
    (defecto) Ensemble de los folds sobre todos los HDF5 dados (Utrecht: ningún
              modelo de BigMAC vio esos pacientes).
    --add_expr  Ensemble entre experimentos (misma arquitectura y coordenadas): se promedian
              también los checkpoints de estos experimentos (en --oof, su fold k). Ej. v18 +
              v23 con --expr Multimodal_pCT_v23_utrecht_ft --add_expr Multimodal_pCT_v18_stageA_torso.

Diagnóstico de coordenadas (plan B2b): --bbox_modes attrs pet_xy ct_xy ct_xyz evalúa cada
sujeto con varios bboxes para las coordenadas "body" (ver utils/body_coords.py; los modos
distintos de attrs necesitan --calib). La columna `bbox` del CSV dice cuál se usó.

Métricas extra en cada fila:
    mae_core        MAE del cuerpo sin una banda de 3 vóxeles en el borde
    mae_thorax      MAE del cuerpo en la losa pulmonar (z de la clase 3)
    mae_rest        MAE del cuerpo fuera de esa losa
    ghost_frac      fracción de vóxeles fuera del cuerpo con pseudo-CT > -500 HU
    mae_body_clamp, ghost_frac_clamp
                    lo mismo tras poner a -1000 HU lo que queda fuera de la máscara de
                    cuerpo del NAC-PET (dilatada 4 vóxeles); post-proceso desplegable

Las filas se AÑADEN a --out_csv, así varias ejecuciones (v12, v18, v19...) quedan en
el mismo archivo; al final se imprime la tabla resumen de todo el CSV.

Ejemplo (A6, BigMAC out-of-fold):
    python -m utils.evaluate_models --label v18 \
        --config_path config/mutimodal_v18_stageA.yaml \
        --checkpoints_dir /mnt/d/Data/007_pseudoCT/training/experiments/checkpoints \
        --expr Multimodal_pCT_v18_stageA_torso \
        --h5_dir /mnt/c/Users/edyA0/Proyectos/data_multimask --oof \
        --out_csv /mnt/d/Data/007_pseudoCT/eval/bigmac_oof.csv

Owner: Edwing Ulin
"""

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from tqdm import tqdm

from config.defaults import get_cfg_defaults
from data.preprocessing import compute_body_bbox, patch_coords
from utils.body_coords import BBOX_MODES, body_mask_pet, build_bbox, lung_z_range, upsample_mask
from models.networks import build_model
from utils.logger import logger_gen

logger = logger_gen("evaluate_models")

# régimen -> (mri anulado, topograma anulado)
REGIMES = {
    "full": (False, False),
    "no_topo": (False, True),
    "no_mri": (True, False),
    "pet_only": (True, True),
}

METRICS = ["mae_body", "me_body", "mae_air", "mae_soft", "mae_bone", "dice_bone"]
EXTRA_METRICS = ["mae_core", "mae_thorax", "mae_rest", "ghost_frac", "mae_body_clamp", "ghost_frac_clamp"]
FIELDS = ["label", "expr", "fold", "h5", "subject", "regime", "bbox"] + METRICS + EXTRA_METRICS
GHOST_HU = -500.0


def load_model(cfg, ckpt_path, device):
    model = build_model(cfg).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Cargado {ckpt_path} (época {checkpoint.get('epoch')}, "
                f"best_metric {checkpoint.get('best_metric')})")
    return model


def grid_starts(size, patch, step):
    """Inicios de parche que cubren todo el eje sin padding (el último pegado al borde)."""
    starts = list(range(0, max(size - patch, 0) + 1, step))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def hann_window(patch_size):
    """Ventana de Hann 3D sin ceros en el borde, para promediar parches solapados."""
    w = [np.hanning(p + 2)[1:-1].astype(np.float32) for p in patch_size]
    return w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]


def normalize_patches(pet, mri):
    """Versión en GPU de normalize_pet / normalize_mri (dataset.py): min-max por parche y por canal."""
    p_min = pet.amin(dim=(1, 2, 3, 4), keepdim=True)
    p_max = pet.amax(dim=(1, 2, 3, 4), keepdim=True)
    pet_n = torch.where(p_max > 0, (pet - p_min) / (p_max - p_min).clamp_min(1e-8), pet)
    if mri is None:
        return pet_n, None
    m_min = mri.amin(dim=(2, 3, 4), keepdim=True)
    m_max = mri.amax(dim=(2, 3, 4), keepdim=True)
    denom = m_max - m_min
    mri_n = torch.where(denom > 1e-8, (mri - m_min) / (denom + 1e-8), mri - m_min)
    return pet_n, mri_n


@torch.no_grad()
def predict_volume(cfg, models, pet, mri, topo, bbox, regimes, device, batch_size):
    """
    Sliding window sobre el volumen completo. Devuelve {régimen: pseudo-CT en HU}.
    El ensemble promedia en el espacio normalizado, como inference.py.
    """
    patch_size = tuple(cfg.TRAIN.PATCH_SIZE)
    step = [p - o for p, o in zip(patch_size, cfg.TRAIN.PATCH_STRIDE)]   # PATCH_STRIDE es el solape
    shape = pet.shape
    starts = [(z, y, x)
              for z in grid_starts(shape[0], patch_size[0], step[0])
              for y in grid_starts(shape[1], patch_size[1], step[1])
              for x in grid_starts(shape[2], patch_size[2], step[2])]

    window = hann_window(patch_size)
    acc = {r: np.zeros(shape, dtype=np.float32) for r in regimes}
    weight = np.zeros(shape, dtype=np.float32)

    topo_t = torch.from_numpy(topo).float().to(device)[None, None]
    topo_zero = torch.zeros_like(topo_t)
    need_mri = any(not REGIMES[r][0] for r in regimes)

    for b in range(0, len(starts), batch_size):
        batch_starts = starts[b:b + batch_size]
        sl = [tuple(slice(s, s + p) for s, p in zip(st, patch_size)) for st in batch_starts]
        pet_b = torch.from_numpy(np.stack([pet[s] for s in sl]))[:, None].float().to(device)
        mri_b = None
        if need_mri:
            mri_b = torch.from_numpy(np.stack([mri[(slice(None),) + s] for s in sl])).float().to(device)
        pet_n, mri_n = normalize_patches(pet_b, mri_b)
        mri_zero = torch.zeros((len(sl), 2) + patch_size, device=device)
        coords = torch.from_numpy(np.stack([
            patch_coords(st, patch_size, shape, bbox) for st in batch_starts
        ])).float().to(device)

        for r in regimes:
            mri_dropped, topo_dropped = REGIMES[r]
            mri_in = mri_zero if mri_dropped else mri_n
            topo_in = (topo_zero if topo_dropped else topo_t).expand(len(sl), -1, -1, -1)
            pred = torch.zeros_like(pet_n)
            with torch.amp.autocast(device_type="cuda", enabled=cfg.TRAIN.AMP):
                for model in models:
                    pred += model(pet_n, mri_in, topo_in, coords).float()
            pred = (pred / len(models)).squeeze(1).cpu().numpy()
            for i, s in enumerate(sl):
                acc[r][s] += pred[i] * window

        for s in sl:
            weight[s] += window

    scale = cfg.DATA.CT_MAX - cfg.DATA.CT_MIN
    return {r: acc[r] / weight * scale + cfg.DATA.CT_MIN for r in regimes}


def case_regions(mask, pet_mask):
    """Regiones fijas por sujeto (no dependen del modelo) para las métricas extra."""
    body = mask > 0
    core = ndi.binary_erosion(body, iterations=3)
    thorax = np.zeros_like(body)
    lung = lung_z_range(mask)
    if lung is not None:
        thorax[:, :, lung[0]:lung[1]] = True
    keep = upsample_mask(ndi.binary_dilation(pet_mask, iterations=2), mask.shape)
    return {"body": body, "core": core, "thorax": body & thorax, "rest": body & ~thorax,
            "outside": ~body, "keep": keep}


def case_metrics(pred_hu, ct, mask, bone_hu, regions):
    """Métricas en HU dentro del cuerpo (mask > 0). Clases: 3 aire interno, 2/4 blandos, 5/6 hueso."""
    body = regions["body"]
    err = pred_hu - ct

    def mae(sel):
        return float(np.abs(err[sel]).mean()) if sel.any() else float("nan")

    pred_bone = (pred_hu >= bone_hu) & body
    gt_bone = (ct >= bone_hu) & body
    denom = pred_bone.sum() + gt_bone.sum()
    return {
        "mae_body": mae(body),
        "me_body": float(err[body].mean()),
        "mae_air": mae(mask == 3),
        "mae_soft": mae((mask == 2) | (mask == 4)),
        "mae_bone": mae(mask >= 5),
        "dice_bone": float(2 * (pred_bone & gt_bone).sum() / denom) if denom else float("nan"),
        "mae_core": mae(regions["core"]),
        "mae_thorax": mae(regions["thorax"]),
        "mae_rest": mae(regions["rest"]),
        "ghost_frac": float((pred_hu[regions["outside"]] > GHOST_HU).mean()),
        # Post-proceso: aire fuera de la máscara del PET. Solo cambia vóxeles fuera de "keep".
        "mae_body_clamp": float(np.abs(np.where(regions["keep"], pred_hu, -1000.0) - ct)[body].mean()),
        "ghost_frac_clamp": float(((pred_hu > GHOST_HU) & regions["keep"])[regions["outside"]].mean()),
    }


def save_prediction(pred_hu, sub_id, regime, label, args):
    """Guarda el pseudo-CT; el affine se toma del NAC-PET de --ref_dir (mismo RAS que el HDF5)."""
    affine = np.eye(4)
    if args.ref_dir:
        ref = Path(args.ref_dir) / sub_id / "features" / "nacpet.nii.gz"
        if ref.is_file():
            affine = nib.as_closest_canonical(nib.load(str(ref))).affine
        else:
            logger.warning(f"Sin referencia para {sub_id}: se guarda con affine identidad.")
    out = Path(args.save_dir) / label / regime / sub_id / "ct.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred_hu.astype(np.int16), affine), str(out))


def evaluate_file(h5_path, models, fold_label, cfg, args, device, writer):
    with h5py.File(h5_path, "r") as hf:
        sub_ids = list(hf.keys())
        if args.subjects:
            sub_ids = [s for s in sub_ids if s in args.subjects]
        if args.max_subjects:
            sub_ids = sub_ids[: args.max_subjects]
        for sub_id in tqdm(sub_ids, desc=Path(h5_path).name):
            g = hf[sub_id]
            has_mri = bool(g.attrs.get("has_mri", True)) and "mri" in g
            has_topo = bool(g.attrs.get("has_topo", True)) and "topo" in g
            regimes = [r for r in args.regimes
                       if (REGIMES[r][0] or has_mri) and (REGIMES[r][1] or has_topo)]
            if not regimes:
                logger.warning(f"{sub_id}: ningún régimen pedido es posible (has_mri={has_mri}, has_topo={has_topo}).")
                continue

            pet = g["pet"][:]
            ct = g["ct"][:]
            mask = g["mask"][:]
            mri = g["mri"][:] if has_mri and any(not REGIMES[r][0] for r in regimes) else None
            topo = np.squeeze(g["topo"][:]) if has_topo else np.zeros(cfg.DATA.TOPO_SHAPE, dtype=np.float32)

            attrs_bbox = None
            if cfg.DATA.COORDS_MODE == "body":
                if "body_lo" in g.attrs:
                    attrs_bbox = (np.asarray(g.attrs["body_lo"]), np.asarray(g.attrs["body_hi"]))
                else:
                    attrs_bbox = compute_body_bbox(pet)

            pet_mask = body_mask_pet(pet)
            regions = case_regions(mask, pet_mask)
            for mode in args.bbox_modes:
                bbox = None
                if attrs_bbox is not None:
                    bbox = build_bbox(mode, attrs_bbox, mask, pet_mask, args.calib_data)
                    logger.info(f"{sub_id} bbox {mode}: lo {np.round(bbox[0]).astype(int)} "
                                f"hi {np.round(bbox[1]).astype(int)}")
                preds = predict_volume(cfg, models, pet, mri, topo, bbox, regimes, device, args.batch_size)
                for r, pred_hu in preds.items():
                    row = {"label": args.label, "expr": "+".join([args.expr] + list(args.add_expr)), "fold": fold_label,
                           "h5": Path(h5_path).name, "subject": sub_id, "regime": r, "bbox": mode}
                    row.update(case_metrics(pred_hu, ct, mask, args.bone_hu, regions))
                    writer.writerow(row)
                    logger.info(f"{args.label} {sub_id} {r} [{mode}]: MAE {row['mae_body']:.1f} HU, "
                                f"aire {row['mae_air']:.1f}, tórax {row['mae_thorax']:.1f}, "
                                f"hueso {row['mae_bone']:.1f} HU, Dice hueso {row['dice_bone']:.3f}")
                    if args.save_dir and (not args.save_subjects or sub_id in args.save_subjects):
                        label = args.label if mode == "attrs" else f"{args.label}_{mode}"
                        save_prediction(pred_hu, sub_id, r, label, args)
                del preds
            del pet, ct, mask, mri, regions, pet_mask


def print_summary(csv_path):
    """Media ± desviación por modelo, régimen y archivo (test.h5 se separa de los folds)."""
    rows = list(csv.DictReader(open(csv_path)))
    groups = {}
    for row in rows:
        split = "test" if row["h5"] == "test.h5" else "folds"
        bbox = row.get("bbox") or "attrs"
        groups.setdefault((row["label"], split, row["regime"], bbox), []).append(row)

    for metrics in (METRICS, EXTRA_METRICS):
        if metrics is EXTRA_METRICS and "mae_core" not in rows[0]:
            break
        header = (f"{'modelo':<10}{'split':<7}{'régimen':<10}{'bbox':<8}{'n':>4}"
                  + "".join(f"{m:>18}" for m in metrics))
        print("\n" + header + "\n" + "-" * len(header))
        for (label, split, regime, bbox), rs in sorted(groups.items()):
            cells = []
            for m in metrics:
                v = np.array([float(r[m]) for r in rs if r.get(m) not in (None, "")], dtype=float)
                v = v[~np.isnan(v)]
                frac = m == "dice_bone" or m.startswith("ghost")
                cells.append(f"{v.mean():>10.3f} ± {v.std():<5.3f}" if frac and v.size
                             else f"{v.mean():>10.1f} ± {v.std():<5.1f}" if v.size else f"{'-':>18}")
            print(f"{label:<10}{split:<7}{regime:<10}{bbox:<8}{len(rs):>4}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser("Evaluación volumétrica por régimen de modalidades")
    parser.add_argument("--label", required=True, help="Nombre corto del modelo en el CSV (ej. v12, v18)")
    parser.add_argument("--config_path", required=True, help="YAML con el que se entrenó el modelo")
    parser.add_argument("--checkpoints_dir", required=True, help="Carpeta con <expr>_Fold_k/Best_training_state.pkl")
    parser.add_argument("--expr", required=True, help="Nombre base del experimento (sin _Fold_k)")
    parser.add_argument("--h5_dir", default=None, help="Carpeta con fold_*.h5 (y test.h5)")
    parser.add_argument("--h5", nargs="*", default=None, help="Archivos HDF5 concretos (en vez de --h5_dir)")
    parser.add_argument("--add_expr", nargs="*", default=[],
                        help="Experimentos extra a promediar con --expr (mismo --checkpoints_dir)")
    parser.add_argument("--oof", action="store_true",
                        help="Modelo del fold k sobre fold_k.h5 (BigMAC). Sin esto: ensemble sobre todos los HDF5.")
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--regimes", nargs="*", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bone_hu", type=float, default=300.0, help="Umbral de hueso para el Dice")
    parser.add_argument("--max_subjects", type=int, default=None, help="Solo los N primeros sujetos por archivo (prueba)")
    parser.add_argument("--save_dir", default=None, help="Opcional: guardar los pseudo-CT en NIfTI")
    parser.add_argument("--save_subjects", nargs="*", default=None,
                        help="Con --save_dir: guardar solo estos sujetos (ej. para las figuras)")
    parser.add_argument("--ref_dir", default=None, help="Carpeta sub-*/features/nacpet.nii.gz para el affine al guardar")
    parser.add_argument("--subjects", nargs="*", default=None, help="Evaluar solo estos sujetos")
    parser.add_argument("--bbox_modes", nargs="*", default=["attrs"], choices=BBOX_MODES,
                        help="Bboxes para las coordenadas body (diagnóstico B2b, ver utils/body_coords.py)")
    parser.add_argument("--calib", default=None,
                        help="JSON de utils/body_coords.py (necesario si --bbox_modes no es solo attrs)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.calib_data = None
    if set(args.bbox_modes) != {"attrs"}:
        if not args.calib:
            parser.error("--bbox_modes distinto de attrs necesita --calib")
        args.calib_data = json.loads(Path(args.calib).read_text())

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    device = torch.device(args.device)
    logger.info(f"{args.label}: COORDS_MODE={cfg.DATA.COORDS_MODE}, regímenes={args.regimes}")

    ckpt = lambda k, expr=args.expr: Path(args.checkpoints_dir) / f"{expr}_Fold_{k}" / "Best_training_state.pkl"
    exprs = [args.expr] + list(args.add_expr)
    if args.add_expr:
        logger.info(f"Ensemble entre experimentos: {exprs}")

    if args.h5:
        h5_files = [Path(p) for p in args.h5]
    elif args.h5_dir:
        h5_files = sorted(Path(args.h5_dir).glob("*.h5"))
    else:
        parser.error("Indica --h5_dir o --h5")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    new_file = not os.path.isfile(args.out_csv) or os.path.getsize(args.out_csv) == 0
    if not new_file:
        with open(args.out_csv) as f:
            header = f.readline().strip().split(",")
        if header != FIELDS:
            parser.error(f"{args.out_csv} tiene otras columnas (versión anterior del script): "
                         "usa otro --out_csv o renómbralo.")
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        if args.oof:
            for k in args.folds:
                h5_path = next((p for p in h5_files if p.name == f"fold_{k}.h5"), None)
                if h5_path is None:
                    logger.warning(f"No existe fold_{k}.h5: se omite el fold {k}.")
                    continue
                models = [load_model(cfg, ckpt(k, e), device) for e in exprs]
                evaluate_file(h5_path, models, str(k), cfg, args, device, writer)
                f.flush()
                del models
                torch.cuda.empty_cache()
        else:
            models = [load_model(cfg, ckpt(k, e), device) for e in exprs for k in args.folds]
            for h5_path in h5_files:
                evaluate_file(h5_path, models, "ens" + "".join(map(str, args.folds)), cfg, args, device, writer)
                f.flush()

    print_summary(args.out_csv)


if __name__ == "__main__":
    main()
