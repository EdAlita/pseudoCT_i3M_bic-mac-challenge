"""
utils/feature_embedding.py

¿Acerca la augmentation de dominio del NAC-PET (v23) las representaciones internas de BIC-MAC
y Utrecht? Para cada modelo (por defecto v21 y v23, mismo fold) se extraen los rasgos del
cuello de botella (tras el FiLM, 192 canales, media espacial) de parches tomados en:

    BIC-MAC        NAC-PET de adultos tal cual
    BIC-MAC + aug  el mismo parche con la augmentation de dominio (parámetros a mitad de rango)
    Utrecht       exámenes de validación del fold (el modelo no los vio)

Cada parche se centra en un punto al azar del cuerpo; los tres grupos usan las mismas
reglas. Salida: <out_dir>/cache_features.npz (rasgos, cohorte, modelo) y, con
make_presentation_figures --only 32, la figura de t-SNE y la separabilidad de cohortes.

Ejemplo:
    python -m utils.feature_embedding --fold 0 --n_patches 300

Owner: Edwing Ulin
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from config.defaults import get_cfg_defaults
from data.preprocessing import patch_coords
from utils.evaluate_models import load_model, normalize_patches
from utils.make_presentation_figures import domain_aug_mid

CKPT = "/mnt/d/Data/007_pseudoCT/training/experiments/checkpoints"
MODELS = {"v21": ("config/mutimodal_v21_utrecht_ft.yaml", "Multimodal_pCT_v21_utrecht_ft"),
          "v23": ("config/mutimodal_v23_utrecht_ft.yaml", "Multimodal_pCT_v23_utrecht_ft")}


def sample_patches(h5_path, n, patch, rng):
    """n parches (pet, start, shape, bbox) centrados en vóxeles del cuerpo, repartidos entre sujetos."""
    out = []
    with h5py.File(h5_path, "r") as hf:
        subs = list(hf)
        for i in range(n):
            g = hf[subs[i % len(subs)]]
            shape = g["pet"].shape
            m = g["mask"][::4, ::4, ::4] > 0
            idx = np.argwhere(m)
            c = idx[rng.integers(len(idx))] * 4
            start = [int(np.clip(c[a] - patch[a] // 2, 0, shape[a] - patch[a])) for a in range(3)]
            sl = tuple(slice(s, s + p) for s, p in zip(start, patch))
            pet = g["pet"][sl].astype(np.float32)
            body = g["mask"][sl] > 0
            bbox = (np.asarray(g.attrs["body_lo"]), np.asarray(g.attrs["body_hi"]))
            out.append((pet, body, start, shape, bbox))
    return out


@torch.no_grad()
def features(model, patches, patch, device, aug=False):
    feats = []
    hook_out = {}
    h = model.coord_cond.register_forward_hook(lambda mod, inp, out: hook_out.__setitem__("x", out))
    for pet, body, start, shape, bbox in patches:
        if aug and body.any() and not body.all():
            pet = domain_aug_mid(pet, body)
        p = torch.from_numpy(pet)[None, None].to(device)
        p, _ = normalize_patches(p, None)
        mri = torch.zeros((1, 2) + tuple(patch), device=device)
        topo = torch.zeros((1, 1, 512, 531), device=device)
        coords = torch.from_numpy(patch_coords(start, patch, shape, bbox))[None].to(device)
        with torch.amp.autocast(device_type="cuda"):
            model(p, mri, topo, coords)
        feats.append(hook_out["x"].float().mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy())
    h.remove()
    return np.stack(feats)


def main():
    ap = argparse.ArgumentParser("Rasgos del cuello de botella: BIC-MAC vs Utrecht")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_patches", type=int, default=300)
    ap.add_argument("--models", nargs="*", default=["v21", "v23"])
    ap.add_argument("--bigmac_h5_dir", default="/mnt/c/Users/edyA0/Proyectos/data_multimask")
    ap.add_argument("--utrecht_h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    ap.add_argument("--out_dir", default="/mnt/d/Data/007_pseudoCT/eval/figures_final")
    args = ap.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    patch = (64, 128, 128)
    big = sample_patches(Path(args.bigmac_h5_dir) / f"fold_{args.fold}.h5", args.n_patches, patch, rng)
    utr = sample_patches(Path(args.utrecht_h5_dir) / f"fold_{args.fold}.h5", args.n_patches, patch, rng)
    out = {}
    for name in args.models:
        cfg_path, expr = MODELS[name]
        cfg = get_cfg_defaults()
        cfg.merge_from_file(cfg_path)
        model = load_model(cfg, Path(CKPT) / f"{expr}_Fold_{args.fold}" / "Best_training_state.pkl", device)
        out[f"{name}|BIC-MAC"] = features(model, big, patch, device)
        out[f"{name}|BIC-MAC + aug"] = features(model, big, patch, device, aug=True)
        out[f"{name}|Utrecht"] = features(model, utr, patch, device)
        print(f"{name}: {len(big)} + {len(big)} + {len(utr)} patches", flush=True)
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(Path(args.out_dir) / "cache_features.npz", **out)
    print("terminado")


if __name__ == "__main__":
    main()
