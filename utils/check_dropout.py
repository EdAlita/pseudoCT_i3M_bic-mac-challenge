"""
utils/check_dropout.py

Comprueba en el dataset real que el MRI y el topograma se anulan como dice el YAML:
fracción de parches con mri_dropped / topo_dropped y que el tensor sea cero
exactamente cuando el flag está activo (y no cero cuando no lo está).

Ejemplo:
    python -m utils.check_dropout --config_path config/mutimodal_v19_pet_only.yaml \
        --h5 /mnt/c/Users/edyA0/Proyectos/data_multimask/fold_0.h5 --mode val

Owner: Edwing Ulin
"""

import argparse

from config.defaults import get_cfg_defaults
from data.dataset import MultimodalpCTDataset


def main():
    parser = argparse.ArgumentParser("Comprobación de MRI / TOPO dropout")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--h5", required=True, nargs="+")
    parser.add_argument("--mode", default="train", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=100, help="Parches a muestrear")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    ds = MultimodalpCTDataset(args.h5, cfg, mode=args.mode)

    n = min(args.n, len(ds))
    topo_dropped = mri_dropped = errors = 0
    for i in range(n):
        d = ds[i]
        t, m = bool(d["topo_dropped"]), bool(d["mri_dropped"])
        topo_dropped += t
        mri_dropped += m
        topo_zero = d["topo"].abs().sum() == 0
        errors += (t != topo_zero) + (m and d["mri"].abs().sum() > 0)

    p_topo = cfg.DATA.TOPO_DROPOUT_PROB if args.mode == "train" else cfg.DATA.TOPO_DROPOUT_PROB_VAL
    p_mri = cfg.DATA.MRI_DROPOUT_PROB if args.mode == "train" else cfg.DATA.MRI_DROPOUT_PROB_VAL
    print(f"{n} parches ({args.mode})")
    print(f"  topo_dropped: {topo_dropped / n:.2f}  (YAML {p_topo}; 1.0 si el sujeto no tiene topograma)")
    print(f"  mri_dropped:  {mri_dropped / n:.2f}  (YAML {p_mri}; 1.0 si el sujeto no tiene MRI)")
    print(f"  inconsistencias flag/tensor: {errors}")


if __name__ == "__main__":
    main()
