"""
qa_utrecht.py

Control visual de los HDF5 de Utrecht: un PNG por sujeto con un corte coronal
del CT, el NAC-PET superpuesto con el bbox corporal, y la máscara de pesos.
Imprime además el % de vóxeles de hueso (clases 5/6) para ajustar umbrales.

Uso:
    python -m utils.qa_utrecht --hdf5_dir <utrecht_hdf5> --out_dir <carpeta_png>

Owner: Edwing Ulin
Version: v1.0.0
"""

import argparse
import glob
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser("QA visual de los HDF5 de Utrecht")
    parser.add_argument("--hdf5_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for path in sorted(glob.glob(f"{args.hdf5_dir}/*.h5")):
        with h5py.File(path, "r") as hf:
            for sub_id, g in hf.items():
                lo, hi = g.attrs["body_lo"], g.attrs["body_hi"]
                y = int((lo[1] + hi[1]) // 2)                 # corte coronal por el centro del cuerpo
                ct, pet, mask = g["ct"][:, y, :], g["pet"][:, y, :], g["mask"][:, y, :]

                fig, ax = plt.subplots(1, 3, figsize=(15, 6))
                ax[0].imshow(ct.T, cmap="gray", vmin=-1000, vmax=1500, origin="lower")
                ax[0].set_title(f"{sub_id} CT (y={y})")
                ax[1].imshow(ct.T, cmap="gray", vmin=-1000, vmax=1500, origin="lower")
                ax[1].imshow(np.ma.masked_less_equal(pet.T, np.percentile(pet, 90)),
                             cmap="hot", alpha=0.5, origin="lower")
                ax[1].add_patch(plt.Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2],
                                              fill=False, ec="cyan"))
                ax[1].set_title("NAC-PET sobre CT + bbox")
                ax[2].imshow(mask.T, cmap="tab10", vmin=0, vmax=9, origin="lower")
                ax[2].set_title("Máscara (2 cuerpo / 3 aire / 5-6 hueso)")
                for a in ax:
                    a.axis("off")
                fig.savefig(f"{args.out_dir}/{sub_id}.png", dpi=80, bbox_inches="tight")
                plt.close(fig)

                counts = np.bincount(g["mask"][:].ravel(), minlength=7)
                body = counts[[2, 3, 5, 6]].sum()
                bone_pct = 100 * counts[[5, 6]].sum() / counts.sum()
                air_ratio = counts[3] / max(body, 1)
                # Valores normales en Utrecht: cuerpo 2-7 %, aire interno / cuerpo 0.02-0.07.
                # Un ratio alto indica que la camilla (rellena de "aire") entró en el cuerpo.
                flag = "  <-- REVISAR" if body == 0 or air_ratio > 0.15 else ""
                print(f"{Path(path).name} {sub_id}: cuerpo = {100 * body / counts.sum():.2f}% | "
                      f"aire/cuerpo = {air_ratio:.3f} | hueso 5/6 = {bone_pct:.2f}% | "
                      f"bbox lo={lo.tolist()} hi={hi.tolist()}{flag}")
