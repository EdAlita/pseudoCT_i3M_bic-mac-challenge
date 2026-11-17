"""
utils/analyze_cohorts.py

Análisis de las cohortes (tarea 1 tras la reunión con UMC): distribución por edad y
tamaño corporal de Utrecht (pediátrica) frente a BigMAC (adultos), medido en las
máscaras de los HDF5 para que ambas cohortes se midan igual. Los DICOM de Utrecht están
anonimizados (sin peso ni altura), así que el tamaño sale de la imagen:

    height_cm        extensión del cuerpo en z (solo Utrecht, cuerpo entero; BigMAC es torso)
    lung_len_cm      longitud de la losa pulmonar (cortes con aire interno > 25 % del máximo)
    chest_width_cm   ancho del cuerpo en el corte central de la losa pulmonar (eje x)
    chest_ap_cm      profundidad antero-posterior en ese corte (eje y)
    lung_vol_l       volumen de aire interno (clase 3) dentro de la losa pulmonar
    body_vol_l       volumen del cuerpo dentro de la losa pulmonar

Salidas: <out_dir>/cohortes.csv, fig13_cohortes_edad.png, fig14_cohortes_tamano.png,
y una tabla resumen por consola (también en cohortes.md).

Ejemplo:
    python -m utils.analyze_cohorts --out_dir /mnt/d/Data/007_pseudoCT/eval/figuras

Owner: Edwing Ulin
"""

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np

from utils.body_coords import lung_z_range

VOXEL_MM = (1.5234375, 1.5234375, 2.0)   # malla de BigMAC (también la de Utrecht)
MIN_COUNT = 50                           # vóxeles para que un corte cuente como cuerpo


def extent(m, axis):
    """(lo, hi) de los índices con más de MIN_COUNT vóxeles en el eje dado."""
    counts = m.sum(axis=tuple(a for a in range(m.ndim) if a != axis))
    idx = np.where(counts > (MIN_COUNT if m.ndim == 3 else 0))[0]
    return (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, 0)


def measure(mask):
    """Medidas anatómicas de un sujeto a partir de su máscara (clases: 0 fuera, 3 aire interno)."""
    body = mask > 0
    z0, z1 = extent(body, 2)
    out = {"height_cm": (z1 - z0) * VOXEL_MM[2] / 10}
    lung = lung_z_range(mask, contiguous=True)
    if lung is None:
        return out
    zl0, zl1 = lung
    zm = (zl0 + zl1) // 2
    sl = body[:, :, zm]
    x0, x1 = extent(sl, 0)
    y0, y1 = extent(sl, 1)
    vox_l = np.prod(VOXEL_MM) / 1e6
    out.update({
        "lung_len_cm": (zl1 - zl0) * VOXEL_MM[2] / 10,
        "chest_width_cm": (x1 - x0) * VOXEL_MM[0] / 10,
        "chest_ap_cm": (y1 - y0) * VOXEL_MM[1] / 10,
        "lung_vol_l": float((mask[:, :, zl0:zl1] == 3).sum() * vox_l),
        "body_vol_l": float(body[:, :, zl0:zl1].sum() * vox_l),
    })
    return out


def utrecht_meta(path):
    meta = {}
    for r in csv.DictReader(open(path)):
        m = re.match(r"(\d+)Y", r.get("age", "") or "")
        meta[r["sub_id"]] = {"age": int(m.group(1)) if m else np.nan, "sex": r.get("sex", "")}
    return meta


def bigmac_meta(raw_dir):
    meta = {}
    for p in Path(raw_dir).glob("*/sub-*/features/metadata.json"):
        d = json.loads(p.read_text())
        meta[p.parts[-3]] = {"age": d.get("age", np.nan), "sex": d.get("sex", ""),
                             "height_true_cm": 100 * d["height"] if d.get("height") else np.nan,
                             "weight_kg": d.get("weight", np.nan)}
    return meta


def collect(args):
    rows = []
    cohorts = [("Utrecht", args.utrecht_h5_dir, utrecht_meta(args.participants)),
               ("BigMAC", args.bigmac_h5_dir, bigmac_meta(args.bigmac_raw_dir))]
    for cohort, h5_dir, meta in cohorts:
        for f in sorted(Path(h5_dir).glob("*.h5")):
            with h5py.File(f, "r") as hf:
                for sub_id in hf:
                    row = {"cohort": cohort, "subject": sub_id, "split": f.stem,
                           **meta.get(sub_id, {}), **measure(hf[sub_id]["mask"][:])}
                    if cohort == "BigMAC":
                        row["height_cm"] = np.nan   # FOV de torso: la extensión en z no es la altura
                    rows.append(row)
                    print(f"{cohort:8s} {sub_id:26s} edad {row.get('age')}: "
                          f"tórax {row.get('chest_width_cm', np.nan):.1f} x {row.get('chest_ap_cm', np.nan):.1f} cm, "
                          f"pulmón {row.get('lung_vol_l', np.nan):.2f} l", flush=True)
    return rows


def summary(rows):
    keys = ["age", "height_cm", "chest_width_cm", "chest_ap_cm", "lung_len_cm", "lung_vol_l", "body_vol_l"]
    groups = [("BigMAC (adultos)", lambda r: r["cohort"] == "BigMAC"),
              ("Utrecht < 13 años", lambda r: r["cohort"] == "Utrecht" and r["age"] < 13),
              ("Utrecht ≥ 13 años", lambda r: r["cohort"] == "Utrecht" and r["age"] >= 13),
              ("Utrecht todos", lambda r: r["cohort"] == "Utrecht")]
    lines = ["| Grupo | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    stats = {}
    for name, f in groups:
        rs = [r for r in rows if f(r)]
        cells = []
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            stats[(name, k)] = v
            cells.append(f"{np.median(v):.1f} [{v.min():.1f}–{v.max():.1f}]" if v.size else "-")
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    # Factor de escala niño/adulto por medida (mediana Utrecht / mediana BigMAC): guía para
    # la augmentación de escala (tarea 2) y la deformación hacia anatomía pediátrica (tarea 3).
    lines.append("\nFactor de escala (mediana del grupo / mediana BigMAC):\n")
    lines.append("| Grupo | ancho tórax | AP tórax | longitud pulmón | volumen pulmón^(1/3) |")
    lines.append("|---|---|---|---|---|")
    for name, _ in groups[1:]:
        ratio = lambda k, p=1.0: (np.median(stats[(name, k)]) / np.median(stats[("BigMAC (adultos)", k)])) ** p
        lines.append(f"| {name} | {ratio('chest_width_cm'):.2f} | {ratio('chest_ap_cm'):.2f} | "
                     f"{ratio('lung_len_cm'):.2f} | {ratio('lung_vol_l', 1 / 3):.2f} |")
    return "\n".join(lines)


def figures(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.make_figures import GRID, INK2, SERIES, SURFACE, save  # mismo estilo y paleta

    utr = [r for r in rows if r["cohort"] == "Utrecht"]
    big = [r for r in rows if r["cohort"] == "BigMAC"]
    col_test, col_folds, col_big = SERIES[0], SERIES[2], SERIES[6]

    # fig13: edad en Utrecht por conjunto (test / folds) y por sexo.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.arange(0, 22, 1)
    for ax, (key, groups) in zip(axes, [
            ("split", [("folds", "Folds (entrenamiento / validación)", col_folds,
                        lambda r: r["split"] != "test"),
                       ("test", "test.h5 (nunca visto)", col_test, lambda r: r["split"] == "test")]),
            ("sex", [("F", "Femenino", SERIES[4], lambda r: r["sex"] == "F"),
                     ("M", "Masculino", SERIES[0], lambda r: r["sex"] == "M")])]):
        bottom = np.zeros(len(bins) - 1)
        for _, name, col, f in groups:
            h, _ = np.histogram([r["age"] for r in utr if f(r)], bins)
            ax.bar(bins[:-1] + 0.5, h, width=0.9, bottom=bottom, color=col, edgecolor=SURFACE, lw=1.5,
                   label=f"{name} (n = {int(h.sum())})")
            bottom += h
        ax.axvline(12.5, color=INK2, ls="--", lw=1)
        ax.annotate("13 años", (12.6, 0.55 * bottom.max()), fontsize=8.5, color=INK2)
        ax.set_xlabel("Edad (años)")
        ax.set_ylabel("Exámenes")
        ax.set_xticks(range(0, 22, 2))
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    axes[0].set_title("Por conjunto", loc="left")
    axes[1].set_title("Por sexo", loc="left")
    n_young = sum(r["age"] < 13 for r in utr)
    fig.suptitle(f"Utrecht: {len(utr)} exámenes de {len({r['subject'][:14] for r in utr})} pacientes, "
                 f"{min(r['age'] for r in utr)}–{max(r['age'] for r in utr)} años; "
                 f"{n_young} menores de 13 (BigMAC: {len(big)} adultos, "
                 f"{int(np.nanmin([r['age'] for r in big]))}–{int(np.nanmax([r['age'] for r in big]))} años)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "fig13_cohortes_edad.png")

    # fig14: tamaño corporal frente a la edad; BigMAC como banda de referencia (mediana y rango).
    metrics = [("chest_width_cm", "Ancho del tórax (cm)"), ("chest_ap_cm", "Profundidad AP del tórax (cm)"),
               ("lung_vol_l", "Volumen pulmonar (l)"), ("height_cm", "Altura estimada (cm, solo Utrecht)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.2))
    for ax, (k, name) in zip(axes, metrics):
        bv = np.array([r.get(k, np.nan) for r in big], dtype=float)
        bv = bv[~np.isnan(bv)]
        if bv.size:
            ax.axhspan(bv.min(), bv.max(), color=col_big, alpha=0.12, lw=0)
            ax.axhline(np.median(bv), color=col_big, lw=2)
            ax.annotate(f"BigMAC (adultos): mediana {np.median(bv):.1f}", (0.5, np.median(bv)),
                        xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
        for split, col in (("folds", col_folds), ("test", col_test)):
            rs = [r for r in utr if (r["split"] == "test") == (split == "test") and not np.isnan(r.get(k, np.nan))]
            ax.scatter([r["age"] for r in rs], [r[k] for r in rs], s=40, color=col, edgecolor=SURFACE,
                       lw=1.2, zorder=3, label="test.h5" if split == "test" else "folds")
        ax.set_xlabel("Edad (años)")
        ax.set_title(name, loc="left", fontsize=10)
        ax.axvline(12.5, color=GRID, ls="--", lw=1)
    axes[0].legend(loc="lower right")
    fig.suptitle("Tamaño corporal medido en la máscara del CT: niños de Utrecht frente a adultos de BigMAC "
                 "(banda = rango de BigMAC)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "fig14_cohortes_tamano.png")


def main():
    parser = argparse.ArgumentParser("Análisis de cohortes: edad y tamaño corporal")
    parser.add_argument("--utrecht_h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    parser.add_argument("--bigmac_h5_dir", default="/mnt/c/Users/edyA0/Proyectos/data_multimask")
    parser.add_argument("--participants", default="/mnt/d/Data/007_pseudoCT/utrecht_bigmac_space/participants.csv")
    parser.add_argument("--bigmac_raw_dir", default="/mnt/d/bic-mac-data", help="Con */sub-*/features/metadata.json")
    parser.add_argument("--out_dir", default="/mnt/d/Data/007_pseudoCT/eval/figuras")
    parser.add_argument("--from_csv", action="store_true", help="Reusar cohortes.csv (solo rehacer figuras y tabla)")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "cohortes.csv"

    if args.from_csv:
        rows = []
        for r in csv.DictReader(open(csv_path)):
            rows.append({k: (v if k in ("cohort", "subject", "split", "sex") else float(v) if v not in ("", "nan") else np.nan)
                         for k, v in r.items()})
    else:
        rows = collect(args)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("cohort", "subject", "split"), k))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    text = summary(rows)
    print("\n" + text)
    (out / "cohortes.md").write_text("# Cohortes: edad y tamaño corporal\n\n" + text + "\n")
    figures(rows, out)


if __name__ == "__main__":
    main()
