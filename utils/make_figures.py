"""
utils/make_figures.py

Figuras y tablas para el resumen de la Fase A (Utrecht). Cada figura se omite si
faltan sus datos, así que se puede correr en cualquier momento:

    fig1_<expr>_curvas_val.png       MAE de validación por época y fold (logs de train)
    fig2_bigmac_oof_regimenes.png    A6: MAE cuerpo / MAE hueso / Dice hueso por régimen y modelo
    fig3_bigmac_oof_pareado.png      A6: cambio por caso en "solo PET" entre modelos
    fig4_utrecht_zero_shot.png       Utrecht: métricas por modelo (solo PET)
    fig5_utrecht_mae_vs_edad.png     Utrecht: MAE por examen frente a la edad
    fig6_utrecht_<sub>.png           Utrecht: cortes coronales PET / CT / pseudo-CT / error
    fig7_bigmac_<sub>.png            BigMAC: cortes coronales con y sin MRI
    fig8_utrecht_B4_pareado_<m>.png  B4: v18 zero-shot -> modelo con fine-tuning (v20, v21...) por examen
    fig9_utrecht_B4_<sub>.png        B4: cortes coronales v18 frente a los modelos con fine-tuning
    fig10_utrecht_B4_resumen.png     B4: media ± desviación por modelo, folds (out-of-fold) y test
    fig11_utrecht_curvas_val.png     Curvas de val en Utrecht de v20, v21... superpuestas por fold
    fig12_utrecht_B4_edad.png        B4: MAE cuerpo y aire interno por examen según la edad, todos los modelos
    tablas.md                        Media ± desviación de todas las métricas

Ejemplo:
    python -m utils.make_figures --out_dir /mnt/d/Data/007_pseudoCT/eval/figuras

Owner: Edwing Ulin
"""

import argparse
import csv
import glob
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

EVAL = "/mnt/d/Data/007_pseudoCT/eval"

# Paleta categórica validada (orden fijo; el color sigue al modelo, no a su posición).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_ORDER = ["v12", "v18", "v19", "v20", "v21", "v22", "v23", "v24"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
# Rampa secuencial de un solo tono (azul) para los mapas de error.
ERR_CMAP = LinearSegmentedColormap.from_list(
    "err_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

REGIME_ORDER = ["full", "no_topo", "no_mri", "pet_only"]
REGIME_NAME = {"full": "PET + MRI + topo", "no_topo": "PET + MRI",
               "no_mri": "PET + topo", "pet_only": "Solo PET"}
METRIC_NAME = {"mae_body": "MAE cuerpo (HU)", "mae_bone": "MAE hueso (HU)", "dice_bone": "Dice hueso (>300 HU)",
               "mae_soft": "MAE tejido blando (HU)", "mae_air": "MAE aire interno (HU)", "me_body": "Sesgo cuerpo (HU)",
               "mae_thorax": "MAE tórax (HU)", "ghost_frac": "Fantasmas fuera del cuerpo"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def model_color(label, labels):
    """Color fijo por modelo: v12/v18/v19 ocupan siempre las mismas posiciones."""
    order = MODEL_ORDER + sorted(l for l in labels if l not in MODEL_ORDER)
    return SERIES[order.index(label) % len(SERIES)]


def sort_models(labels):
    return sorted(labels, key=lambda l: (MODEL_ORDER.index(l) if l in MODEL_ORDER else 99, l))


def read_csv(path):
    if not path or not Path(path).is_file():
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k not in ("label", "expr", "fold", "h5", "subject", "regime") else v
            except ValueError:
                pass
    # Si un modelo se evaluó dos veces, la última fila de cada (modelo, sujeto, régimen) manda.
    dedup = {(r["label"], r["subject"], r["regime"]): r for r in rows}
    return list(dedup.values())


def save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ----------------------------------------------------------------------------- fig 1
def parse_train_logs(paths):
    """
    {expr_fold: {época: {"con_mri": x, "sin_mri": y}}} a partir de logs/train_*.log.
    Maneja reanudaciones: un "--resume" con FINE_TUNE antiguo reiniciaba la época a 0,
    así que se desplaza por la época original del checkpoint.
    """
    curves = {}
    for path in sorted(paths):
        expr = resume_expr = None
        offset = 0
        epoch = None
        for line in open(path, errors="ignore"):
            if line.startswith("EXPR_NUM:"):
                expr, resume_expr, offset = line.split(":", 1)[1].strip(), None, 0
                curves.setdefault(expr, {})
            elif "RESUME_EXPR_NUM:" in line:
                resume_expr = line.split(":", 1)[1].strip()
            elif "Fine-tuning desde pesos" in line and expr:
                orig = int(re.search(r"original (\d+)", line).group(1))
                offset = orig if resume_expr == expr else 0
                curves[expr] = {e: v for e, v in curves[expr].items() if e <= offset}
            elif "Resuming from epoch" in line and expr and offset == 0 and resume_expr == expr:
                start = int(re.search(r"epoch (\d+)", line).group(1))
                curves[expr] = {e: v for e, v in curves[expr].items() if e <= start}
            elif "Evaluating model at epoch" in line:
                epoch = int(re.search(r"epoch (\d+)", line).group(1)) + offset
            elif "Val MAE " in line and expr and epoch is not None:
                m = re.search(r"Val MAE (con_mri|sin_mri): (\d+\.\d+)", line)
                if m:
                    curves[expr].setdefault(epoch, {})[m.group(1)] = float(m.group(2))
    return {k: v for k, v in curves.items() if v}


def v12_best(ckpt_dir, fold):
    """Mejor MAE de validación de v12 (con MRI) guardado en su checkpoint, como referencia."""
    p = Path(ckpt_dir) / f"Multimodal_pCT_v12_torso_Fold_{fold}" / "Best_training_state.pkl"
    if not p.is_file():
        return None
    import torch
    return torch.load(str(p), map_location="cpu", weights_only=False).get("best_metric")


def fig_training_curves(args):
    curves = parse_train_logs(glob.glob(args.train_logs))
    bases = sorted({re.sub(r"_Fold_\d+$", "", e) for e in curves if "smoke" not in e})
    for base in bases:
        folds = sorted(int(e.rsplit("_", 1)[1]) for e in curves if e.startswith(base + "_Fold_"))
        fig, axes = plt.subplots(1, len(folds), figsize=(3.2 * len(folds), 3.3), sharey=True, squeeze=False)
        for ax, k in zip(axes[0], folds):
            c = curves[f"{base}_Fold_{k}"]
            epochs = sorted(c)
            for j, (key, name) in enumerate([("con_mri", "Con MRI"), ("sin_mri", "Sin MRI")]):
                pts = [(e, c[e][key]) for e in epochs if key in c[e]]
                if not pts:
                    continue
                x, y = zip(*pts)
                ax.plot(x, y, color=SERIES[j], lw=2, label=name)
                b = int(np.argmin(y))
                ax.plot(x[b], y[b], "o", ms=8, color=SERIES[j], mec=SURFACE, mew=2, zorder=3)
                ax.annotate(f"{y[b]:.1f}", (x[b], y[b]), textcoords="offset points", xytext=(0, 7 if j else -14),
                            ha="center", fontsize=8, color=INK2,
                            bbox=dict(fc=SURFACE, ec="none", pad=0.5, alpha=0.85))
            # La referencia de v12 es su val en BigMAC con MRI: no aplica a la val en Utrecht.
            ref = None if "utrecht" in base else v12_best(args.v12_ckpt_dir, k)
            if ref:
                ax.axhline(ref, color=INK2, lw=1, ls="--")
                ax.annotate(f"v12 con MRI {ref:.1f}", (ax.get_xlim()[0], ref), textcoords="offset points",
                            xytext=(4, 3), ha="left", fontsize=7.5, color=INK2)
            ax.set_title(f"Fold {k}")
            ax.set_xlabel("Época")
        axes[0][0].set_ylabel("MAE validación (HU)")
        axes[0][0].legend(loc="upper right")
        fig.suptitle(f"{base}: MAE de validación (parches) — el punto marca el mínimo", x=0.01, ha="left",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        save(fig, args.out_dir, f"fig1_{base}_curvas_val.png")
    fig_utrecht_curves(curves, args)


def fig_utrecht_curves(curves, args):
    """fig11: curvas de val (Utrecht fold k) de todos los fine-tuning de Utrecht superpuestas."""
    exprs = sorted({re.sub(r"_Fold_\d+$", "", e) for e in curves if "utrecht" in e and "smoke" not in e})
    if len(exprs) < 2:
        return
    label = lambda e: re.search(r"_(v\d+)_", e).group(1)
    labels = [label(e) for e in exprs]
    folds = sorted({int(e.rsplit("_", 1)[1]) for e in curves if any(e.startswith(b + "_Fold_") for b in exprs)})
    fig, axes = plt.subplots(1, len(folds), figsize=(3.2 * len(folds), 3.4), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], folds):
        for e in exprs:
            c = curves.get(f"{e}_Fold_{k}", {})
            pts = [(ep, c[ep]["sin_mri"]) for ep in sorted(c) if "sin_mri" in c[ep]]
            if not pts:
                continue
            x, y = zip(*pts)
            col = model_color(label(e), labels)
            ax.plot(x, y, color=col, lw=2)
            b = int(np.argmin(y))
            ax.plot(x[b], y[b], "o", ms=8, color=col, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(f"{y[b]:.1f}", (x[b], y[b]), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8, color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=0.5, alpha=0.85))
        ax.set_title(f"Fold {k}")
        ax.set_xlabel("Época")
    axes[0][0].set_ylabel("MAE validación (HU)")
    fig.suptitle("Fine-tuning en Utrecht: MAE de validación (parches de Utrecht fold k, solo PET); "
                 "el punto marca el mínimo", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    model_legend(fig, labels)
    save(fig, args.out_dir, "fig11_utrecht_curvas_val.png")


# ----------------------------------------------------------------------------- fig 2 / 4
def grouped_bars(ax, rows, metric, groups, group_key, labels, group_names):
    """Barras agrupadas (media ± desviación) con los casos individuales como puntos."""
    n = len(labels)
    width = 0.8 / n
    rng = np.random.default_rng(0)
    for i, lab in enumerate(labels):
        col = model_color(lab, labels)
        for g_i, g in enumerate(groups):
            v = np.array([r[metric] for r in rows if r["label"] == lab and r[group_key] == g], dtype=float)
            v = v[~np.isnan(v)]
            if not v.size:
                continue
            x = g_i + (i - (n - 1) / 2) * width
            ax.bar(x, v.mean(), width * 0.92, color=col, alpha=0.85, edgecolor=SURFACE, linewidth=2)
            ax.errorbar(x, v.mean(), yerr=v.std(), color=INK2, lw=1, capsize=2.5)
            ax.scatter(x + rng.uniform(-width * 0.25, width * 0.25, v.size), v, s=9, color=INK, alpha=0.35,
                       lw=0, zorder=3)
            fmt = f"{v.mean():.2f}" if metric == "dice_bone" else f"{v.mean():.0f}"
            ax.annotate(fmt, (x, v.mean() + v.std()), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([group_names.get(g, g) for g in groups])
    ax.set_title(METRIC_NAME[metric])
    ax.grid(axis="x", visible=False)


def model_legend(fig, labels):
    """Leyenda común de modelos en una fila, entre el título y los paneles."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_color(l, labels), alpha=0.85) for l in labels]
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.995, 0.955),
               title="Modelo", title_fontsize=9, fontsize=9, alignment="right")


def fig_bigmac_regimes(rows, args):
    if not rows:
        return
    labels = sort_models({r["label"] for r in rows})
    regimes = [g for g in REGIME_ORDER if any(r["regime"] == g for r in rows)]
    metrics = ["mae_body", "mae_bone", "dice_bone"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m in zip(axes, metrics):
        grouped_bars(ax, rows, m, regimes, "regime", labels, REGIME_NAME)
    n_cases = len({r["subject"] for r in rows})
    fig.suptitle(f"BigMAC out-of-fold (modelo del fold k sobre fold k, n = {n_cases} casos): "
                 "media ± desviación, puntos = casos", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    model_legend(fig, labels)
    save(fig, args.out_dir, "fig2_bigmac_oof_regimenes.png")


def fig_bigmac_paired(rows, args, regime="pet_only", metric="mae_body"):
    sel = [r for r in rows if r["regime"] == regime]
    labels = sort_models({r["label"] for r in sel})
    if len(labels) < 2:
        return
    subjects = sorted({r["subject"] for r in sel})
    val = {(r["label"], r["subject"]): r[metric] for r in sel}
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 1.5, 4.2))
    for s in subjects:
        ys = [val.get((l, s), np.nan) for l in labels]
        ax.plot(range(len(labels)), ys, color=GRID, lw=1, zorder=1)
    for i, l in enumerate(labels):
        ys = np.array([val.get((l, s), np.nan) for s in subjects], dtype=float)
        ax.scatter(np.full(len(ys), i), ys, s=36, color=model_color(l, labels), edgecolor=SURFACE, lw=1.5, zorder=2)
        med = np.nanmedian(ys)
        ax.annotate(f"mediana {med:.0f}", (i, np.nanmax(ys)), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylabel(METRIC_NAME[metric])
    ax.grid(axis="x", visible=False)
    ax.set_title(f"BigMAC, {REGIME_NAME[regime]}: mismo caso unido entre modelos", loc="left")
    fig.tight_layout()
    save(fig, args.out_dir, "fig3_bigmac_oof_pareado.png")


def fig_utrecht(rows, args):
    rows = [r for r in rows if r["regime"] == "pet_only"]
    if not rows:
        return
    labels = sort_models({r["label"] for r in rows})
    for r in rows:
        r["all"] = "all"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, m in zip(axes, ["mae_body", "mae_bone", "dice_bone"]):
        grouped_bars(ax, rows, m, ["all"], "all", labels, {"all": ""})
        # Un solo grupo: el nombre del modelo bajo cada barra hace de leyenda.
        w = 0.8 / len(labels)
        ax.set_xticks([(i - (len(labels) - 1) / 2) * w for i in range(len(labels))])
        ax.set_xticklabels(labels)
    n = len({r["subject"] for r in rows})
    fig.suptitle(f"Utrecht zero-shot, solo PET (ensemble de 5 folds, n = {n} exámenes): "
                 "media ± desviación, puntos = exámenes", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, args.out_dir, "fig4_utrecht_zero_shot.png")


def read_ages(path):
    if not path or not Path(path).is_file():
        return {}
    ages = {}
    for r in csv.DictReader(open(path)):
        m = re.match(r"(\d+)Y", r.get("age", "") or "")
        if m:
            ages[r["sub_id"]] = int(m.group(1))
    return ages


def fig_utrecht_age(rows, args):
    ages = read_ages(args.participants)
    rows = [r for r in rows if r["regime"] == "pet_only" and r["subject"] in ages]
    if not rows:
        return
    labels = sort_models({r["label"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for l in labels:
        rs = [r for r in rows if r["label"] == l]
        ax.scatter([ages[r["subject"]] for r in rs], [r["mae_body"] for r in rs], s=42,
                   color=model_color(l, labels), edgecolor=SURFACE, lw=1.5, label=l, zorder=2)
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel(METRIC_NAME["mae_body"])
    ax.legend(title="Modelo")
    ax.set_title("Utrecht zero-shot, solo PET: MAE por examen según la edad", loc="left")
    fig.tight_layout()
    save(fig, args.out_dir, "fig5_utrecht_mae_vs_edad.png")


# ----------------------------------------------------------------------------- fig 10
def fig_b4_summary(rows, args):
    """Barras agrupadas por conjunto (folds out-of-fold / test.h5) para cada modelo de B4."""
    rows = [dict(r, split="test" if r["h5"] == "test.h5" else "folds")
            for r in rows if r["regime"] == "pet_only"]
    labels = sort_models({r["label"] for r in rows})
    if len(labels) < 2:
        return
    names = {"folds": "Folds (n = 20)\nout-of-fold", "test": "test.h5 (n = 5)\nnunca visto"}
    metrics = ["mae_body", "mae_soft", "mae_air", "dice_bone"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.4))
    for ax, m in zip(axes, metrics):
        grouped_bars(ax, rows, m, ["folds", "test"], "split", labels, names)
    fig.suptitle("Utrecht B4, solo PET: v18 zero-shot frente a fine-tuning; media ± desviación, puntos = exámenes",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    model_legend(fig, labels)
    save(fig, args.out_dir, "fig10_utrecht_B4_resumen.png")


# ----------------------------------------------------------------------------- fig 12
def fig_b4_age(rows, args, young_age=13):
    """MAE por examen frente a la edad para cada modelo de B4, con la media de < y >= young_age."""
    ages = read_ages(args.participants)
    rows = [r for r in rows if r["regime"] == "pet_only" and r["subject"] in ages]
    labels = sort_models({r["label"] for r in rows})
    if len(labels) < 2:
        return
    metrics = ["mae_body", "mae_air"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.6))
    offset = {l: (i - (len(labels) - 1) / 2) * 0.25 for i, l in enumerate(labels)}
    x_max = max(ages.values()) + 1
    for ax, m in zip(axes, metrics):
        ax.axvspan(0, young_age - 0.5, color=GRID, alpha=0.35, lw=0, zorder=0)
        for l in labels:
            rs = [r for r in rows if r["label"] == l]
            col = model_color(l, labels)
            ax.scatter([ages[r["subject"]] + offset[l] for r in rs], [r[m] for r in rs], s=38,
                       color=col, edgecolor=SURFACE, lw=1.2, zorder=3, label=l)
            # Media por grupo de edad como segmento horizontal.
            for lo, hi in ((0, young_age), (young_age, 99)):
                v = [r[m] for r in rs if lo <= ages[r["subject"]] < hi]
                if v:
                    x0, x1 = (1.5, young_age - 0.7) if lo == 0 else (young_age - 0.3, x_max - 0.2)
                    ax.plot([x0, x1], [np.mean(v)] * 2, color=col, lw=2, alpha=0.9, zorder=2)
                    # Medias como etiquetas apiladas bajo el nombre de cada grupo (pueden coincidir).
                    ax.annotate(f"{l}: {np.mean(v):.0f}", (x0, 1), xycoords=("data", "axes fraction"),
                                xytext=(0, -26 - 14 * labels.index(l)), textcoords="offset points",
                                fontsize=8, color=INK, zorder=4,
                                bbox=dict(fc=SURFACE, ec=col, lw=1, boxstyle="round,pad=0.2"))
        ax.set_xlim(1, x_max)
        ax.set_xlabel("Edad (años)")
        ax.set_title(METRIC_NAME[m], loc="left")
        for x0, name in ((1.5, f"< {young_age} años"), (young_age - 0.3, f"≥ {young_age} años")):
            ax.annotate(f"Media {name}", (x0, 1), xycoords=("data", "axes fraction"), xytext=(0, -10),
                        textcoords="offset points", fontsize=8.5, color=INK2, fontweight="bold")
    fig.suptitle("Utrecht B4, solo PET: error por examen según la edad; las líneas marcan la media "
                 f"de < {young_age} y ≥ {young_age} años", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    model_legend(fig, labels)
    save(fig, args.out_dir, "fig12_utrecht_B4_edad.png")


# ----------------------------------------------------------------------------- fig 8
def fig_b4_paired(rows, args, base="v18", new="v20"):
    """Por examen (ordenado por edad): punto del modelo base -> punto del fine-tuning."""
    ages = read_ages(args.participants)
    val = {(r["label"], r["subject"]): r for r in rows if r["regime"] == "pet_only"}
    subjects = sorted({s for l, s in val if l == new and (base, s) in val},
                      key=lambda s: (ages.get(s, 99), s))
    if not subjects:
        return
    test = {r["subject"] for r in rows if r["h5"] == "test.h5"}
    labels = [base, new]
    metrics = ["mae_body", "mae_air", "dice_bone"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9.5), sharex=True)
    x = np.arange(len(subjects))
    for ax, m in zip(axes, metrics):
        a = np.array([val[(base, s)][m] for s in subjects], dtype=float)
        b = np.array([val[(new, s)][m] for s in subjects], dtype=float)
        better = b > a if m == "dice_bone" else b < a
        for xi, ai, bi in zip(x, a, b):
            ax.plot([xi, xi], [ai, bi], color=GRID, lw=2, zorder=1)
        for lab, v in ((base, a), (new, b)):
            ax.scatter(x, v, s=40, color=model_color(lab, labels), edgecolor=SURFACE, lw=1.5, zorder=2, label=lab)
        fmt = "{:.3f}" if m == "dice_bone" else "{:.1f}"
        ax.set_title(f"{METRIC_NAME[m]}: media {base} {fmt.format(a.mean())} → {new} {fmt.format(b.mean())}; "
                     f"{new} mejor en {better.sum()} de {len(subjects)}", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{ages.get(s, '?')}a" + ("\ntest" if s in test else "") for s in subjects], fontsize=8.5)
    axes[-1].set_xlabel("Examen, ordenado por edad (test = test.h5, nunca visto; el resto: out-of-fold)")
    fig.suptitle(f"Utrecht B4, solo PET: {base} zero-shot frente a {new} con fine-tuning, por examen",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, args.out_dir, f"fig8_utrecht_B4_pareado_{new}.png")


# ----------------------------------------------------------------------------- fig 6 / 7
def coronal(vol, y):
    """Corte coronal con la cabeza arriba (volúmenes RAS como en los HDF5)."""
    return np.rot90(vol[:, y, :])


def load_case(h5_dir, sub_id):
    for p in sorted(Path(h5_dir).glob("*.h5")):
        with h5py.File(p, "r") as f:
            if sub_id in f:
                g = f[sub_id]
                lo, hi = g.attrs.get("body_lo"), g.attrs.get("body_hi")
                return g["pet"][:], g["ct"][:], g["mask"][:], lo, hi
    return None


def fig_qualitative(case, sub_id, pred_dir, panels, title, out_name, args):
    """panels: lista de (etiqueta del modelo, régimen, título de la columna)."""
    pet, ct, mask, lo, hi = case
    preds = []
    for label, regime, name in panels:
        p = Path(pred_dir) / label / regime / sub_id / "ct.nii.gz"
        if p.is_file():
            preds.append((name, np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)))
    if not preds:
        return
    y = int((lo[1] + hi[1]) // 2) if lo is not None else ct.shape[1] // 2
    body = mask > 0
    ncol = 2 + len(preds)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 2 * 3.6), squeeze=False)
    ct_kw = dict(cmap="gray", vmin=-1000, vmax=1500)
    axes[0][0].imshow(coronal(pet, y), cmap="gray_r", vmax=np.percentile(pet[body], 99.5))
    axes[0][0].set_title("NAC-PET")
    axes[0][1].imshow(coronal(ct, y), **ct_kw)
    axes[0][1].set_title("CT")
    err_im = None
    for j, (name, pr) in enumerate(preds):
        mae = np.abs(pr - ct)[body].mean()
        axes[0][2 + j].imshow(coronal(pr, y), **ct_kw)
        axes[0][2 + j].set_title(f"{name}\nMAE {mae:.0f} HU", fontsize=9.5)
        err = np.where(body, np.abs(pr - ct), 0)
        err_im = axes[1][2 + j].imshow(coronal(err, y), cmap=ERR_CMAP, vmin=0, vmax=500)
        axes[1][2 + j].set_title("|pseudo-CT − CT|", fontsize=9, fontweight="normal", color=INK2)
    axes[1][0].imshow(coronal(mask, y), cmap="gray", vmin=0, vmax=6)
    axes[1][0].set_title("Máscara (clases)", fontsize=9, fontweight="normal", color=INK2)
    axes[1][1].axis("off")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    if err_im is not None:
        cax = axes[1][1].inset_axes([0.55, 0.15, 0.08, 0.7])
        cb = fig.colorbar(err_im, cax=cax)
        cb.set_label("Error absoluto (HU)", color=INK2)
        cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, args.out_dir, out_name)


def figs_qualitative(utr_rows, args):
    pred_root = Path(args.pred_dir)
    # Utrecht: los exámenes con pseudo-CT guardado de algún modelo.
    utr_subjects = sorted({p.parent.name for p in pred_root.glob("utrecht_pred/*/pet_only/*/ct.nii.gz")})
    if args.utrecht_subjects:
        utr_subjects = [s for s in utr_subjects if s in args.utrecht_subjects]
    elif utr_rows:
        # Por defecto: el examen con MAE mediana de v18 y el peor, para no elegir a mano el mejor caso.
        v = sorted((r["mae_body"], r["subject"]) for r in utr_rows if r["label"] == "v18" and r["subject"] in utr_subjects)
        if v:
            utr_subjects = sorted({v[len(v) // 2][1], v[-1][1]})
    ages = read_ages(args.participants)
    for s in utr_subjects:
        case = load_case(args.utrecht_h5_dir, s)
        if case is None:
            continue
        panels = [(m, "pet_only", m) for m in ["v12", "v18", "v19"]]
        age = f", {ages[s]} años" if s in ages else ""
        fig_qualitative(case, s, pred_root / "utrecht_pred", panels,
                        f"Utrecht {s}{age} — zero-shot, solo PET (corte coronal central)", f"fig6_utrecht_{s}.png", args)

    # B4: v18 frente a v20 en los exámenes guardados por run_B3_train.sh.
    b4_models = sorted({p.parts[-4] for p in pred_root.glob("utrecht_pred/v2*/pet_only/*/ct.nii.gz")})
    for s in sorted({p.parent.name for p in pred_root.glob("utrecht_pred/v2*/pet_only/*/ct.nii.gz")}):
        case = load_case(args.utrecht_h5_dir, s)
        if case is None:
            continue
        age = f", {ages[s]} años" if s in ages else ""
        panels = [("v18", "pet_only", "v18 zero-shot")] + [(m, "pet_only", f"{m} fine-tuning") for m in b4_models]
        fig_qualitative(case, s, pred_root / "utrecht_pred", panels,
                        f"Utrecht {s}{age} — B4, solo PET (corte coronal central)", f"fig9_utrecht_B4_{s}.png", args)

    for s in sorted({p.parent.name for p in pred_root.glob("bigmac_pred/*/*/*/ct.nii.gz")}):
        case = load_case(args.bigmac_h5_dir, s)
        if case is None:
            continue
        panels = [("v12", "full", "v12 con MRI"), ("v18", "full", "v18 con MRI"),
                  ("v12", "pet_only", "v12 solo PET"), ("v18", "pet_only", "v18 solo PET"),
                  ("v19", "pet_only", "v19 solo PET")]
        fig_qualitative(case, s, pred_root / "bigmac_pred", panels,
                        f"BigMAC {s} — out-of-fold, con y sin MRI (corte coronal central)", f"fig7_bigmac_{s}.png", args)


# ----------------------------------------------------------------------------- tablas
def write_tables(oof_rows, utr_rows, args, b4_rows=()):
    metrics = ["mae_body", "mae_soft", "mae_air", "mae_bone", "dice_bone", "me_body"]
    lines = []

    def table(title, rows, keys, metrics=metrics):
        if not rows:
            return
        lines.append(f"## {title}\n")
        lines.append("| Modelo | " + " | ".join(k[0] for k in keys) + " | n | "
                     + " | ".join(METRIC_NAME[m] for m in metrics) + " |")
        lines.append("|" + "---|" * (len(keys) + 2 + len(metrics)))
        groups = {}
        for r in rows:
            groups.setdefault((r["label"],) + tuple(k[1](r) for k in keys), []).append(r)
        order = lambda g: (MODEL_ORDER.index(g[0]) if g[0] in MODEL_ORDER else 99,
                           [REGIME_ORDER.index(x) if x in REGIME_ORDER else 9 for x in g[1:]], g)
        for g in sorted(groups, key=order):
            rs = groups[g]
            cells = []
            for m in metrics:
                v = np.array([r[m] for r in rs], dtype=float)
                v = v[~np.isnan(v)]
                cells.append("-" if not v.size else f"{v.mean():.3f} ± {v.std():.3f}" if m in ("dice_bone", "ghost_frac")
                             else f"{v.mean():.1f} ± {v.std():.1f}")
            names = [REGIME_NAME.get(x, x) for x in g[1:]]
            lines.append(f"| {g[0]} | " + " | ".join(names) + f" | {len(rs)} | " + " | ".join(cells) + " |")
        lines.append("")

    table("BigMAC out-of-fold (A6)", oof_rows, [("Régimen", lambda r: r["regime"])])
    table("Utrecht zero-shot, solo PET", [r for r in utr_rows if r["regime"] == "pet_only"],
          [("Conjunto", lambda r: "test.h5" if r["h5"] == "test.h5" else "folds (sin test)")])
    if utr_rows:
        table("Utrecht zero-shot, solo PET, todos los exámenes",
              [dict(r, h5="todos") for r in utr_rows if r["regime"] == "pet_only"],
              [("Conjunto", lambda r: "todos")])
    if b4_rows:
        table("Utrecht B4, solo PET: v18 zero-shot (ensemble) frente a fine-tuning (v20, v21)",
              [r for r in b4_rows if r["regime"] == "pet_only"],
              [("Conjunto", lambda r: "test.h5 (ensemble)" if r["h5"] == "test.h5"
                else "folds (out-of-fold)" if r["fold"] != "ens01234" else "folds (ensemble)")],
              metrics=metrics + ["mae_thorax", "ghost_frac"])
    if lines:
        path = Path(args.out_dir) / "tablas.md"
        path.write_text("# Resultados Fase A\n\n" + "\n".join(lines))
        print(f"  {path}")


def main():
    parser = argparse.ArgumentParser("Figuras y tablas de la Fase A")
    parser.add_argument("--out_dir", default=f"{EVAL}/figuras")
    parser.add_argument("--oof_csv", default=f"{EVAL}/bigmac_oof.csv")
    parser.add_argument("--utrecht_csv", default=f"{EVAL}/utrecht_zero_shot.csv")
    parser.add_argument("--b4_csv", default=f"{EVAL}/utrecht_B4.csv")
    parser.add_argument("--pred_dir", default=EVAL, help="Contiene utrecht_pred/ y bigmac_pred/")
    parser.add_argument("--train_logs", default="logs/train_*.log")
    parser.add_argument("--v12_ckpt_dir", default="checkpoints")
    parser.add_argument("--bigmac_h5_dir", default="/mnt/c/Users/edyA0/Proyectos/data_multimask")
    parser.add_argument("--utrecht_h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    parser.add_argument("--participants", default="/mnt/d/Data/007_pseudoCT/utrecht_bigmac_space/participants.csv")
    parser.add_argument("--utrecht_subjects", nargs="*", default=None,
                        help="Exámenes para las figuras cualitativas (por defecto: mediana y peor de v18)")
    parser.add_argument("--skip", nargs="*", default=[], choices=["curvas", "bigmac", "utrecht", "b4", "cortes"])
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    oof_rows = read_csv(args.oof_csv)
    utr_rows = read_csv(args.utrecht_csv)
    b4_rows = read_csv(args.b4_csv)
    print("Figuras:")
    if "curvas" not in args.skip:
        fig_training_curves(args)
    if "bigmac" not in args.skip:
        fig_bigmac_regimes(oof_rows, args)
        fig_bigmac_paired(oof_rows, args)
    if "utrecht" not in args.skip:
        fig_utrecht(utr_rows, args)
        fig_utrecht_age(utr_rows, args)
    if "b4" not in args.skip:
        for new in sort_models({r["label"] for r in b4_rows} - {"v18"}):
            fig_b4_paired(b4_rows, args, new=new)
        fig_b4_summary(b4_rows, args)
        fig_b4_age(b4_rows, args)
    if "cortes" not in args.skip:
        figs_qualitative(utr_rows, args)
    write_tables(oof_rows, utr_rows, args, b4_rows)


if __name__ == "__main__":
    main()
