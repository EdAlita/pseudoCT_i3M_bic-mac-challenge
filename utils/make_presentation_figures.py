"""
utils/make_presentation_figures.py

Figures for the final UMC Utrecht presentation: English, large text, one message per figure.
Every figure has a title that states its finding and a subtitle that explains how to read it.
Each figure is skipped if its data are missing, so the script can be run at any time.

    01_cohorts.png                 Who is in the data: adults (BIC-MAC) vs children (Utrecht)
    02_bigmac_dataset.png          BIC-MAC: age, sex, body size, what each subject has
    05_bigmac_inputs.png           BIC-MAC: error by available inputs; why train with MRI dropout
    06_bigmac_tissues.png          BIC-MAC: error per tissue with and without MRI
    07_bigmac_example.png          BIC-MAC: one case with and without MRI
    08_pet_domain_gap.png          NAC-PET looks different in Utrecht -> domain augmentation
    09_synthetic_paediatric.png    Adult CT/PET deformed to a child's anatomy (synthetic pairs)
    11_utrecht_model_progress.png  Utrecht: body MAE of every model, all exams and < 13 years
    13_utrecht_per_patient.png     Utrecht: zero-shot vs final model, patient by patient
    15_utrecht_tissues.png         Utrecht: error per tissue, zero-shot vs final model
    18_utrecht_error_vs_size.png   Utrecht: error vs age and chest width
    19_utrecht_hu_per_tissue.png   Utrecht: mean HU per tissue, CT vs pseudo-CT, per patient
    20_utrecht_hu_histogram.png    Utrecht: voxel-wise pseudo-CT vs CT (all patients)
    22_utrecht_patient_table.png   Utrecht: every number for every patient (table)
    23_utrecht_examples.png        Utrecht: youngest, median and worst patient
    utrecht_per_patient.csv / .md  The table of fig. 14, plus the HU per tissue
    figure_guide.md                Caption and talking points for each figure

Example:
    python -m utils.make_presentation_figures --final v23
    python -m utils.make_presentation_figures --final v25 --models v18 v20 v21 v23 v24 v25 v18+v23

Figs. 12, 13 and 15 need the saved pseudo-CTs of the final model for all exams
(evaluate_models --save_dir eval/utrecht_pred, without --save_subjects).

Owner: Edwing Ulin
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy import stats  # noqa: E402

from utils.body_coords import lung_z_range  # noqa: E402
from utils.make_figures import ERR_CMAP, coronal, parse_train_logs, read_ages, read_csv  # noqa: E402

EVAL = "/mnt/d/Data/007_pseudoCT/eval"
VOXEL_MM = (1.5234375, 1.5234375, 2.0)   # grid of both HDF5 cohorts
YOUNG = 13
Z_BINS = 40   # bins along the body height for the error profile
LUNG_HU = -400   # pseudo-CT voxels below this inside the lung slab count as predicted lung
HU_KEYS = {"z", "body", "lungvol", "acf", "kvp"}   # cache entries computed before these existed are recomputed

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
FINAL_C, BASE_C, OTHER_C = "#2a78d6", "#eb6834", "#b9b8b3"
ADULT_C, CHILD_C, TEEN_C = "#4a3aa7", "#e34948", "#eda100"
GOOD_C, BAD_C = "#1baf7a", "#e34948"

MODEL_INFO = {
    "v18": "Adult model (BIC-MAC), no fine-tuning",
    "v20": "Fine-tuned on Utrecht",
    "v21": "+ lung and young-patient sampling",
    "v22": "+ adult-shrink augmentation",
    "v23": "+ NAC-PET domain augmentation",
    "v24": "+ synthetic paediatric pairs (20 %)",
    "v25": "+ synthetic young pairs (40 %)",
    "v18+v23": "Average of v18 and v23",
    "water": "Baseline: water inside the NAC-PET body contour",
}
MODEL_ORDER = list(MODEL_INFO)
REGIME_NAME = {"full": "PET + MRI\n+ topogram", "no_topo": "PET + MRI", "no_mri": "PET\n+ topogram",
               "pet_only": "PET only"}
TISSUES = [("mae_body", "Whole body", "MAE (HU)"), ("mae_soft", "Soft tissue", "MAE (HU)"),
           ("mae_air", "Lungs / internal air", "MAE (HU)"), ("mae_bone", "Bone", "MAE (HU)"),
           ("dice_bone", "Bone overlap", "Dice (CT > 300 HU)")]
# HU classes for the CT-number analysis (inside the body; mask: 3 internal air, 2 soft, 5/6 bone).
HU_CLASSES = [("lung", "Lungs / internal air"), ("fat", "Fat (−190 to −30 HU)"),
              ("soft", "Soft tissue (−30 to 150 HU)"), ("bone", "Bone")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1.0, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "legend.frameon": False, "axes.titlepad": 12,
})
GUIDE = []
ages_global = {}
PAPER = False   # --paper: no titles/subtitles (moved to captions), shorter labels, cropped to content


# ----------------------------------------------------------------------------- helpers
def header(fig, title, subtitle, top=None):
    """Finding as the title, how to read the figure as the subtitle (placed in inches, so any figure height works)."""
    h = fig.get_size_inches()[1]
    n_lines = subtitle.count("\n") + 1
    fig.text(0.01, 1 - 0.2 / h, title, ha="left", va="top", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.75 / h, subtitle, ha="left", va="top", fontsize=15, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - (1.05 + 0.32 * n_lines) / h))


def save(fig, args, name, caption, points=()):
    path = Path(args.out_dir) / name
    if PAPER:
        # Titles and subtitles are figure-level texts in the top band; drop them and crop to the content.
        for t in list(fig.texts):
            if t.get_position()[1] > 0.8 and t.get_fontsize() >= 14:
                t.remove()
        if fig._suptitle is not None:
            fig._suptitle.remove()
            fig._suptitle = None
    try:
        fig.savefig(path, dpi=200, **({"bbox_inches": "tight", "pad_inches": 0.15} if PAPER else {}))
        print(f"  {path}")
    except OSError as e:
        # Windows locks files open in a viewer: keep going so the other figures and the guide are written.
        print(f"  NOT SAVED (file open in another program?): {path} ({e})")
    plt.close(fig)
    GUIDE.append((name, caption, list(points)))


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def body_window(img, msk, size):
    """Crop a coronal image to a fixed (h, w) window centred on the body, so panels share one scale."""
    zs, xs = np.where(msk)
    cz, cx = ((zs.min() + zs.max()) // 2, (xs.min() + xs.max()) // 2) if zs.size else (img.shape[0] // 2, img.shape[1] // 2)
    h, w = size
    p = max(h, w)
    big = np.pad(img, p, mode="constant", constant_values=float(img.min()))
    return big[cz + p - h // 2: cz + p - h // 2 + h, cx + p - w // 2: cx + p - w // 2 + w]


def body_size(msks, margin=12):
    """Window (h, w) that fits the body in every coronal mask of a figure."""
    h = w = 0
    for m in msks:
        zs, xs = np.where(m)
        if zs.size:
            h, w = max(h, zs.max() - zs.min() + 1), max(w, xs.max() - xs.min() + 1)
    return h + 2 * margin, w + 2 * margin


def mid_y(case, vol):
    lo, hi = case["attrs"].get("body_lo"), case["attrs"].get("body_hi")
    return int((lo[1] + hi[1]) // 2) if lo is not None else vol.shape[1] // 2


def short_id(sub):
    """sub-PUMP136413-20241213 -> PUMP136413 (2024-12-13)."""
    p = sub.replace("sub-", "").split("-")
    return f"{p[0]} ({p[1][:4]}-{p[1][4:6]}-{p[1][6:]})" if len(p) > 1 else p[0]


def wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        return stats.wilcoxon(a, b).pvalue
    except ValueError:
        return np.nan


def p_text(p):
    return "n/a" if np.isnan(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def mean_sd(v, fmt="{:.0f}"):
    v = np.asarray(v, float)
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std())}"


def read_participants(path):
    out = {}
    if Path(path).is_file():
        for r in csv.DictReader(open(path)):
            out[r["sub_id"]] = r
    return out


def read_cohorts(path):
    if not Path(path).is_file():
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["cohort"] = r["cohort"].replace("BigMAC", "BIC-MAC")   # cohortes.csv predates the rename
        for k in list(r):
            if k not in ("cohort", "split", "subject", "sex"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = np.nan
    return rows


def b4_table(rows, models):
    """{model: {subject: row}} for the PET-only B4 rows (out-of-fold for folds, ensemble for test)."""
    out = {}
    for r in rows:
        if r["regime"] == "pet_only" and r["label"] in models:
            out.setdefault(r["label"], {})[r["subject"]] = r
    return out


def load_h5(h5_dir, sub_id, keys=("pet", "ct", "mask")):
    for p in sorted(Path(h5_dir).glob("*.h5")):
        with h5py.File(p, "r") as f:
            if sub_id in f:
                g = f[sub_id]
                out = {k: g[k][:] for k in keys if k in g}
                out["attrs"] = dict(g.attrs)
                out["split"] = p.stem
                return out
    return None


def load_pred(pred_dir, label, regime, sub_id):
    p = Path(pred_dir) / label / regime / sub_id / "ct.nii.gz"
    return np.asarray(nib.load(str(p)).dataobj, dtype=np.float32) if p.is_file() else None


# ----------------------------------------------------------------------------- 01 / 02 data
def fig_cohorts(coh, args):
    if not coh:
        return
    big = [r for r in coh if r["cohort"] == "BIC-MAC"]
    utr = [r for r in coh if r["cohort"] == "Utrecht"]
    young = [r for r in utr if r["age"] < YOUNG]
    old = [r for r in utr if r["age"] >= YOUNG]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    bins = np.arange(0, 95, 5)
    ax.hist([r["age"] for r in big], bins=bins, color=ADULT_C, alpha=0.85, label=f"BIC-MAC adults (n = {len(big)})",
            edgecolor=SURFACE, lw=1.5)
    ax.hist([r["age"] for r in utr], bins=np.arange(0, 22, 2), color=CHILD_C, alpha=0.85,
            label=f"Utrecht children (n = {len(utr)})", edgecolor=SURFACE, lw=1.5)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of exams")
    ax.set_title("Age", loc="left")
    ax.legend(loc="upper right")
    groups = [("BIC-MAC\nadults", big, ADULT_C), (f"Utrecht\n≥ {YOUNG} y", old, TEEN_C),
              (f"Utrecht\n< {YOUNG} y", young, CHILD_C)]
    rng = np.random.default_rng(0)
    for ax, key, name in ((axes[1], "chest_width_cm", "Chest width (cm)"), (axes[2], "lung_vol_l", "Lung volume (L)")):
        ref = np.nanmedian([r[key] for r in big])
        for i, (g, rs, c) in enumerate(groups):
            v = np.array([r[key] for r in rs], float)
            v = v[~np.isnan(v)]
            ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=60, color=c, alpha=0.75, edgecolor=SURFACE, lw=1)
            med = np.median(v)
            ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=3)
            ax.annotate(f"{med:.1f}" + ("" if i == 0 else f"\n(×{med / ref:.2f})"), (i + 0.33, med), va="center",
                        fontsize=13, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels([g for g, _, _ in groups])
        ax.set_xlim(-0.5, 2.9)
        ax.set_title(name, loc="left")
        ax.grid(axis="x", visible=False)
    header(fig, "The model learned on adults; the target patients are children",
           f"BIC-MAC: {len(big)} adults ({np.nanmin([r['age'] for r in big]):.0f}–{np.nanmax([r['age'] for r in big]):.0f} y) "
           f"with NAC-PET, MRI and CT.   UMC Utrecht: {len(utr)} paediatric exams "
           f"({np.nanmin([r['age'] for r in utr]):.0f}–{np.nanmax([r['age'] for r in utr]):.0f} y) with NAC-PET and CT, no MRI.\n"
           "Black bar = median; (×) = size relative to the adult median, measured on the CT masks of both cohorts.")
    ys = [r["chest_width_cm"] for r in young]
    save(fig, args, "01_cohorts.png",
         "Age and body size of the two cohorts. Body size is measured the same way in both cohorts, from the CT "
         "masks at the middle of the lungs.",
         [f"Children under {YOUNG} have a chest about {np.nanmedian(ys) / np.nanmedian([r['chest_width_cm'] for r in big]):.2f}× "
          "as wide as an adult chest and much smaller lungs.",
          f"Only {len(young)} of the {len(utr)} Utrecht exams are under {YOUNG} years, so the youngest children are rare "
          "in the data the model can learn from."])


def fig_bigmac_dataset(coh, oof, args):
    big = [r for r in coh if r["cohort"] == "BIC-MAC"]
    if not big:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
    sexc = {"F": CHILD_C, "M": ADULT_C}
    sexn = {"F": "Female", "M": "Male"}
    ax = axes[0]
    bins = np.arange(20, 95, 5)
    data = [[r["age"] for r in big if r["sex"] == s] for s in ("F", "M")]
    ax.hist(data, bins=bins, stacked=True, color=[sexc["F"], sexc["M"]], edgecolor=SURFACE, lw=1.5,
            label=[f"{sexn[s]} (n = {len(d)})" for s, d in zip(("F", "M"), data)])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Age and sex", loc="left")
    ax.legend(loc="upper left")
    ax = axes[1]
    for s in ("F", "M"):
        rs = [r for r in big if r["sex"] == s and not np.isnan(r.get("weight_kg", np.nan))
              and not np.isnan(r.get("height_true_cm", np.nan))]
        ax.scatter([r["height_true_cm"] for r in rs], [r["weight_kg"] for r in rs], s=70, color=sexc[s],
                   alpha=0.8, edgecolor=SURFACE, lw=1, label=sexn[s])
    h = np.linspace(150, 200, 50)
    for bmi in (18.5, 25, 30):
        ax.plot(h, bmi * (h / 100) ** 2, color=INK2, lw=1, ls="--")
        ax.annotate(f"BMI {bmi:g}", (200, bmi * 4), fontsize=12, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title("Body size", loc="left")
    ax.legend(loc="upper left")
    ax = axes[2]
    lungs = np.array([r["lung_vol_l"] for r in big], float)
    width = np.array([r["chest_width_cm"] for r in big], float)
    ax.scatter(width, lungs, s=70, color=ADULT_C, alpha=0.8, edgecolor=SURFACE, lw=1)
    ax.set_xlabel("Chest width (cm)")
    ax.set_ylabel("Lung volume (L)")
    ax.set_title("Thorax", loc="left")
    folds = sorted({r["split"] for r in big})
    bmis = np.array([r["weight_kg"] / (r["height_true_cm"] / 100) ** 2 for r in big
                     if r.get("weight_kg") and r.get("height_true_cm")], float)
    bmis = bmis[~np.isnan(bmis)]
    header(fig, f"BIC-MAC: {len(big)} adults with PET, MRI and CT of the torso",
           f"Each subject has NAC-PET, MRI, a CT topogram and the CT target, registered to the same grid. "
           f"Split into {len(folds)} folds of {len(big) // max(len(folds), 1)} subjects for cross-validation.\n"
           f"Median age {np.nanmedian([r['age'] for r in big]):.0f} y; median BMI {np.median(bmis):.1f}; dashed lines = BMI 18.5 / 25 / 30.")
    save(fig, args, "02_bigmac_dataset.png",
         "The adult training cohort: age, sex, body size and thorax size of the BIC-MAC subjects.",
         [f"{sum(r['sex'] == 'F' for r in big)} women and {sum(r['sex'] == 'M' for r in big)} men, "
          f"{np.nanmin([r['age'] for r in big]):.0f}–{np.nanmax([r['age'] for r in big]):.0f} years.",
          "This is the only cohort with MRI, so it is where the model learns the PET → CT mapping first."])


# ----------------------------------------------------------------------------- 03 / 04 / 05 BIC-MAC results
def fig_bigmac_inputs(oof, args):
    v18 = [r for r in oof if r["label"] == "v18"]
    v12 = [r for r in oof if r["label"] == "v12"]
    if not v18:
        return
    regimes = [g for g in REGIME_NAME if any(r["regime"] == g for r in v18)]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    means = {}
    for i, g in enumerate(regimes):
        v = np.array([r["mae_body"] for r in v18 if r["regime"] == g], float)
        means[g] = v.mean()
        c = FINAL_C if g == "pet_only" else OTHER_C
        ax.bar(i, v.mean(), 0.65, color=c, edgecolor=SURFACE)
        ax.scatter(i + rng.uniform(-0.2, 0.2, v.size), v, s=14, color=INK, alpha=0.35, lw=0, zorder=3)
        ax.annotate(f"{v.mean():.0f} HU", (i, v.mean() / 2), ha="center", va="center", fontsize=20,
                    fontweight="bold", color=INK if c == OTHER_C else "white",
                    bbox=dict(fc=c, ec="none", pad=2))
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([REGIME_NAME[g] for g in regimes])
    ax.set_ylabel("Body MAE (HU), lower is better")
    ax.set_title("(a) v18, inputs at test time" if PAPER else "Final adult model (v18): inputs available at test time", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(means.values()) * 1.45)
    ax = axes[1]
    pairs = [("v12", "Trained always\nwith MRI (v12)", v12), ("v18", "Trained with\nMRI dropout (v18)", v18)]
    pairs = [p for p in pairs if any(r["regime"] == "pet_only" for r in p[2])]
    pet = {}
    for i, (lab, name, rs) in enumerate(pairs):
        v = np.array([r["mae_body"] for r in rs if r["regime"] == "pet_only"], float)
        pet[lab] = v.mean()
        ax.bar(i, v.mean(), 0.6, color=BASE_C if lab == "v12" else FINAL_C, edgecolor=SURFACE)
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=14, color=INK, alpha=0.35, lw=0, zorder=3)
        ax.annotate(f"{v.mean():.0f} HU", (i, v.mean() / 2 if lab == "v12" else v.mean()),
                    xytext=(0, 0 if lab == "v12" else 40), textcoords="offset points", ha="center", va="center",
                    fontsize=20, fontweight="bold", color="white" if lab == "v12" else INK,
                    bbox=dict(fc=BASE_C if lab == "v12" else SURFACE, ec="none", pad=2))
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p[1] for p in pairs])
    ax.set_title("(b) PET only" if PAPER else "Both tested with PET only", loc="left")
    ax.grid(axis="x", visible=False)
    n = len({r["subject"] for r in v18})
    header(fig, f"On adults, PET alone gives {means.get('pet_only', np.nan):.0f} HU "
                f"vs {means.get('full', np.nan):.0f} HU with MRI",
           f"BIC-MAC, out-of-fold: each of the {n} adults is predicted by the model that did not see it in training. "
           "Bars = mean, dots = subjects.\nThe right panel shows why the model was trained with MRI randomly removed: "
           "a model that always saw MRI fails when MRI is missing.")
    pts = [f"Removing MRI raises the body error from {means.get('full', np.nan):.0f} to "
           f"{means.get('pet_only', np.nan):.0f} HU; the topogram adds almost nothing."]
    if "v12" in pet:
        pts.append(f"Without MRI dropout in training, PET-only error is {pet['v12']:.0f} HU (v12); with it, "
                   f"{pet['v18']:.0f} HU (v18). Utrecht has no MRI, so this was essential.")
    save(fig, args, "05_bigmac_inputs.png",
         "BIC-MAC out-of-fold body MAE of the adult model for each combination of inputs, and the effect of training "
         "with MRI dropout.", pts)


def fig_bigmac_tissues(oof, args):
    v18 = [r for r in oof if r["label"] == "v18"]
    if not any(r["regime"] == "full" for r in v18):
        return
    fig, axes = plt.subplots(1, len(TISSUES), figsize=(20, 7))
    pts = []
    for ax, (m, name, unit) in zip(axes, TISSUES):
        vals = []
        for i, (g, c) in enumerate((("full", OTHER_C), ("pet_only", FINAL_C))):
            v = np.array([r[m] for r in v18 if r["regime"] == g], float)
            v = v[~np.isnan(v)]
            vals.append(v.mean())
            ax.bar(i, v.mean(), 0.65, color=c, edgecolor=SURFACE)
            ax.errorbar(i, v.mean(), yerr=v.std(), color=INK2, lw=1.5, capsize=5)
            fmt = "{:.2f}" if m == "dice_bone" else "{:.0f}"
            ax.annotate(fmt.format(v.mean()), (i, v.mean() + v.std()), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=16, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["With\nMRI", "PET\nonly"])
        ax.set_title(name, loc="left")
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        pts.append(f"{name}: {vals[0]:.2f} → {vals[1]:.2f}" if m == "dice_bone" else f"{name}: {vals[0]:.0f} → {vals[1]:.0f} HU")
    header(fig, "Without MRI, bone is the hardest tissue to predict",
           "BIC-MAC out-of-fold, adult model v18. Bars = mean, whiskers = ± 1 standard deviation across subjects. "
           "MAE: lower is better; Dice: higher is better\n(overlap of voxels > 300 HU between pseudo-CT and CT).")
    save(fig, args, "06_bigmac_tissues.png",
         "Error per tissue on BIC-MAC with and without MRI (adult model v18, out-of-fold).", pts)


def fig_bigmac_example(oof, args):
    subs = sorted({p.parent.name for p in Path(args.pred_dir, "bigmac_pred", "v18", "pet_only").glob("*/ct.nii.gz")})
    if not subs:
        return
    val = {r["subject"]: r["mae_body"] for r in oof if r["label"] == "v18" and r["regime"] == "pet_only"}
    subs = sorted(subs, key=lambda s: val.get(s, 0))
    sub = subs[len(subs) // 2]
    case = load_h5(args.bigmac_h5_dir, sub, keys=("pet", "ct", "mask", "mri"))
    full = load_pred(Path(args.pred_dir, "bigmac_pred"), "v18", "full", sub)
    pet_only = load_pred(Path(args.pred_dir, "bigmac_pred"), "v18", "pet_only", sub)
    if case is None or full is None or pet_only is None:
        return
    ct, mask, pet = case["ct"], case["mask"], case["pet"]
    body = mask > 0
    y = mid_y(case, ct)
    bm = coronal(body, y)
    win = body_size([bm])
    W = lambda im: body_window(im, bm, win)
    fig, axes = plt.subplots(2, 3, figsize=(16, 15))
    ct_kw = dict(cmap="gray", vmin=-1000, vmax=1500)
    axes[0, 0].imshow(W(coronal(pet, y)), cmap="gray_r", vmax=np.percentile(pet[body], 99.5))
    axes[0, 0].set_title("NAC-PET (input)")
    mri = case.get("mri")
    if mri is not None:
        m0 = mri[0] if mri.ndim == 4 else mri
        axes[0, 1].imshow(W(coronal(m0, y)), cmap="gray", vmax=np.percentile(m0[body], 99.5))
    axes[0, 1].set_title("MRI (input, not available in Utrecht)")
    axes[0, 2].imshow(W(coronal(ct, y)), **ct_kw)
    axes[0, 2].set_title("Real CT (target)")
    for j, (pr, name) in enumerate(((full, "with MRI"), (pet_only, "PET only"))):
        mae = np.abs(pr - ct)[body].mean()
        axes[1, j].imshow(W(coronal(pr, y)), **ct_kw)
        axes[1, j].set_title(f"Pseudo-CT, {name}\nMAE {mae:.0f} HU")
    im = axes[1, 2].imshow(W(coronal(np.where(body, np.abs(pet_only - ct), 0), y)), cmap=ERR_CMAP, vmin=0, vmax=500)
    axes[1, 2].set_title("|Error|, PET only")
    for ax in axes.flat:
        clean_axes(ax)
    cb = fig.colorbar(im, ax=axes[1, 2], fraction=0.05, pad=0.02)
    cb.set_label("Absolute error (HU)", color=INK2, fontsize=15)
    cb.outline.set_visible(False)
    header(fig, "On adults, the PET-only pseudo-CT keeps the anatomy but loses bone detail",
           f"BIC-MAC subject {sub} (median error of the {len(val)} subjects), central coronal slice, adult model v18 "
           "tested on a fold it did not see.\nError map: white = correct, dark blue = error of 500 HU or more.",
           top=0.89)
    save(fig, args, "07_bigmac_example.png",
         f"One BIC-MAC adult ({sub}) predicted with and without MRI.",
         ["Without MRI the body outline, lungs and soft tissue are preserved; bone becomes blurrier."])


# ----------------------------------------------------------------------------- 06 PET domain gap
def pet_profile(pet, body, aug=None):
    """Mean NAC-PET (relative to the 95th percentile in the body) per band of distance to the skin."""
    pet = pet[::2, ::2, ::2].astype(np.float32)
    body = body[::2, ::2, ::2]
    if aug is not None:
        pet = aug(pet, body)
    pet /= np.percentile(pet[body], 95)
    sp = (2 * VOXEL_MM[0], 2 * VOXEL_MM[1], 2 * VOXEL_MM[2])
    din = ndi.distance_transform_edt(body, sampling=sp)
    dout = ndi.distance_transform_edt(~body, sampling=sp)
    prof_in = [float(pet[body & (din > a) & (din <= b)].mean()) for a, b in zip(IN_BINS[:-1], IN_BINS[1:])]
    prof_out = [float(pet[~body & (dout > a) & (dout <= b)].mean()) for a, b in zip(OUT_BINS[:-1], OUT_BINS[1:])]
    return prof_in, prof_out


IN_BINS = [0, 10, 20, 30, 45, 60, 80, 100, 130]
OUT_BINS = [0, 10, 20, 35, 50, 70, 100]


def domain_aug_mid(pet, body):
    """The training augmentation (data/dataset.py) with every parameter at the middle of its range."""
    from config.defaults import get_cfg_defaults
    d = get_cfg_defaults().DATA
    mid = lambda k: sum(getattr(d, k)) / 2
    dark, depth, rim, halo, halo_mm = (mid("PET_DOMAIN_DARK"), mid("PET_DOMAIN_DEPTH_MM"), mid("PET_DOMAIN_RIM_MM"),
                                       mid("PET_DOMAIN_HALO"), mid("PET_DOMAIN_HALO_MM"))
    # Same convention as training: half-resolution distance with voxel 2 x VOXEL_SIZES[0] of the config.
    vox = 2 * 2.0644
    b2 = body[::2, ::2, ::2]
    d_in = ndi.distance_transform_edt(b2) * vox
    d_out = ndi.distance_transform_edt(~b2) * vox
    up = lambda a: a.repeat(2, 0).repeat(2, 1).repeat(2, 2)[:pet.shape[0], :pet.shape[1], :pet.shape[2]]
    d_in, d_out = up(d_in), up(d_out)
    ref = float(np.percentile(pet[body], 95))
    gain = np.where(body, 1.0 - dark * np.clip((d_in - rim) / depth, 0, 1), 1.0)
    return (pet * gain + np.where(body, 0.0, halo * ref * np.exp(-d_out / halo_mm))).astype(np.float32)


def fig_pet_domain(args):
    cache = Path(args.cache_dir) / "cache_pet_profiles.json"
    if cache.is_file():
        prof = {k.replace("BigMAC", "BIC-MAC"): v for k, v in json.loads(cache.read_text()).items()}
    else:
        prof = {"BIC-MAC": [], "BIC-MAC + augmentation": [], "Utrecht": []}
        for h5 in sorted(Path(args.bigmac_h5_dir).glob("fold_*.h5")):
            with h5py.File(h5, "r") as f:
                for s in list(f)[:5]:
                    pet, body = f[s]["pet"][:], f[s]["mask"][:] > 0
                    prof["BIC-MAC"].append(pet_profile(pet, body))
                    prof["BIC-MAC + augmentation"].append(pet_profile(pet, body, aug=domain_aug_mid))
                    print(f"    profile BIC-MAC {s}", flush=True)
        for h5 in sorted(Path(args.utrecht_h5_dir).glob("*.h5")):
            with h5py.File(h5, "r") as f:
                for s in f:
                    prof["Utrecht"].append(pet_profile(f[s]["pet"][:], f[s]["mask"][:] > 0))
                    print(f"    profile Utrecht {s}", flush=True)
        cache.write_text(json.dumps(prof))
    if not prof.get("Utrecht"):
        return
    xin = [(a + b) / 2 for a, b in zip(IN_BINS[:-1], IN_BINS[1:])]
    xout = [(a + b) / 2 for a, b in zip(OUT_BINS[:-1], OUT_BINS[1:])]
    style = {"BIC-MAC": (ADULT_C, "-", "BIC-MAC adults (training)"),
             "Utrecht": (CHILD_C, "-", "Utrecht children (target)"),
             "BIC-MAC + augmentation": (ADULT_C, "--", "BIC-MAC after domain augmentation")}
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    summ = {}
    for name in ("BIC-MAC", "BIC-MAC + augmentation", "Utrecht"):
        c, ls, lab = style[name]
        pin = np.array([p[0] for p in prof[name]], float)
        pout = np.array([p[1] for p in prof[name]], float)
        for ax, x, v in ((axes[0], xin, pin), (axes[1], xout, pout)):
            m = np.nanmean(v, 0)
            ax.plot(x, m, color=c, ls=ls, lw=3.5, marker="o", ms=8, label=f"{lab} (n = {len(v)})")
            if ls == "-":
                ax.fill_between(x, np.nanpercentile(v, 25, 0), np.nanpercentile(v, 75, 0), color=c, alpha=0.15, lw=0)
        summ[name] = (np.nanmean(pin, 0), np.nanmean(pout, 0))
    axes[0].set_xlabel("Depth inside the body, from the skin (mm)")
    axes[0].set_ylabel("NAC-PET / 95th percentile in body")
    axes[0].set_title("Inside the body", loc="left")
    axes[1].set_xlabel("Distance outside the body, from the skin (mm)")
    axes[1].set_ylabel("NAC-PET / 95th percentile in body")
    axes[1].set_title("Outside the body (scatter halo)", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    header(fig, "Utrecht NAC-PET is darker in the body centre; the augmentation moves adult PET towards it",
           "Mean NAC-PET signal vs distance to the skin; band = middle 50 % of exams. Without attenuation correction, "
           "deep tissue looks darker, by a scanner-dependent amount.\nDashed = adult PET after the domain augmentation "
           "used in training from v23 on (mid-range settings shown; training draws random, up to stronger, settings).")
    fig.tight_layout(rect=(0, 0.08, 1, 1 - 1.69 / fig.get_size_inches()[1]))
    b, u = summ["BIC-MAC"][0], summ["Utrecht"][0]
    save(fig, args, "08_pet_domain_gap.png",
         "NAC-PET intensity profile versus distance to the skin in both cohorts, and the adult profile after the "
         "domain augmentation used in training.",
         [f"Deepest band: adults {b[-1]:.2f}, Utrecht {u[-1]:.2f}, adults after augmentation "
          f"{summ['BIC-MAC + augmentation'][0][-1]:.2f}.",
          "Without this augmentation, the model sees adult-looking PET in training and Utrecht-looking PET in "
          "practice. Adding it (with the left–right flip) is the v21 → v23 gain."])


# ----------------------------------------------------------------------------- 07 synthetic pairs
def fig_synthetic(args):
    qa = Path(args.ped_h5_dir) / "deform_qa.csv"
    if not qa.is_file():
        return
    rows = list(csv.DictReader(open(qa)))
    ok = [r for r in rows if r["qa_ok"] == "True"]
    pick = sorted(ok, key=lambda r: (float(r["child_age"]), -float(r["dice_body"])))[0]
    pair, adult, child = pick["pair"], pick["adult"], pick["child"]
    ped = load_h5(args.ped_h5_dir, pair, keys=("pet", "ct", "mask"))
    ad = load_h5(args.bigmac_h5_dir, adult, keys=("ct", "mask"))
    ch = load_h5(args.utrecht_h5_dir, child, keys=("pet", "ct", "mask"))
    if ped is None or ad is None or ch is None:
        return
    panels = [(ad, "ct", "1. Adult CT\n(BIC-MAC)"), (ped, "ct", "2. Adult CT deformed\nto the child"),
              (ch, "ct", f"3. Real child CT\n({float(pick['child_age']):.0f} y, Utrecht)"),
              (ped, "pet", "4. Adult PET deformed\n(new training input)"), (ch, "pet", "5. Real child PET\n(for comparison)")]
    if PAPER:
        panels = [(c, k, t) for (c, k, _), t in zip(panels, ("(a) Adult CT", "(b) Deformed CT", "(c) Child CT",
                                                             "(d) Deformed PET", "(e) Child PET"))]
    ys = {id(c): mid_y(c, c["ct"]) for c, _, _ in panels}
    win = body_size([coronal(c["mask"] > 0, ys[id(c)]) for c, _, _ in panels])
    fig, axes = plt.subplots(1, 5, figsize=(20, 8.5))
    for ax, (c, key, title) in zip(axes, panels):
        y = ys[id(c)]
        bm = coronal(c["mask"] > 0, y)
        im = body_window(coronal(c[key], y), bm, win)
        if key == "pet":
            ax.imshow(im, cmap="gray_r", vmax=np.percentile(c["pet"][c["mask"] > 0], 99.5))
        else:
            ax.imshow(im, cmap="gray", vmin=-1000, vmax=1500)
        ax.set_title(title)
        clean_axes(ax)
    ages = sorted({float(r["child_age"]) for r in ok})
    header(fig, f"Synthetic paediatric training data: {len(ok)} adult scans reshaped into children",
           "Each adult's CT, PET and MRI are deformed with the same transform (B-spline + demons) onto the body and lungs "
           "of a Utrecht child, giving a matched PET/CT pair with child anatomy.\n"
           f"{len(ok)} of {len(rows)} pairs passed visual and numeric QA; target ages {ages[0]:.0f}–{ages[-1]:.0f} y. "
           "Used in v24 (20 % of training patches) and v25 (40 %, children under 13 only).", top=0.84)
    save(fig, args, "09_synthetic_paediatric.png",
         f"Example deformation: BIC-MAC {adult} deformed onto Utrecht {child} "
         f"(Dice body {float(pick['dice_body']):.2f}, lungs {float(pick['dice_lung']):.2f}).",
         ["The deformed scans have child anatomy but keep adult tissue properties (e.g. bone density) and adult PET.",
          "A child's own deformed pairs are never used to train the model that evaluates that child."])


# ----------------------------------------------------------------------------- 08-11 Utrecht metrics
def ci95(v, n_boot=2000):
    rng = np.random.default_rng(0)
    m = rng.choice(v, (n_boot, len(v))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def fig_progress(tab, ages, args):
    models = [m for m in args.models if m in tab]
    if len(models) < 2:
        return
    subs = sorted(set.intersection(*[set(tab[m]) for m in models]))
    young = [s for s in subs if ages.get(s, 99) < YOUNG]
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5), sharey=True)
    res = {}
    for ax, (sel, name) in zip(axes, ((subs, f"All exams (n = {len(subs)})"),
                                      (young, f"Children under {YOUNG} (n = {len(young)})"))):
        if not sel:
            continue
        for i, m in enumerate(models):
            v = np.array([tab[m][s]["mae_body"] for s in sel], float)
            lo, hi = ci95(v)
            c = FINAL_C if m == args.final else BASE_C if m == "v18" else OTHER_C
            ax.barh(i, v.mean(), 0.7, color=c, edgecolor=SURFACE)
            ax.plot([lo, hi], [i, i], color=INK2, lw=2)
            ax.annotate(f"{v.mean():.1f}", (hi, i), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=16, fontweight="bold" if m == args.final else "normal")
            res.setdefault(m, []).append(v.mean())
        ax.set_title(name, loc="left")
        ax.set_xlabel("Body MAE (HU), lower is better")
        ax.grid(axis="y", visible=False)
        ref = np.mean([tab["v18"][s]["mae_body"] for s in sel]) if "v18" in tab else None
        if ref:
            ax.axvline(ref, color=BASE_C, lw=1.5, ls="--")
        ax.set_xlim(0, max(r[-1] for r in res.values()) * 1.3)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([f"{m}   {MODEL_INFO.get(m, '')}" for m in models], fontsize=14)
    axes[0].invert_yaxis()
    f, b = res[args.final], res.get("v18", [np.nan, np.nan])
    header(fig, f"Fine-tuning on Utrecht: body error {b[0]:.0f} → {f[0]:.0f} HU; "
                f"the youngest children improve least",
           "Utrecht, PET only. Every exam is predicted by a model that never saw it (out-of-fold for the 20 fold exams, "
           "5-fold ensemble for the 5 test exams).\nBars = mean over exams, line = 95 % bootstrap interval, "
           f"dashed = zero-shot adult model v18. Blue = model chosen as final ({args.final}). "
           "Rows in order of development.", top=0.82)
    pts = [f"All exams: v18 {b[0]:.1f} → {args.final} {f[0]:.1f} HU ({100 * (f[0] - b[0]) / b[0]:+.0f} %).",
           f"Under {YOUNG}: v18 {b[1]:.1f} → {args.final} {f[1]:.1f} HU.",
           "Intervals overlap: with 25 exams, differences of a few HU between versions are within noise; "
           "the paired comparisons (figs. 12 and 13) are more sensitive."]
    save(fig, args, "11_utrecht_model_progress.png",
         "Mean body MAE on the 25 Utrecht exams for every model version, for all exams and for children under 13.", pts)


def fig_per_patient(tab, ages, part, args):
    if "v18" not in tab or args.final not in tab:
        return
    subs = sorted(set(tab["v18"]) & set(tab[args.final]), key=lambda s: (ages.get(s, 99), s))
    a = np.array([tab["v18"][s]["mae_body"] for s in subs], float)
    b = np.array([tab[args.final][s]["mae_body"] for s in subs], float)
    fig, ax = plt.subplots(figsize=(19, 8.5))
    x = np.arange(len(subs))
    for xi, ai, bi in zip(x, a, b):
        ax.annotate("", (xi, bi), (xi, ai), arrowprops=dict(arrowstyle="-|>", lw=2.5, mutation_scale=18,
                                                              color=GOOD_C if bi < ai else BAD_C))
    ax.scatter(x, a, s=110, color=BASE_C, edgecolor=SURFACE, lw=1.5, zorder=3, label="v18, adult model (zero-shot)")
    ax.scatter(x, b, s=110, color=FINAL_C, edgecolor=SURFACE, lw=1.5, zorder=3, label=f"{args.final}, fine-tuned (final)")
    n_young = sum(ages.get(s, 99) < YOUNG for s in subs)
    ax.axvspan(-0.5, n_young - 0.5, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.annotate(f"Under {YOUNG} years", (n_young / 2 - 0.5, 1.0), xycoords=("data", "axes fraction"), ha="center",
                va="bottom", fontsize=15, color=INK2, fontweight="bold")
    test = {s for s in subs if tab[args.final][s]["h5"] == "test.h5"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ages.get(s, '?')}" + ("\n" + part.get(s, {}).get("sex", "") if part else "")
                        + ("\ntest" if s in test else "") for s in subs], fontsize=13)
    ax.set_xlabel("Exams sorted by age (years; sex; 'test' = held-out test set)")
    ax.set_ylabel("Body MAE (HU)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(subs) - 0.4)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.12 * (hi - lo))
    better = int((b < a).sum())
    p = wilcoxon(a, b)
    header(fig, f"The final model is better than the adult model in {better} of {len(subs)} exams",
           f"Utrecht, PET only, body MAE per exam. Arrow from zero-shot v18 (orange) to fine-tuned {args.final} (blue): "
           "green = improved, red = worse.\n"
           f"Mean {a.mean():.1f} → {b.mean():.1f} HU; paired Wilcoxon test {p_text(p)}.", top=0.84)
    worse = [f"{ages.get(s)} y" for s, ai, bi in zip(subs, a, b) if bi >= ai]
    save(fig, args, "13_utrecht_per_patient.png",
         f"Body MAE per Utrecht exam for the adult model v18 and the final model {args.final}.",
         [f"Improved in {better}/{len(subs)} exams; not improved: {', '.join(worse) or 'none'}.",
          f"Mean change {b.mean() - a.mean():+.1f} HU ({p_text(p)})."])


def fig_tissues(tab, ages, args):
    if "v18" not in tab or args.final not in tab:
        return
    subs = sorted(set(tab["v18"]) & set(tab[args.final]))
    fig, axes = plt.subplots(1, len(TISSUES), figsize=(21, 7.5))
    pts = []
    rng = np.random.default_rng(0)
    for ax, (m, name, unit) in zip(axes, TISSUES):
        a = np.array([tab["v18"][s][m] for s in subs], float)
        b = np.array([tab[args.final][s][m] for s in subs], float)
        for i, (v, c) in enumerate(((a, BASE_C), (b, FINAL_C))):
            ax.bar(i, np.nanmean(v), 0.65, color=c, edgecolor=SURFACE)
            ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=18, color=INK, alpha=0.35, lw=0, zorder=3)
        for xi in range(len(subs)):
            ax.plot([0.25, 0.75], [a[xi], b[xi]], color=INK2, alpha=0.15, lw=1)
        fmt = "{:.2f}" if m == "dice_bone" else "{:.0f}"
        top = np.nanmax(np.concatenate([a, b]))
        for i, v in enumerate((a, b)):
            ax.annotate(fmt.format(np.nanmean(v)), (i, top), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=17, fontweight="bold")
        ax.set_ylim(0, top * 1.18)
        ch = (np.nanmean(b) - np.nanmean(a)) / np.nanmean(a) * 100
        p = wilcoxon(a, b)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["v18", args.final])
        ax.set_title(name, loc="left")
        ax.set_ylabel(unit)
        ax.set_xlabel(f"{ch:+.0f} %, {p_text(p)}", fontsize=14, color=INK)
        ax.grid(axis="x", visible=False)
        pts.append(f"{name}: {fmt.format(np.nanmean(a))} → {fmt.format(np.nanmean(b))} ({ch:+.0f} %, {p_text(p)})")
    header(fig, "Fine-tuning mostly fixes the lungs; bone remains the weak point",
           f"Utrecht, PET only, n = {len(subs)} exams. Orange = adult model v18 (zero-shot), blue = final model "
           f"{args.final}; bars = mean, dots = exams.\nMAE: lower is better. Dice: overlap of bone (> 300 HU) between "
           "pseudo-CT and CT, higher is better. Below each panel: relative change and paired Wilcoxon p-value.")
    save(fig, args, "15_utrecht_tissues.png", "Error per tissue on Utrecht, adult model vs final model.", pts)


def fig_error_vs_size(tab, ages, coh, args):
    if args.final not in tab:
        return
    width = {r["subject"]: r["chest_width_cm"] for r in coh if r["cohort"] == "Utrecht"}
    subs = sorted(s for s in tab[args.final] if s in ages)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
    pts = []
    for ax, (xs, name) in zip(axes, (({s: ages[s] for s in subs}, "Age (years)"),
                                     ({s: width.get(s, np.nan) for s in subs}, "Chest width (cm)"))):
        for m, c, lab in (("v18", BASE_C, "v18, adult model"), (args.final, FINAL_C, f"{args.final}, final")):
            if m not in tab:
                continue
            x = np.array([xs[s] for s in subs], float)
            y = np.array([tab[m][s]["mae_body"] for s in subs], float)
            ok = ~np.isnan(x)
            ax.scatter(x[ok], y[ok], s=110, color=c, edgecolor=SURFACE, lw=1.5, alpha=0.95 if m != "v18" else 0.6,
                       label=lab, zorder=3)
            if m == args.final and ok.sum() > 3:
                r, p = stats.spearmanr(x[ok], y[ok])
                k, q = np.polyfit(x[ok], y[ok], 1)
                xx = np.linspace(x[ok].min(), x[ok].max(), 10)
                ax.plot(xx, k * xx + q, color=c, lw=2.5, ls="--")
                ax.annotate(f"{args.final}: Spearman ρ = {r:.2f} ({p_text(p)})", (0.02, 0.03), xycoords="axes fraction",
                            va="bottom", fontsize=15, color=c, fontweight="bold",
                            bbox=dict(fc=SURFACE, ec="none", alpha=0.85))
                pts.append(f"Final-model error vs {name.split(' (')[0].lower()}: ρ = {r:.2f}, {p_text(p)}.")
        ax.set_xlabel(name)
        ax.set_title(f"Error vs {name.split(' (')[0].lower()}", loc="left")
    axes[0].set_ylabel("Body MAE (HU)")
    axes[1].legend(loc="upper right")
    header(fig, "The two smallest children have the largest errors; the overall trend is weak",
           "Utrecht, PET only, one dot per exam. Chest width measured on the CT at the middle of the lungs. "
           "Dashed line = linear trend of the final model.\n"
           "With 25 exams the trend is not significant; it is driven by the 2- and 4-year-olds, the body sizes "
           "least represented in training.")
    save(fig, args, "18_utrecht_error_vs_size.png", "Body MAE versus age and chest width for each Utrecht exam.", pts)


# ----------------------------------------------------------------------------- 12-14 CT numbers
# Bilinear HU -> 511 keV mu (1/cm), Carney et al., Med Phys 33:976 (2006), Table I: per kVp the bone-line
# slope a, intercept b and break point (HU). Below the break point: water-air line 9.6e-5 * (HU + 1000).
CARNEY = {80: (3.64e-5, 6.26e-2, 50), 100: (4.43e-5, 5.44e-2, 52), 110: (4.92e-5, 4.88e-2, 43),
          120: (5.10e-5, 4.71e-2, 47), 130: (5.51e-5, 4.24e-2, 37), 140: (5.64e-5, 4.08e-2, 30)}
KVP_FILE = Path(EVAL, "utrecht_ct_kvp.json")
KVP = json.loads(KVP_FILE.read_text()) if KVP_FILE.is_file() else {}


def hu_to_mu(hu, kvp=120):
    """Linear attenuation at 511 keV (1/cm) from CT HU with the bilinear transform for the CT's kVp."""
    a, b, bp = CARNEY[min(CARNEY, key=lambda k: abs(k - kvp))]
    hu = np.asarray(hu, np.float32)
    return np.where(hu <= bp, 9.6e-5 * (hu + 1000), a * (hu + 1000) + b).clip(min=0).astype(np.float32)


def acf_maps(mu_ct, mu_pr):
    """Relative error of the attenuation correction factor, exp(∫μ), for AP and lateral parallel rays."""
    out = {}
    for name, axis, step_cm in (("ap", 1, VOXEL_MM[1] / 10), ("lat", 0, VOXEL_MM[0] / 10)):
        li_ct = mu_ct.sum(axis=axis) * step_cm
        li_pr = mu_pr.sum(axis=axis) * step_cm
        out[name] = (np.expm1(li_pr - li_ct), li_ct)
    return out


def acf_stats(mu_ct, mu_pr, body, z0, z1):
    """Summary of the ACF error (%) over rays that cross the body, overall and along the body height."""
    res = {}
    for name, (err, li_ct) in acf_maps(mu_ct, mu_pr).items():
        ray = li_ct > 0.1          # at least ~1 cm of water-equivalent tissue
        e = 100 * err[ray]
        res[name] = {"mean_abs": float(np.abs(e).mean()), "mean": float(e.mean()),
                     "p95_abs": float(np.percentile(np.abs(e), 95)), "frac10": float((np.abs(e) > 10).mean())}
        # Profile along z (axis 1 of the 2D map is z for both projections).
        zz = np.arange(err.shape[1])
        frac = (zz - z0) / max(z1 - z0, 1)
        zbin = np.clip((frac * Z_BINS).astype(int), 0, Z_BINS - 1)
        ok = (frac >= 0) & (frac <= 1)
        num = np.bincount(zbin[ok], weights=(np.abs(100 * err) * ray).sum(axis=0)[ok], minlength=Z_BINS)
        den = np.bincount(zbin[ok], weights=ray.sum(axis=0)[ok], minlength=Z_BINS)
        res[name]["z"] = (num / np.maximum(den, 1)).tolist()
    return res


def hu_stats(args, models, subs):
    """Per exam and model: mean CT and pseudo-CT HU per tissue class, plus a pooled 2D histogram."""
    cache = Path(args.cache_dir) / "cache_hu_stats.json"
    old = json.loads(cache.read_text()) if cache.is_file() else {}
    edges = np.arange(-1024, 2001, 16)
    hist_file = Path(args.cache_dir) / "cache_hu_hist.npz"
    hists = dict(np.load(hist_file)) if hist_file.is_file() else {}
    pred_root = Path(args.pred_dir, "utrecht_pred")
    changed = False
    for s in subs:
        need = [m for m in models if (f"{m}|{s}" not in old or not HU_KEYS <= set(old[f"{m}|{s}"]))
                and (pred_root / m / "pet_only" / s / "ct.nii.gz").is_file()]
        if not need:
            continue
        case = load_h5(args.utrecht_h5_dir, s, keys=("ct", "mask"))
        ct, mask = case["ct"].astype(np.float32), case["mask"]
        body = mask > 0
        cls = {"lung": mask == 3, "fat": (mask == 2) & (ct >= -190) & (ct < -30),
               "soft": (mask == 2) & (ct >= -30) & (ct < 150), "bone": mask >= 5, "body": body}
        kvp = KVP.get(s, 120)
        mu_ct = hu_to_mu(np.where(body, ct, -1000), kvp)
        zb = np.where(body.any(axis=(0, 1)))[0]
        z0, z1 = int(zb[0]), int(zb[-1])
        lung = lung_z_range(mask, contiguous=True)
        slab = np.zeros_like(body)
        if lung:
            slab[:, :, lung[0]:lung[1]] = True
        n_z = body.sum(axis=(0, 1)).astype(float)
        for m in need:
            pr = load_pred(pred_root, m, "pet_only", s)
            err_z = np.where(body, np.abs(pr - ct), 0).sum(axis=(0, 1))
            frac = (np.arange(ct.shape[2]) - z0) / max(z1 - z0, 1)
            zbin = np.clip((frac * Z_BINS).astype(int), 0, Z_BINS - 1)
            ok = (frac >= 0) & (frac <= 1)
            z_err = np.bincount(zbin[ok], weights=err_z[ok], minlength=Z_BINS)
            z_n = np.bincount(zbin[ok], weights=n_z[ok], minlength=Z_BINS)
            first = f"{m}|{s}" not in old
            d = {"z": {"err": z_err.tolist(), "n": z_n.tolist(),
                       "lung": [(lung[0] - z0) / max(z1 - z0, 1), (lung[1] - z0) / max(z1 - z0, 1)] if lung else None}}
            mu_pr = hu_to_mu(np.where(body, pr, -1000), kvp)
            for k, sel in cls.items():
                if sel.sum() > 100:
                    d[k] = {"ct": float(ct[sel].mean()), "pct": float(pr[sel].mean()),
                            "mae": float(np.abs(pr[sel] - ct[sel]).mean()), "n": int(sel.sum()),
                            "mu_ct": float(mu_ct[sel].mean()), "mu_pct": float(mu_pr[sel].mean())}
            vox_l = np.prod(VOXEL_MM) / 1e6
            d["lungvol"] = {"ct": float((mask[slab] == 3).sum() * vox_l),
                            "pct": float(((pr < LUNG_HU) & body & slab).sum() * vox_l)}
            d["acf"] = acf_stats(mu_ct, mu_pr, body, z0, z1)
            d["kvp"] = kvp
            old[f"{m}|{s}"] = d
            if first:
                h, _, _ = np.histogram2d(np.clip(ct[body], -1024, 2000), np.clip(pr[body], -1024, 2000),
                                         bins=[edges, edges])
                hists[m] = hists.get(m, 0) + h
                hists[f"n_{m}"] = hists.get(f"n_{m}", 0) + 1
            changed = True
            print(f"    HU {m} {s}", flush=True)
    if changed:
        cache.write_text(json.dumps(old))
        np.savez_compressed(hist_file, **hists)
    return old, hists, edges


def fig_hu_tissue(hu, tab, ages, args):
    models = [m for m in ("v18", args.final) if any(k.startswith(m + "|") for k in hu)]
    if args.final not in models:
        return
    fig, axes = plt.subplots(1, 4, figsize=(21, 8))
    pts = []
    summ = {}
    for ax, (k, name) in zip(axes, HU_CLASSES):
        allv = []
        biases = {}
        for m in models:
            c = FINAL_C if m == args.final else BASE_C
            subs = [key.split("|")[1] for key in hu if key.startswith(m + "|") and k in hu[key]]
            x = np.array([hu[f"{m}|{s}"][k]["ct"] for s in subs])
            y = np.array([hu[f"{m}|{s}"][k]["pct"] for s in subs])
            young = np.array([ages.get(s, 99) < YOUNG for s in subs])
            ax.scatter(x[~young], y[~young], s=90, color=c, edgecolor=SURFACE, lw=1.2, alpha=0.9 if m == args.final else 0.55,
                       zorder=3, label=f"{m}, ≥ {YOUNG} y")
            ax.scatter(x[young], y[young], s=110, marker="D", color=c, edgecolor=INK, lw=1.2,
                       alpha=0.9 if m == args.final else 0.55, zorder=4, label=f"{m}, < {YOUNG} y")
            allv += list(x) + list(y)
            bias = (y - x).mean()
            if m == args.final:
                pts.append(f"{name}: real CT {x.mean():.0f} HU, pseudo-CT {y.mean():.0f} HU (bias {bias:+.0f} HU); "
                           f"v18 bias {np.mean([hu[f'v18|{s}'][k]['pct'] - hu[f'v18|{s}'][k]['ct'] for s in subs if f'v18|{s}' in hu and k in hu[f'v18|{s}']]):+.0f} HU"
                           if "v18" in models else f"{name}: bias {bias:+.0f} HU")
            biases[m] = bias
        lo, hi = min(allv), max(allv)
        pad = 0.1 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=1.5, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_title(name, loc="left", fontsize=16)
        ax.set_xlabel("Real CT, mean HU\nbias: " + ",  ".join(f"{m} {b:+.0f}" for m, b in biases.items()),
                      fontsize=14)
        summ[k] = biases.get(args.final, np.nan)
    axes[0].set_ylabel("Pseudo-CT, mean HU")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    header(fig, f"CT numbers: fat is right ({summ.get('fat', np.nan):+.0f} HU), soft tissue a little low "
                f"({summ.get('soft', np.nan):+.0f}), lungs too dense ({summ.get('lung', np.nan):+.0f}), "
                f"bone far too low ({summ.get('bone', np.nan):+.0f})",
           "Utrecht, PET only. Each marker = mean HU of one tissue in one exam (diamonds = children under 13). "
           "Dashed line = perfect agreement; bias = mean (pseudo-CT − CT).\n"
           "Tissues are taken from the real CT: lungs and internal air from the mask, fat and soft tissue by HU range, "
           "bone from the mask (≥ 200 HU).")
    fig.tight_layout(rect=(0, 0.06, 1, 1 - 1.69 / fig.get_size_inches()[1]))
    save(fig, args, "19_utrecht_hu_per_tissue.png",
         "Mean HU per tissue in the real CT and the pseudo-CT for every Utrecht exam (attenuation-relevant values).",
         pts)


def fig_hu_hist(hists, edges, args):
    models = [m for m in ("v18", args.final) if m in hists]
    if args.final not in models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models) + 1, 8.5), squeeze=False)
    for ax, m in zip(axes[0], models):
        h = hists[m].T
        im = ax.imshow(h, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]], cmap="Blues",
                       norm=LogNorm(vmin=max(h[h > 0].min(), 1), vmax=h.max()), aspect="equal")
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color=BAD_C, lw=1.5, ls="--")
        ax.set_xlabel("Real CT (HU)")
        ax.set_ylabel("Pseudo-CT (HU)")
        ax.set_title(f"{m}  ({MODEL_INFO.get(m, '')})", loc="left", fontsize=15)
        ax.grid(False)
        for v, name in ((-800, "lung"), (-50, "fat / soft"), (900, "bone")):
            ax.annotate(name, (v, edges[-1] - 60), color=INK, fontsize=14, ha="center", va="top", fontweight="bold")
    fig.subplots_adjust(top=0.8, left=0.06, right=0.88, bottom=0.1, wspace=0.25)
    cb = fig.colorbar(im, cax=fig.add_axes([0.9, 0.15, 0.015, 0.6]))
    cb.set_label("Number of voxels (log scale)")
    cb.outline.set_visible(False)
    n = int(hists.get(f"n_{args.final}", 0))
    fig.text(0.01, 0.985, "Voxel by voxel: fine-tuning recovers the lungs; bone is still predicted as soft tissue",
             ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.01, 0.915, f"All body voxels of {n} Utrecht exams pooled (PET only). A perfect pseudo-CT would put every "
                          "voxel on the red dashed line.\nBone (real CT > 300 HU) mostly lands near 0–200 HU in the pseudo-CT. "
                          "Lung voxels move from ~0 HU (v18) towards the diagonal (final model).",
             ha="left", va="top", fontsize=15, color=INK2, linespacing=1.4)
    save(fig, args, "20_utrecht_hu_histogram.png",
         "2D histogram of pseudo-CT versus real CT HU over all body voxels of all Utrecht exams.",
         ["Soft tissue and fat sit on the diagonal; bone above ~500 HU is predicted too low.",
          "This is the expected limit of PET-only input: NAC-PET carries little bone information."])


def per_patient_rows(tab, hu, ages, part, args):
    if args.final not in tab:
        return []
    out = []
    for s in sorted(tab[args.final], key=lambda s: (ages.get(s, 99), s)):
        f, z = tab[args.final][s], tab.get("v18", {}).get(s)
        split = "test" if f["h5"] == "test.h5" else f["h5"].replace(".h5", "").replace("_", " ")
        r = {"subject": s, "exam": short_id(s), "age": ages.get(s, ""), "sex": part.get(s, {}).get("sex", ""),
             "set": split, "v18_mae_body": z["mae_body"] if z else np.nan}
        for m, _, _ in TISSUES:
            r[f"mae_{m.replace('mae_', '')}" if m != "dice_bone" else "dice_bone"] = f[m]
        r["bias_body"] = f["me_body"]
        r["mae_thorax"] = f.get("mae_thorax", np.nan)
        for k, _ in HU_CLASSES:
            d = hu.get(f"{args.final}|{s}", {}).get(k)
            r[f"hu_{k}_ct"] = d["ct"] if d else np.nan
            r[f"hu_{k}_pct"] = d["pct"] if d else np.nan
        out.append(r)
    return out


def write_patient_table(rows, args):
    if not rows:
        return
    keys = list(rows[0])
    with open(Path(args.out_dir) / "utrecht_per_patient.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    cols = [("exam", "Exam", "{}"), ("age", "Age", "{}"), ("sex", "Sex", "{}"), ("set", "Set", "{}"),
            ("v18_mae_body", "Body MAE v18", "{:.1f}"), ("mae_body", f"Body MAE {args.final}", "{:.1f}"),
            ("mae_soft", "Soft tissue MAE", "{:.1f}"), ("mae_air", "Lung MAE", "{:.1f}"),
            ("mae_bone", "Bone MAE", "{:.1f}"), ("dice_bone", "Bone Dice", "{:.3f}"), ("bias_body", "Body bias", "{:+.1f}")]
    for k, _ in HU_CLASSES:
        cols.append((f"hu_{k}", f"{k.capitalize()} HU CT / pCT", None))
    lines = [f"# Utrecht per-patient results ({args.final}, PET only)\n",
             "MAE and bias in HU inside the body; bias = mean(pseudo-CT − CT). Folds: out-of-fold prediction; "
             "test: 5-fold ensemble. HU CT / pCT = mean HU of the tissue in the real CT and in the pseudo-CT.\n",
             "| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    fmt = lambda r, k, f: (f"{r[f'{k}_ct']:.0f} / {r[f'{k}_pct']:.0f}" if f is None and not np.isnan(r[f"{k}_ct"])
                           else "–" if f is None else f.format(r[k]))
    for r in rows:
        lines.append("| " + " | ".join(fmt(r, k, f) for k, _, f in cols) + " |")
    for name, sel in ((f"Mean, under {YOUNG}", lambda r: r["age"] != "" and r["age"] < YOUNG),
                      (f"Mean, {YOUNG} and over", lambda r: r["age"] != "" and r["age"] >= YOUNG),
                      ("Mean, all", lambda r: True)):
        rs = [r for r in rows if sel(r)]
        cells = [f"**{name}** (n = {len(rs)})", "", "", ""]
        for k, _, f in cols[4:]:
            if f is None:
                a, b = np.nanmean([r[f"{k}_ct"] for r in rs]), np.nanmean([r[f"{k}_pct"] for r in rs])
                cells.append(f"{a:.0f} / {b:.0f}")
            else:
                cells.append(f.format(np.nanmean([r[k] for r in rs])))
        lines.append("| " + " | ".join(cells) + " |")
    (Path(args.out_dir) / "utrecht_per_patient.md").write_text("\n".join(lines) + "\n")
    print(f"  {Path(args.out_dir) / 'utrecht_per_patient.md'}")


def fig_patient_table(rows, args):
    if not rows:
        return
    cols = [("exam", "Exam", "{}", None), ("age", "Age", "{}", None), ("sex", "Sex", "{}", None),
            ("set", "Set", "{}", None), ("v18_mae_body", "Body MAE\nv18", "{:.0f}", "low"),
            ("mae_body", f"Body MAE\n{args.final}", "{:.0f}", "low"), ("delta", "Change\nvs v18", "{:+.0f}", "delta"),
            ("mae_soft", "Soft tissue\nMAE", "{:.0f}", "low"), ("mae_air", "Lung\nMAE", "{:.0f}", "low"),
            ("mae_bone", "Bone\nMAE", "{:.0f}", "low"), ("dice_bone", "Bone\nDice", "{:.2f}", "high"),
            ("bias_body", "Body\nbias", "{:+.0f}", "bias")]
    for r in rows:
        r["delta"] = r["mae_body"] - r["v18_mae_body"]
    summary = []
    for name, sel in ((f"Mean < {YOUNG} y", lambda r: r["age"] != "" and r["age"] < YOUNG),
                      (f"Mean ≥ {YOUNG} y", lambda r: r["age"] != "" and r["age"] >= YOUNG), ("Mean, all", lambda r: True)):
        rs = [r for r in rows if sel(r)]
        s = {"exam": name, "age": f"n = {len(rs)}", "sex": "", "set": ""}
        for k, _, _, _ in cols[4:]:
            s[k] = float(np.nanmean([r[k] for r in rs]))
        summary.append(s)
    allr = rows + summary
    n = len(allr)
    fig, ax = plt.subplots(figsize=(21, 0.42 * n + 3.2))
    ax.axis("off")
    widths = [2.6, 0.6, 0.6, 0.9] + [1.0] * (len(cols) - 4)
    xs = np.concatenate([[0], np.cumsum(widths)])
    total = xs[-1]
    ax.set_xlim(0, total)
    ax.set_ylim(n + 1.6, -0.2)
    for j, (k, name, _, _) in enumerate(cols):
        ax.text(xs[j] + widths[j] / 2, 0.6, name, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    ranges = {k: (np.nanmin([r[k] for r in rows]), np.nanmax([r[k] for r in rows])) for k, _, _, kind in cols if kind}
    import matplotlib.colors as mcolors
    good = mcolors.to_rgb(GOOD_C)
    bad = mcolors.to_rgb(BAD_C)
    for i, r in enumerate(allr):
        y = i + 1.6 + (0.4 if i >= len(rows) else 0)
        if i == len(rows):
            ax.plot([0, total], [y - 0.7, y - 0.7], color=INK2, lw=1.5)
        elif i % 2 == 0 and i < len(rows):
            ax.add_patch(plt.Rectangle((0, y - 0.5), total, 1, color=GRID, alpha=0.35, lw=0))
        for j, (k, _, f, kind) in enumerate(cols):
            v = r.get(k, "")
            txt = f.format(v) if isinstance(v, float) and not np.isnan(v) else ("–" if isinstance(v, float) else str(v))
            if kind and isinstance(v, float) and not np.isnan(v) and i < len(rows):
                lo, hi = ranges[k]
                if kind == "delta":
                    t = min(abs(v) / max(abs(lo), abs(hi), 1e-6), 1)
                    col = good if v < 0 else bad
                elif kind == "bias":
                    t = min(abs(v) / max(abs(lo), abs(hi), 1e-6), 1) * 0.8
                    col = bad
                else:
                    t = (v - lo) / (hi - lo + 1e-9)
                    t = 1 - t if kind == "high" else t
                    col = bad
                ax.add_patch(plt.Rectangle((xs[j] + 0.05, y - 0.42), widths[j] - 0.1, 0.84, color=col,
                                           alpha=0.08 + 0.45 * t, lw=0))
            ax.text(xs[j] + widths[j] / 2, y, txt, ha="center", va="center", fontsize=13,
                    fontweight="bold" if i >= len(rows) else "normal", color=INK)
    fig.text(0.01, 0.995, f"All {len(rows)} Utrecht exams: every number of the final model ({args.final})",
             ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.01, 0.995 - 0.75 / fig.get_size_inches()[1], "PET only, HU inside the body. Sorted by age. "
             "Red shading = worse within the column; green/red in 'Change' = better/worse than the adult model v18. "
             "Bias = mean(pseudo-CT − CT).\nFolds = out-of-fold prediction; test = held-out test set (5-fold ensemble).",
             ha="left", va="top", fontsize=14, color=INK2, linespacing=1.4)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.8 / fig.get_size_inches()[1], bottom=0.01)
    save(fig, args, "22_utrecht_patient_table.png",
         "Per-exam metrics of the final model on Utrecht (also in utrecht_per_patient.csv / .md, with HU per tissue).",
         [f"{summary[0]['exam']}: body MAE {summary[0]['mae_body']:.0f} HU vs {summary[1]['mae_body']:.0f} HU for "
          f"{summary[1]['exam'][5:]}.",
          f"Body bias {summary[2]['bias_body']:+.0f} HU on average: the pseudo-CT is slightly "
          f"{'too dense' if summary[2]['bias_body'] > 0 else 'too light'} overall."])


# ----------------------------------------------------------------------------- 15 examples
def example_picks(tab, ages, args):
    """Youngest exam, median and largest body MAE of the final model (chosen by rule, not by hand)."""
    pred_root = Path(args.pred_dir, "utrecht_pred")
    have = [s for s in tab.get(args.final, {}) if (pred_root / args.final / "pet_only" / s / "ct.nii.gz").is_file()
            and (pred_root / "v18" / "pet_only" / s / "ct.nii.gz").is_file()]
    if len(have) < 3:
        return []
    by_mae = sorted(have, key=lambda s: tab[args.final][s]["mae_body"])
    youngest = min(have, key=lambda s: (ages.get(s, 99), s))
    picks = [("Youngest", youngest)]
    for name, s in (("Median error", by_mae[len(by_mae) // 2]), ("Largest error", by_mae[-1])):
        if s not in [p[1] for p in picks]:
            picks.append((name, s))
        else:
            picks.append((name, by_mae[len(by_mae) // 2 + 1] if name == "Median error" else by_mae[-2]))
    return picks


def fig_examples(tab, ages, args):
    pred_root = Path(args.pred_dir, "utrecht_pred")
    picks = example_picks(tab, ages, args)
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 5, figsize=(18, 6.2 * len(picks)))
    ct_kw = dict(cmap="gray", vmin=-1000, vmax=1500)
    for row, (name, s) in zip(axes, picks):
        case = load_h5(args.utrecht_h5_dir, s)
        pet, ct, mask = case["pet"], case["ct"], case["mask"]
        body = mask > 0
        # Coronal slice with the most body: the bbox centre can miss a bent or rotated body.
        y = int(np.argmax(body.sum(axis=(0, 2))))
        # Crop to the body extent so small children are not lost in a mostly empty frame.
        bx = np.where(body.any(axis=(1, 2)))[0]
        bz = np.where(body.any(axis=(0, 1)))[0]
        crop = lambda im: im[ct.shape[2] - 1 - bz[-1]: ct.shape[2] - bz[0], bx[0]: bx[-1] + 1]
        row[0].imshow(crop(coronal(pet, y)), cmap="gray_r", vmax=np.percentile(pet[body], 99.5))
        row[1].imshow(crop(coronal(ct, y)), **ct_kw)
        im = None
        for j, m in enumerate(("v18", args.final)):
            pr = load_pred(pred_root, m, "pet_only", s)
            row[2 + j].imshow(crop(coronal(pr, y)), **ct_kw)
            row[2 + j].set_title(f"{m}: MAE {tab[m][s]['mae_body']:.0f} HU", fontsize=16)
            if m == args.final:
                im = row[4].imshow(crop(coronal(np.where(body, np.abs(pr - ct), 0), y)), cmap=ERR_CMAP, vmin=0, vmax=500)
        first = row is axes[0]
        if first or not PAPER:
            row[0].set_title("NAC-PET (input)", fontsize=16)
            row[1].set_title("Real CT", fontsize=16)
            row[4].set_title(f"|Error| {args.final}", fontsize=16)
        row[0].set_ylabel(f"{name}\n{ages.get(s, '?')} y, {short_id(s).split(' ')[0]}", fontsize=16, color=INK,
                          fontweight="bold")
        for ax in row:
            clean_axes(ax)
        row[0].yaxis.label.set_visible(True)
        cb = fig.colorbar(im, ax=row[4], fraction=0.05, pad=0.02)
        cb.set_label("HU", color=INK2)
        cb.outline.set_visible(False)
    fig.text(0.01, 0.995, "Examples: youngest child, typical case and worst case", ha="left", va="top",
             fontsize=22, fontweight="bold")
    fig.text(0.01, 0.995 - 0.75 / fig.get_size_inches()[1],
             f"Utrecht, PET only. Cases chosen by rule, not by hand: youngest exam, median and "
             f"largest body MAE of {args.final}.\nCoronal slice through the most body tissue. Error map: white = correct, dark blue = 500 HU or more.",
             ha="left", va="top", fontsize=15, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.9 / fig.get_size_inches()[1]))
    save(fig, args, "23_utrecht_examples.png",
         "Pseudo-CTs of the adult model v18 and the final model for three Utrecht exams chosen by rule.",
         ["Most remaining error sits at tissue boundaries, in bone and at the lung edges, where small "
          "misplacements give large HU differences."])


# ----------------------------------------------------------------------------- 03 / 04 method
def _box(ax, x, y, w, h, text, fc, ec=None, fs=13, color=INK, bold=False, ls="-", alpha=1.0):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc,
                                ec=ec or fc, lw=1.8, ls=ls, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def _arrow(ax, x0, y0, x1, y1, color=INK2, lw=2.0, ls="-", rad=0.0):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls,
                                                        mutation_scale=16, shrinkA=0, shrinkB=0,
                                                        connectionstyle=f"arc3,rad={rad}"))


def fig_architecture(args):
    from config.defaults import get_cfg_defaults
    from models.networks import build_model
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.final_config)
    n_par = sum(p.numel() for p in build_model(cfg).parameters())
    f = cfg.MODEL.NUM_FILTERS
    d, h, w = cfg.TRAIN.PATCH_SIZE
    fig, ax = plt.subplots(figsize=(20, 10.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    pet_c, mri_c, fuse_c, dec_c, cond_c = "#cde2fb", "#e4e3de", "#fbe3c4", "#cdeee0", "#f3d7e5"
    # PET encoder
    _box(ax, 0.1, 6.9, 2.0, 1.3, f"NAC-PET patch\n{d}×{h}×{w}", "#2a78d6", color="white", bold=True)
    _box(ax, 2.8, 6.9, 2.3, 1.3, f"Conv block\n{f} ch, full res.", pet_c)
    _box(ax, 5.8, 6.9, 2.3, 1.3, f"Conv block\n{2 * f} ch, ½ res.", pet_c)
    _box(ax, 8.8, 6.9, 2.3, 1.3, f"Bottleneck\n{4 * f} ch, ¼ res.", pet_c)
    for x0, x1 in ((2.1, 2.8), (5.1, 5.8), (8.1, 8.8)):
        _arrow(ax, x0, 7.55, x1, 7.55)
    ax.text(5.45, 6.55, "stride-2 conv", ha="center", fontsize=12, color=INK2)
    ax.text(8.45, 6.55, "stride-2 conv", ha="center", fontsize=12, color=INK2)
    ax.text(0.1, 8.55, "PET encoder" if PAPER else "PET encoder (trained)", fontsize=15, fontweight="bold", color="#2a78d6")
    # MRI encoder
    _box(ax, 0.1, 4.6, 2.0, 1.3, "MRI patch\n(Dixon, 2 ch)", "#8a8984", color="white", bold=True)
    _box(ax, 2.8, 4.6, 2.3, 1.3, f"Conv block\n{f} ch", mri_c)
    _box(ax, 5.8, 4.6, 2.3, 1.3, f"Conv block\n{2 * f} ch", mri_c)
    _box(ax, 8.8, 4.6, 2.3, 1.3, f"Bottleneck\n{4 * f} ch", mri_c)
    for x0, x1 in ((2.1, 2.8), (5.1, 5.8), (8.1, 8.8)):
        _arrow(ax, x0, 5.25, x1, 5.25)
    if not PAPER:
        ax.text(0.1, 4.1, "MRI encoder: input set to zero and frozen for Utrecht (no MRI)", fontsize=14,
                color=INK2, style="italic")
    # fusion and conditioning
    _box(ax, 11.8, 5.6, 2.2, 1.6, f"Concatenate\nPET + MRI\n{8 * f} ch", fuse_c)
    _arrow(ax, 11.1, 7.55, 11.8, 6.6)
    _arrow(ax, 11.1, 5.25, 11.8, 6.2)
    _box(ax, 10.2, 2.9, 2.5, 1.3, "Topogram (2D)\n→ spatial gate" if PAPER else "CT topogram (2D)\n→ spatial gate\n(zero for Utrecht)",
         cond_c, ls="--")
    _arrow(ax, 11.45, 4.2, 12.5, 5.6, ls="--")
    _box(ax, 13.1, 2.9, 2.6, 1.3, "Body coordinates\n(x, y, z of patch)\n→ FiLM (γ·x + β)", cond_c)
    _arrow(ax, 14.4, 4.2, 13.3, 5.6)
    # decoder
    _box(ax, 14.8, 5.6, 2.2, 1.6, f"Up ×2 + skip\n{2 * f} ch, ½ res.", dec_c)
    _box(ax, 17.6, 5.6, 2.2, 1.6, f"Up ×2 + skip\n{f} ch, full res.", dec_c)
    _arrow(ax, 14.0, 6.4, 14.8, 6.4)
    _arrow(ax, 17.0, 6.4, 17.6, 6.4)
    _box(ax, 17.6, 2.9, 2.2, 1.6, "1×1×1 conv\n→ pseudo-CT\npatch (HU)", "#eb6834", color="white", bold=True)
    _arrow(ax, 18.7, 5.6, 18.7, 4.5)
    # skips: straight lines above the diagram
    for x_from, x_to in ((3.95, 18.7), (6.95, 15.9)):
        y_top = 9.55 if x_from < 5 else 9.0
        ax.plot([x_from, x_from, x_to], [8.2, y_top, y_top], color="#2a78d6", ls=":", lw=2.2)
        _arrow(ax, x_to, y_top, x_to, 7.2, color="#2a78d6", ls=":", lw=2.2)
    ax.text(11.3, 9.65, "skip connections" if PAPER else "skip connections (PET + MRI features at the same resolution)", fontsize=13,
            color="#2a78d6", ha="center", va="bottom")
    # notes
    notes = (f"• 3D U-Net with two encoders, {n_par / 1e6:.2f} M parameters; GroupNorm + LeakyReLU in every block.\n"
             "• Loss: smooth-L1 weighted by tissue (lungs and dense bone ×10, bone ×3) and by CT intensity,\n"
             "   + 0.1 × 2D projection + 0.3 × gradient + 0.2 × SSIM.\n"
             "• Inference: overlapping patches over the whole body, blended with a Hann window;\n"
             "   test exams: average of the 5 fold models.")
    if PAPER:
        ax.set_ylim(2.7, 10.2)
    else:
        ax.text(0.1, 2.75, notes, fontsize=15, va="top", color=INK, linespacing=1.5)
    fig.text(0.01, 0.985, "Network: a 3D U-Net that turns NAC-PET patches into pseudo-CT", ha="left", va="top",
             fontsize=22, fontweight="bold")
    fig.text(0.01, 0.935, "Same architecture for every version (v12 → v25); versions differ only in training data, "
             "sampling and augmentation. Grey = branch unused in Utrecht.", ha="left", va="top", fontsize=15,
             color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
    save(fig, args, "03_architecture.png",
         f"MultimodalPCTUNet: PET and MRI encoders, fusion, topogram gate, coordinate FiLM and decoder "
         f"({n_par / 1e6:.2f} M parameters).",
         ["The model was built for PET + MRI + topogram (challenge); for Utrecht only the PET path and the body "
          "coordinates carry information.",
          "Body coordinates tell each patch where it sits in the body, which helps place lungs and bones."])


def fig_training_scheme(args):
    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_axes([0.01, 0.66, 0.98, 0.3] if PAPER else [0.01, 0.47, 0.98, 0.39])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    steps = [("Challenge model (v12)", "BIC-MAC, 75 adults\nPET + MRI + topogram\nbody MAE 84 HU", "#e4e3de"),
             ("Stage A: robust to missing inputs (v18)", "BIC-MAC; MRI removed in 50 %\nand topogram in 30 % of patches\n"
              "PET-only MAE 437 → 101 HU", "#cde2fb"),
             ("Stage B: fine-tuning on children", "Utrecht 25 exams, PET only\n+ BIC-MAC (+ synthetic) patches\n"
              "domain aug., lung/young sampling", "#cdeee0"),
             ("Evaluation (B4)", "every Utrecht exam predicted by\na model that never saw it\n(20 out-of-fold + 5 test)",
              "#fbe3c4")]
    if PAPER:
        steps = [("v12: challenge model", "BIC-MAC\nPET + MRI + topogram", "#e4e3de"),
                 ("v18: missing inputs", "BIC-MAC\nMRI / topogram dropout", "#cde2fb"),
                 ("v20–v26: fine-tuning", "Utrecht (+ BIC-MAC)\nPET only", "#cdeee0"),
                 ("Evaluation", "Out-of-fold (folds)\nensemble (test)", "#fbe3c4")]
    for i, (t, body, c) in enumerate(steps):
        x = 0.2 + i * 5.0
        by, bh = (2.0, 1.5) if PAPER else (0.6, 2.9)
        ax.text(x + 2.1, 3.85, t, ha="center", fontsize=15, fontweight="bold")
        _box(ax, x, by, 4.2, bh, body, c, fs=15)
        if i < 3:
            _arrow(ax, x + 4.25, by + bh / 2, x + 4.95, by + bh / 2, lw=3)
    if PAPER:
        ax.set_ylim(1.8, 4.2)
    # cross-validation grid
    ax2 = fig.add_axes([0.06, 0.04, 0.6, 0.56] if PAPER else [0.06, 0.04, 0.6, 0.38])
    cols = [f"fold {k}" for k in range(5)] + ["test"]
    tr, va, te = "#cde2fb", "#eb6834", "#8a8984"
    for r in range(5):
        for c in range(6):
            col = te if c == 5 else va if c == r else tr
            ax2.add_patch(plt.Rectangle((c, 4 - r), 0.95, 0.85, color=col))
            txt = "ensemble" if c == 5 else "validate" if c == r else "train"
            if PAPER:
                txt = {"ensemble": "E", "validate": "V", "train": "T"}[txt]
            ax2.text(c + 0.475, 4 - r + 0.42, txt, ha="center", va="center", fontsize=18 if PAPER else 13,
                     fontweight="bold" if PAPER else "normal", color="white" if col != tr else INK)
    ax2.set_xlim(-1.4, 6)
    ax2.set_ylim(-0.1, 5.2)
    ax2.axis("off")
    for c, name in enumerate(cols):
        ax2.text(c + 0.475, 5.0, name + ("\n(5 exams)" if c == 5 else "\n(4 exams)"), ha="center", va="bottom",
                 fontsize=13)
    for r in range(5):
        ax2.text(-0.1, 4 - r + 0.42, f"Model {r}", ha="right", va="center", fontsize=14, fontweight="bold")
    if PAPER:
        ax2.text(6.3, 2.5, "T  train\nV  validation and\n    early stopping\nE  ensemble of the\n    5 fold models",
                 fontsize=16, va="center", ha="left", linespacing=1.5)
        ax2.set_xlim(-1.4, 9.5)
        save(fig, args, "04_training_scheme.png", "Training stages and the 5-fold design on Utrecht.")
        return
    fig.text(0.68, 0.38, "Rules that keep the evaluation honest", fontsize=16, fontweight="bold", va="top")
    fig.text(0.68, 0.33,
             "• Model k is validated and early-stopped on Utrecht\n   fold k, and continues from the same fold of the\n"
             "   previous version (warm start).\n"
             "• The 5 test exams are never used for training\n   or model choice; they are predicted by the\n"
             "   average of the 5 models.\n"
             "• BIC-MAC patches are training-only.\n"
             "• Synthetic pairs built from fold-k children are\n   excluded from model k.",
             fontsize=14, va="top", linespacing=1.4)
    fig.text(0.01, 0.985, "Training scheme: adult model first, then 5-fold fine-tuning on the children",
             ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.01, 0.935, "Top: the stages. Bottom: which Utrecht exams each of the 5 fold models trains on, "
             "validates on, and the held-out test set.", ha="left", va="top", fontsize=15, color=INK2)
    save(fig, args, "04_training_scheme.png",
         "Training stages and the 5-fold cross-validation design on Utrecht.",
         ["All Utrecht numbers in the following figures are out-of-fold: no exam is predicted by a model that "
          "trained on it."])


# ----------------------------------------------------------------------------- 10 versions table
VERSION_ROWS = [
    # version, starts from, patch mix (Utrecht / BIC-MAC / synthetic), what changes
    ("v12", "–", "BIC-MAC only, PET + MRI + topogram", "Challenge model, never trained without MRI"),
    ("v18", "v12", "BIC-MAC only", "MRI dropout 50 %, topogram dropout 30 %, body coordinates"),
    ("v20", "v18", "70 / 30 / 0", "Fine-tuning on Utrecht, PET only; LR 2e-5 (PET encoder ×0.1)"),
    ("v21", "v18", "70 / 30 / 0", "40 % patches centred in lungs, 40 % from children < 13; LR 5e-5 (×0.3)"),
    ("v22", "v18", "70 / 30 / 0", "v21 + adult scans shrunk 0.65–0.9× + weight averaging (EMA)"),
    ("v23", "v21", "70 / 30 / 0", "v21 + NAC-PET domain augmentation (p 0.7), left–right flip"),
    ("v24", "v21", "70 / 10 / 20", "v23 + synthetic paediatric pairs (all ages)"),
    ("v25", "v23", "50 / 10 / 40", "v24 with only < 13-year synthetic pairs, domain aug. always on them"),
    ("v18+v23", "–", "–", "Average of the v18 and v23 predictions (no training)"),
]


def fig_versions(tab, ages, zs_rows, args):
    zs = {r["subject"]: r["mae_body"] for r in zs_rows if r["label"] == "v12" and r["regime"] == "pet_only"}
    rows = []
    for v, start, mix, change in VERSION_ROWS:
        res = tab.get(v) if v != "v12" else ({s: {"mae_body": x} for s, x in zs.items()} if zs else None)
        if v not in ("v12", "v18") and not res:
            continue
        all_m = np.mean([r["mae_body"] for r in res.values()]) if res else np.nan
        yng = [r["mae_body"] for s, r in (res or {}).items() if ages.get(s, 99) < YOUNG]
        rows.append((v, start, mix, change, all_m, np.mean(yng) if yng else np.nan))
    fig, ax = plt.subplots(figsize=(21, 0.75 * len(rows) + 2.6))
    ax.axis("off")
    heads = ["Version", "Starts\nfrom", "Training patches\nUtrecht / BIC-MAC / synth.", "What changes",
             "Body MAE\nall (HU)", f"Body MAE\n< {YOUNG} y (HU)"]
    widths = [1.3, 1.1, 3.0, 8.6, 1.5, 1.5]
    xs = np.concatenate([[0], np.cumsum(widths)])
    ax.set_xlim(0, xs[-1])
    ax.set_ylim(len(rows) + 0.4, -1.2)
    for j, hname in enumerate(heads):
        ax.text(xs[j] + (widths[j] / 2 if j != 3 else 0.1), -0.55, hname, ha="center" if j != 3 else "left",
                va="center", fontsize=14, fontweight="bold")
    ax.plot([0, xs[-1]], [0.0 - 0.1, -0.1], color=INK2, lw=1.5)
    finite = [r[4] for r in rows if not np.isnan(r[4]) and r[0] != "v12"]
    best = min(finite) if finite else np.nan
    for i, (v, start, mix, change, a, y) in enumerate(rows):
        yy = i + 0.45
        if v == args.final:
            ax.add_patch(plt.Rectangle((0, yy - 0.45), xs[-1], 0.9, color="#cde2fb", lw=0))
        elif i % 2:
            ax.add_patch(plt.Rectangle((0, yy - 0.45), xs[-1], 0.9, color=GRID, alpha=0.4, lw=0))
        cells = [v, start, mix, change, "–" if np.isnan(a) else f"{a:.1f}", "–" if np.isnan(y) else f"{y:.1f}"]
        for j, c in enumerate(cells):
            ax.text(xs[j] + (widths[j] / 2 if j != 3 else 0.1), yy, c, ha="center" if j != 3 else "left",
                    va="center", fontsize=14, fontweight="bold" if (j == 0 or (j == 4 and a == best)) else "normal")
    fig.text(0.01, 0.99, "Model versions: what changed at each step, and the result on Utrecht",
             ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.01, 0.99 - 0.65 / fig.get_size_inches()[1],
             f"Same network for all versions. Body MAE: mean over the 25 Utrecht exams, PET only, out-of-fold "
             f"(v12, v18: zero-shot). Blue row = final model ({args.final}).",
             ha="left", va="top", fontsize=15, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.3 / fig.get_size_inches()[1], bottom=0.01)
    save(fig, args, "10_model_versions.png", "Summary of every model version and its result.",
         [f"{v}: {a:.1f} HU overall, {y:.1f} HU under {YOUNG}" for v, _, _, _, a, y in rows if not np.isnan(a)])


# ----------------------------------------------------------------------------- 12 change effects
CHANGES = [("v18", "v20", "Fine-tune on Utrecht (v18 → v20)"),
           ("v20", "v21", "+ lung / young sampling, higher LR (v20 → v21)"),
           ("v21", "v22", "+ shrunk adults, EMA (v21 → v22)"),
           ("v21", "v23", "+ PET domain aug., L–R flip (v21 → v23)"),
           ("v23", "v24", "+ synthetic pairs 20 % (v23 → v24)"),
           ("v23", "v25", "+ synthetic young pairs 40 % (v23 → v25)"),
           ("v23", "v26", "+ anatomical masks (v23 → v26)"),
           ("v23", "v18+v23", "Average with v18 (v23 → v18+v23)")]
CHANGES_SHORT = {"v20": "v18 → v20  fine-tuning", "v21": "v20 → v21  sampling", "v22": "v21 → v22  shrink + EMA",
                 "v23": "v21 → v23  domain aug.", "v24": "v23 → v24  synthetic", "v25": "v23 → v25  synthetic < 13 y",
                 "v26": "v23 → v26  anatomical masks", "v18+v23": "v23 → v18+v23  average"}


def fig_change_effects(tab, ages, args):
    steps = [(a, b, n) for a, b, n in CHANGES if a in tab and b in tab]
    if args.final != "v23" and "v18" in tab and args.final in tab:
        steps.append(("v23", args.final, f"Final choice (v23 → {args.final})")) if "v23" in tab and args.final != "v18+v23" \
            and not any(b == args.final for _, b, _ in steps) else None
    steps.append(("v18", args.final, f"Total (v18 → {args.final})"))
    if len(steps) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(20, 0.85 * len(steps) + 3.6), sharey=True)
    pts = []
    rng = np.random.default_rng(0)
    for ax, (young, name) in zip(axes, ((False, "All exams (n = 25)"), (True, f"Children under {YOUNG} (n = 6)"))):
        for i, (a, b, lab) in enumerate(steps):
            subs = sorted(s for s in set(tab[a]) & set(tab[b]) if (ages.get(s, 99) < YOUNG) == young or not young)
            dlt = np.array([tab[b][s]["mae_body"] - tab[a][s]["mae_body"] for s in subs], float)
            boot = rng.choice(dlt, (4000, len(dlt))).mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            col = GOOD_C if hi < 0 else BAD_C if lo > 0 else OTHER_C
            ax.plot([lo, hi], [i, i], color=col, lw=4, solid_capstyle="round")
            ax.scatter(dlt, np.full(len(dlt), i) + rng.uniform(-0.18, 0.18, len(dlt)), s=22, color=INK, alpha=0.3,
                       lw=0, zorder=2)
            ax.scatter([dlt.mean()], [i], s=170, color=col, edgecolor=INK, lw=1.2, zorder=3)
            note = (f"{dlt.mean():+.1f} ({int((dlt < 0).sum())}/{len(dlt)})" if PAPER
                    else f"{dlt.mean():+.1f} HU   better in {int((dlt < 0).sum())}/{len(dlt)}")
            ax.annotate(note, (1.0, i),
                        xycoords=("axes fraction", "data"), xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=13, fontweight="bold" if b == args.final else "normal",
                        bbox=dict(fc=SURFACE, ec="none", alpha=0.85))
            if not young:
                pts.append(f"{lab}: {dlt.mean():+.1f} HU (95 % CI {lo:+.1f} to {hi:+.1f}), better in "
                           f"{int((dlt < 0).sum())}/{len(dlt)}")
        ax.axvline(0, color=INK2, lw=1.5)
        ax.set_title(name, loc="left")
        ax.set_xlabel("Δ body MAE (HU)" if PAPER else "Change in body MAE per exam (HU); negative = better")
        ax.grid(axis="y", visible=False)
        xl = ax.get_xlim()
        ax.set_xlim(xl[0], xl[1] + 0.55 * (xl[1] - xl[0]))
    axes[0].set_yticks(range(len(steps)))
    axes[0].set_yticklabels([((f"{a} → {b}  total" if n.startswith("Total") else CHANGES_SHORT.get(b, n)) if PAPER else n)
                             for a, b, n in steps], fontsize=14)
    axes[0].invert_yaxis()
    header(fig, "What each change bought: the domain augmentation and the targeted sampling helped most",
           "Each row compares two versions on the same exams. Big dot = mean change, bar = 95 % bootstrap interval, "
           "small dots = exams.\nGreen = clearly better (interval below 0), red = clearly worse, grey = within noise.")
    save(fig, args, "12_utrecht_change_effects.png",
         "Paired change in body MAE for each modification, all exams and children under 13.", pts)


# ----------------------------------------------------------------------------- 14 model variation
def fig_model_variation(tab, ages, part, args):
    models = [m for m in args.models if m in tab]
    if len(models) < 3:
        return
    subs = sorted(set.intersection(*[set(tab[m]) for m in models]), key=lambda s: (ages.get(s, 99), s))
    M = np.array([[tab[m][s]["mae_body"] for m in models] for s in subs], float)
    fig, ax = plt.subplots(figsize=(4 + 1.6 * len(models), 0.5 * len(subs) + 3.8))
    im = ax.imshow(M, cmap="Reds", aspect="auto", vmin=np.percentile(M, 2), vmax=np.percentile(M, 98))
    for i in range(len(subs)):
        b = int(np.argmin(M[i]))
        for j in range(len(models)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=12,
                    fontweight="bold" if j == b else "normal",
                    color="white" if M[i, j] > np.percentile(M, 80) else INK)
        ax.add_patch(plt.Rectangle((b - 0.48, i - 0.46), 0.96, 0.92, fill=False, ec=FINAL_C, lw=2.5))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=14, fontweight="bold")
    ax.xaxis.tick_top()
    test = {s for s in subs if tab[models[-1]][s]["h5"] == "test.h5"}
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([f"{ages.get(s, '?')} y {part.get(s, {}).get('sex', '')}" + (" (test)" if s in test else "")
                        for s in subs], fontsize=12)
    n_young = sum(ages.get(s, 99) < YOUNG for s in subs)
    ax.axhline(n_young - 0.5, color=INK, lw=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    wins = {m: int(sum(np.argmin(M, 1) == j)) for j, m in enumerate(models)}
    ax.set_xlabel("Mean: " + "   ".join(f"{m} {M[:, j].mean():.1f}" for j, m in enumerate(models))
                  + "\nBest for: " + "   ".join(f"{m} {wins[m]}" for m in models), fontsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Body MAE (HU)")
    cb.outline.set_visible(False)
    header(fig, "No single version is best for every child",
           f"Body MAE (HU) per exam (rows, by age; line = {YOUNG} y) and version (columns).\n"
           "Blue frame = best version for that exam. The youngest children change little\n"
           "between versions; teenagers gain most from fine-tuning.")
    save(fig, args, "14_utrecht_model_variation.png", "Per-exam body MAE for all model versions.",
         [f"Number of exams where each version is best: {', '.join(f'{m} {n}' for m, n in wins.items())}."])


# ----------------------------------------------------------------------------- 16 regions
def fig_regions(tab, args):
    models = [m for m in args.models if m in tab]
    regions = [("mae_thorax", "Thorax (lung slab)"), ("mae_rest", "Outside the thorax"),
               ("mae_core", "Body interior (skin edge removed)")]
    if len(models) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), sharey=False)
    pts = []
    for ax, (k, name) in zip(axes, regions):
        for i, m in enumerate(models):
            v = np.array([r[k] for r in tab[m].values()], float)
            v = v[~np.isnan(v)]
            c = FINAL_C if m == args.final else BASE_C if m == "v18" else OTHER_C
            ax.bar(i, v.mean(), 0.7, color=c, edgecolor=SURFACE)
            ax.annotate(f"{v.mean():.0f}", (i, v.mean()), xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=13, fontweight="bold" if m == args.final else "normal")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=0, fontsize=12)
        ax.set_title(name, loc="left")
        ax.set_ylabel("MAE (HU)")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        a = np.mean([r[k] for r in tab["v18"].values()])
        b = np.mean([r[k] for r in tab[args.final].values()])
        pts.append(f"{name}: v18 {a:.0f} → {args.final} {b:.0f} HU")
    header(fig, "Most of the gain is in the thorax; the rest of the body changes little",
           "Utrecht, PET only, mean over 25 exams. Thorax = axial slab containing the lungs (from the real CT). "
           "Body interior = body eroded by 3 voxels,\nso it excludes the skin boundary where small shifts give large "
           "errors. Orange = v18, blue = final model.")
    save(fig, args, "16_utrecht_regions.png", "Body MAE by region for every version.", pts)


# ----------------------------------------------------------------------------- 17 along the body
def fig_along_body(hu, ages, args):
    models = [m for m in ("v18", args.final) if any(k.startswith(m + "|") and "z" in hu[k] for k in hu)]
    if args.final not in models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5), sharex=True)
    x = (np.arange(Z_BINS) + 0.5) / Z_BINS * 100
    pts = []
    for ax, (young, name) in zip(axes, ((False, "All exams"), (True, f"Children under {YOUNG}"))):
        lungs = []
        for m in models:
            keys = [k for k in hu if k.startswith(m + "|") and "z" in hu[k]
                    and ((ages.get(k.split("|")[1], 99) < YOUNG) == young or not young)]
            prof = np.array([np.array(hu[k]["z"]["err"]) / np.maximum(hu[k]["z"]["n"], 1) for k in keys])
            nz = np.array([np.array(hu[k]["z"]["n"]) > 0 for k in keys])
            prof = np.where(nz, prof, np.nan)
            mean = np.nanmean(prof, 0)
            c = FINAL_C if m == args.final else BASE_C
            ax.plot(mean, x, color=c, lw=3.5, label=f"{m} (n = {len(keys)})")
            ax.fill_betweenx(x, np.nanpercentile(prof, 25, 0), np.nanpercentile(prof, 75, 0), color=c, alpha=0.15,
                             lw=0)
            lungs += [hu[k]["z"]["lung"] for k in keys if hu[k]["z"]["lung"]]
            if not young:
                top = mean[int(0.8 * Z_BINS):].mean()
                mid = mean[int(0.55 * Z_BINS):int(0.8 * Z_BINS)].mean()
                pts.append(f"{m}: mean MAE {mid:.0f} HU at 55–80 % of body height (chest), {top:.0f} HU in the top 20 % "
                           "(head and neck).")
        if lungs:
            lo, hi = np.median([l[0] for l in lungs]) * 100, np.median([l[1] for l in lungs]) * 100
            ax.axhspan(lo, hi, color=GRID, alpha=0.6, lw=0, zorder=0)
            ax.text(ax.get_xlim()[1], (lo + hi) / 2, "lungs\n(median)", ha="right", va="center", fontsize=13,
                    color=INK2)
        ax.set_ylabel("Position along the body (% of height; 0 = feet, 100 = top of head)")
        ax.set_xlabel("Mean absolute error in the slice (HU)")
        ax.set_title(name, loc="left")
        ax.legend(loc="lower right")
    header(fig, "Fine-tuning lowers the error in the lungs and abdomen; head top and feet stay hardest",
           "Utrecht, PET only. Each axial slice: mean |pseudo-CT − CT| inside the body; slices are placed by their "
           "relative height so children and teenagers align.\nLine = mean over exams, band = middle 50 %. Grey band = "
           "median position of the lungs.")
    save(fig, args, "17_utrecht_error_along_body.png", "Slice-wise error along the body height, v18 vs final.", pts)


# ----------------------------------------------------------------------------- 21 training / folds
def fig_training_curves(tab, args):
    if args.final not in tab:
        return
    expr = next(iter(tab[args.final].values()))["expr"].split("+")[0]
    curves = parse_train_logs(Path(".").glob("logs/train_*.log"))
    folds = sorted(int(e.rsplit("_", 1)[1]) for e in curves if e.startswith(expr + "_Fold_"))
    if not folds:
        return
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    best_ep = []
    for k in folds:
        c = curves[f"{expr}_Fold_{k}"]
        pts_ = [(e, c[e]["sin_mri"]) for e in sorted(c) if "sin_mri" in c[e]]
        if not pts_:
            continue
        x, y = zip(*pts_)
        ax.plot(x, y, color=pal[k % 5], lw=2.5, label=f"Fold {k}")
        b = int(np.argmin(y))
        ax.plot(x[b], y[b], "o", ms=12, color=pal[k % 5], mec=INK, mew=1.5, zorder=3)
        best_ep.append(x[b])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (HU, patches)")
    ax.set_title(f"{args.final}: validation curves (dot = checkpoint kept)", loc="left")
    ax.legend(loc="upper right", ncol=5)
    ax = axes[1]
    models = [m for m in args.models if m in tab and m != "v18+v23"]
    for k in range(5):
        ys = []
        for m in models:
            v = [r["mae_body"] for r in tab[m].values() if r["h5"] == f"fold_{k}.h5"]
            ys.append(np.mean(v) if v else np.nan)
        ax.plot(range(len(models)), ys, "-o", color=pal[k], lw=2.5, ms=9, label=f"Fold {k}")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Body MAE, whole volume (HU)")
    ax.set_title("Out-of-fold result per fold (4 exams each)", loc="left")
    ax.grid(axis="x", visible=False)
    header(fig, "Training converges early; fold 2 (the youngest children) stays the hardest",
           "Left: validation MAE on Utrecht patches during fine-tuning (every 2 epochs); early stopping after 30 epochs "
           "without improvement.\nRight: mean body MAE of each fold's 4 exams for every version. Fold 2 holds the "
           "2- and 4-year-olds.")
    save(fig, args, "21_utrecht_training_and_folds.png", "Validation curves of the final model and per-fold results.",
         [f"Best checkpoints at epochs {', '.join(str(e) for e in best_ep)} (of up to 80): longer training would not help.",
          "Fold 2, with the 2- and 4-year-olds, has the highest error for every version."])


# ----------------------------------------------------------------------------- 24-28 clinical view
def _levels(mask):
    """Axial levels (z) for thorax, upper abdomen and pelvis, found from the CT mask."""
    body = mask > 0
    zb = np.where(body.any(axis=(0, 1)))[0]
    z0, z1 = int(zb[0]), int(zb[-1])
    lung = lung_z_range(mask, contiguous=True)
    out = {}
    if lung:
        lo, hi = lung
        out["Thorax"] = (lo + hi) // 2
        out["Upper abdomen"] = max(z0, lo - (hi - lo) // 2)
    # Pelvis: slice with most bone between 25 % and 55 % of the body height (hips).
    a, b = z0 + int(0.25 * (z1 - z0)), z0 + int(0.55 * (z1 - z0))
    bone = (mask[:, :, a:b] >= 5).sum(axis=(0, 1))
    out["Pelvis"] = a + int(np.argmax(bone)) if bone.size else (z0 + z1) // 2
    return out


def axial(vol, z):
    """Axial slice, anterior up."""
    return np.rot90(vol[:, :, z])


def _crop2d(msk, margin=8):
    r, c = np.where(msk)
    if not r.size:
        return slice(None), slice(None)
    return (slice(max(r.min() - margin, 0), r.max() + margin + 1), slice(max(c.min() - margin, 0), c.max() + margin + 1))


def fig_attenuation(hu, tab, ages, args):
    models = [m for m in ("v18", args.final) if any(k.startswith(m + "|") and "acf" in hu[k] for k in hu)]
    if args.final not in models:
        return
    picks = example_picks(tab, ages, args)
    sub = dict(picks).get("Median error")
    case = load_h5(args.utrecht_h5_dir, sub, keys=("ct", "mask")) if sub else None
    fig = plt.figure(figsize=(21, 14))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.3)
    top = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 0.07, 0.45, 1.5], wspace=0.1)
    bot = outer[1].subgridspec(1, 3, width_ratios=[1, 0.25, 2.4], wspace=0.05)
    pts = []
    if case is not None:
        ct, mask = case["ct"].astype(np.float32), case["mask"]
        body = mask > 0
        mu_ct = hu_to_mu(np.where(body, ct, -1000), KVP.get(sub, 120))
        shadow = body.any(axis=1)
        crop = _crop2d(np.rot90(shadow))
        for j, m in enumerate(models):
            pr = load_pred(Path(args.pred_dir, "utrecht_pred"), m, "pet_only", sub)
            err, li = acf_maps(mu_ct, hu_to_mu(np.where(body, pr, -1000), KVP.get(sub, 120)))["ap"]
            img = np.where(li > 0.1, 100 * err, np.nan)
            ax = fig.add_subplot(top[0, j])
            im = ax.imshow(np.rot90(img)[crop], cmap="RdBu_r", vmin=-30, vmax=30)
            e = 100 * err[li > 0.1]
            ax.set_title(f"{m}: mean |ACF error| {np.abs(e).mean():.1f} %\n(example, {ages.get(sub, '?')} y)",
                         fontsize=15)
            clean_axes(ax)
        cb = fig.colorbar(im, cax=fig.add_subplot(top[0, 2]))
        cb.set_label("ACF error (%); red = over-correction", fontsize=13)
        cb.outline.set_visible(False)
    # Profile along the body.
    ax = fig.add_subplot(top[0, 4])
    x = (np.arange(Z_BINS) + 0.5) / Z_BINS * 100
    for m in models:
        prof = np.array([hu[k]["acf"]["ap"]["z"] for k in hu if k.startswith(m + "|") and "acf" in hu[k]])
        ax.plot(prof.mean(0), x, color=FINAL_C if m == args.final else BASE_C, lw=3.5, label=m)
    ax.set_xlabel("Mean |ACF error| (%)")
    ax.set_ylabel("Body height (%; 100 = top of head)")
    ax.set_title("Along the body (AP rays)", loc="left")
    ax.legend(loc="lower right")
    # Per patient.
    ax = fig.add_subplot(bot[0, 0])
    rng = np.random.default_rng(0)
    vals = {}
    for i, m in enumerate(models):
        v = np.array([hu[k]["acf"]["ap"]["mean_abs"] for k in sorted(hu) if k.startswith(m + "|") and "acf" in hu[k]])
        vals[m] = v
        c = FINAL_C if m == args.final else BASE_C
        ax.bar(i, v.mean(), 0.6, color=c)
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=30, color=INK, alpha=0.4, lw=0, zorder=3)
        ax.annotate(f"{v.mean():.1f} %", (i, v.mean()), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean |ACF error| per exam (%)")
    ax.set_title("Per exam (AP rays)", loc="left")
    ax.grid(axis="x", visible=False)
    # Attenuation coefficient per tissue.
    ax = fig.add_subplot(bot[0, 2])
    w = 0.38
    for i, m in enumerate(models):
        means = []
        for k, name in HU_CLASSES:
            r = [100 * (hu[key][k]["mu_pct"] - hu[key][k]["mu_ct"]) / hu[key][k]["mu_ct"]
                 for key in hu if key.startswith(m + "|") and k in hu[key] and "mu_ct" in hu[key][k]]
            means.append(np.mean(r))
        xs = np.arange(len(HU_CLASSES)) + (i - 0.5) * w
        ax.bar(xs, means, w * 0.95, color=FINAL_C if m == args.final else BASE_C, label=m)
        for xx, mm in zip(xs, means):
            ax.annotate(f"{mm:+.0f} %", (xx, mm), xytext=(0, 6 if mm >= 0 else -18), textcoords="offset points",
                        ha="center", fontsize=13)
        if m == args.final:
            pts.append("μ error per tissue (final): " + ", ".join(f"{n.split(' (')[0]} {v:+.0f} %"
                                                                for (_, n), v in zip(HU_CLASSES, means)))
    ax.axhline(0, color=INK2, lw=1.2)
    ax.set_xticks(range(len(HU_CLASSES)))
    ax.set_xticklabels([n.split(" (")[0] for _, n in HU_CLASSES])
    ax.set_ylabel("Attenuation μ, pseudo-CT vs CT (%)")
    ax.set_title("Attenuation coefficient per tissue (mean over exams)", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    f = vals[args.final]
    p95 = np.mean([hu[k]["acf"]["ap"]["p95_abs"] for k in hu if k.startswith(args.final + "|") and "acf" in hu[k]])
    fig.text(0.01, 0.99, f"What it means for PET: attenuation-correction error of ~{f.mean():.0f} % on average "
             f"({args.final}), larger at bone and lungs", ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.01, 0.955, "CT and pseudo-CT converted to 511 keV attenuation (bilinear HU→μ, Carney 2006, at each exam's CT kVp); "
             "body only (table and blankets removed). ACF = exp(∫μ) along parallel anterior–posterior rays.\n"
             "To first order, the relative error of the ACF is the relative error of the reconstructed PET activity "
             "along that ray. This is a proxy: the real effect needs a PET reconstruction with the pseudo-CT.",
             ha="left", va="top", fontsize=14, color=INK2, linespacing=1.4)
    fig.subplots_adjust(top=0.87, bottom=0.06, left=0.05, right=0.97)
    pts = [f"Mean |ACF error| per exam: " + ", ".join(f"{m} {v.mean():.1f} %" for m, v in vals.items()) +
           f"; 95th percentile of rays ({args.final}) {p95:.0f} %."] + pts
    save(fig, args, "24_utrecht_attenuation_correction.png",
         "Attenuation-correction factor error implied by the pseudo-CT (proxy for the PET quantification error).", pts)


def fig_axial(tab, ages, args):
    pred_root = Path(args.pred_dir, "utrecht_pred")
    for name, s in example_picks(tab, ages, args):
        case = load_h5(args.utrecht_h5_dir, s)
        pet, ct, mask = case["pet"], case["ct"], case["mask"]
        body = mask > 0
        levels = _levels(mask)
        preds = {m: load_pred(pred_root, m, "pet_only", s) for m in ("v18", args.final)}
        fig, axes = plt.subplots(len(levels), 5, figsize=(20, 3.3 * len(levels) + 2.0), squeeze=False)
        ct_kw = dict(cmap="gray", vmin=-1000, vmax=1500)
        for row, (lev, z) in zip(axes, levels.items()):
            crop = _crop2d(axial(body, z))
            sl = lambda v: axial(v, z)[crop]
            row[0].imshow(sl(pet), cmap="gray_r", vmax=np.percentile(pet[body], 99.5))
            row[1].imshow(sl(ct), **ct_kw)
            for j, m in enumerate(("v18", args.final)):
                pr = preds[m]
                b = body[:, :, z]
                mae = np.abs(pr[:, :, z] - ct[:, :, z])[b].mean()
                row[2 + j].imshow(sl(pr), **ct_kw)
                row[2 + j].set_title(f"{m}: {mae:.0f} HU in this slice", fontsize=14)
            im = row[4].imshow(sl(np.where(body, np.abs(preds[args.final] - ct), 0)), cmap=ERR_CMAP, vmin=0, vmax=500)
            row[0].set_title("NAC-PET", fontsize=14)
            row[1].set_title("Real CT", fontsize=14)
            row[4].set_title(f"|Error| {args.final}", fontsize=14)
            for ax in row:
                clean_axes(ax)
            row[0].set_ylabel(lev, fontsize=17, fontweight="bold", color=INK)
            row[0].yaxis.label.set_visible(True)
        cb = fig.colorbar(im, ax=axes[:, 4].tolist(), fraction=0.03, pad=0.02)
        cb.set_label("Absolute error (HU)")
        cb.outline.set_visible(False)
        age = ages.get(s, "?")
        fig.text(0.01, 0.995, f"Axial view, {name.lower()}: {age}-year-old ({short_id(s).split(' ')[0]}), "
                 f"body MAE {tab[args.final][s]['mae_body']:.0f} HU", ha="left", va="top", fontsize=22,
                 fontweight="bold")
        fig.text(0.01, 0.995 - 0.7 / fig.get_size_inches()[1],
                 "Levels found automatically from the CT: middle of the lungs, half a lung length below them, "
                 "and the slice with most pelvic bone. Same window for all CT images (−1000 to 1500 HU).",
                 ha="left", va="top", fontsize=14, color=INK2)
        fig.subplots_adjust(top=1 - 1.5 / fig.get_size_inches()[1], bottom=0.02, left=0.04, right=0.9,
                            wspace=0.05, hspace=0.18)
        key = {"Youngest": "a_youngest", "Median error": "b_median", "Largest error": "c_largest"}[name]
        save(fig, args, f"25{key}_axial.png", f"Axial slices at three levels for the {name.lower()} exam "
             f"({age} y): NAC-PET, CT, v18, {args.final} and the error.",
             ["Lungs: the fine-tuned model places air in the lungs, but their edges and the small airways are blurred.",
              "Bone: vertebrae and pelvis appear as faint grey (too low HU); ribs are mostly missing."])


def fig_uncertainty(tab, args):
    root = Path(args.pred_dir, "utrecht_pred_folds")
    labels = [f"{args.final}_fold{k}" for k in range(5)]
    test = sorted(s for s, r in tab.get(args.final, {}).items() if r["h5"] == "test.h5")
    test = [s for s in test if all((root / l / "pet_only" / s / "ct.nii.gz").is_file() for l in labels)]
    if not test:
        return
    edges = np.arange(0, 401, 25)
    num, den = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    rho, per = [], {}
    ex = sorted(test, key=lambda s: tab[args.final][s]["mae_body"])[len(test) // 2]
    ex_imgs = None
    rng = np.random.default_rng(0)
    for s in test:
        case = load_h5(args.utrecht_h5_dir, s, keys=("ct", "mask"))
        ct, mask = case["ct"].astype(np.float32), case["mask"]
        body = mask > 0
        stack = np.stack([load_pred(root, l, "pet_only", s) for l in labels])
        sd = stack.std(0)
        ens = stack.mean(0)
        err = np.abs(ens - ct)
        idx = np.clip(np.digitize(sd[body], edges) - 1, 0, len(edges) - 2)
        num += np.bincount(idx, weights=err[body], minlength=len(edges) - 1)
        den += np.bincount(idx, minlength=len(edges) - 1)
        samp = rng.choice(int(body.sum()), 20000, replace=False)
        rho.append(stats.spearmanr(sd[body][samp], err[body][samp])[0])
        per[s] = (float(sd[body].mean()), float(err[body].mean()))
        if s == ex:
            y = int(np.argmax(body.sum(axis=(0, 2))))
            ex_imgs = (coronal(ct, y), coronal(ens, y), coronal(np.where(body, err, 0), y),
                       coronal(np.where(body, sd, np.nan), y), coronal(body, y))
    fig = plt.figure(figsize=(21, 10))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.6], wspace=0.15)
    crop = _crop2d(ex_imgs[4])
    titles = ["Real CT", f"Pseudo-CT ({args.final}, mean of 5)", "|Error|", "Disagreement (SD of 5 models)"]
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        if j < 2:
            ax.imshow(ex_imgs[j][crop], cmap="gray", vmin=-1000, vmax=1500)
        else:
            im = ax.imshow(ex_imgs[j][crop], cmap=ERR_CMAP if j == 2 else "magma_r", vmin=0,
                           vmax=500 if j == 2 else 250)
            cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02, orientation="horizontal")
            cb.set_label("HU")
            cb.outline.set_visible(False)
        ax.set_title(titles[j], fontsize=14)
        clean_axes(ax)
    ax = fig.add_subplot(gs[0, 4])
    ok = den > 1000
    xc = (edges[:-1] + edges[1:]) / 2
    ax.plot(xc[ok], (num / np.maximum(den, 1))[ok], "-o", color=FINAL_C, lw=3, ms=9)
    ax.set_xlabel("Disagreement between the 5 models (SD, HU)")
    ax.set_ylabel("Mean |error| of the ensemble (HU)")
    ax.set_title("All body voxels of the 5 test exams", loc="left")
    ax2 = ax.twinx()
    ax2.bar(xc[ok], den[ok] / den.sum() * 100, width=20, color=GRID, alpha=0.6, zorder=0)
    ax2.set_ylabel("Share of voxels (%)", color=INK2)
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    r = float(np.mean(rho))
    fig.text(0.01, 0.985, "Where the five fold models disagree, the error is larger: a built-in warning map",
             ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.01, 0.93, f"Test set (5 exams never used in training). The 5 fold models each predict the exam; the "
             f"final pseudo-CT is their mean and their standard deviation (SD) marks uncertain regions.\n"
             f"Example: {short_id(ex)}, {ages_global.get(ex, '?')} y. Voxel-wise Spearman ρ between SD and error: "
             f"{r:.2f} (mean of the 5 exams).", ha="left", va="top", fontsize=14, color=INK2, linespacing=1.4)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.02, right=0.95)
    save(fig, args, "26_utrecht_uncertainty.png",
         "Disagreement of the 5 fold models on the test exams and its relation to the error.",
         [f"Voxel-wise Spearman ρ(SD, |error|) = {r:.2f}.",
          "Disagreement is concentrated at lung edges, bone and body outline: the same places as the errors.",
          "A disagreement map could be shown next to the pseudo-CT so users know where to trust it less."])


def fig_bland_altman(hu, ages, args):
    models = [m for m in ("v18", args.final) if any(k.startswith(m + "|") and "body" in hu[k] for k in hu)]
    if args.final not in models:
        return
    panels = [("body", "Whole body"), ("lung", "Lungs / internal air"), ("bone", "Bone")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.8))
    pts = []
    for ax, (k, name) in zip(axes, panels):
        for m in models:
            keys = [key for key in hu if key.startswith(m + "|") and k in hu[key]]
            a = np.array([hu[key][k]["ct"] for key in keys])
            b = np.array([hu[key][k]["pct"] for key in keys])
            young = np.array([ages.get(key.split("|")[1], 99) < YOUNG for key in keys])
            mean, diff = (a + b) / 2, b - a
            bias, sd = diff.mean(), diff.std(ddof=1)
            c = FINAL_C if m == args.final else BASE_C
            al = 0.95 if m == args.final else 0.5
            ax.scatter(mean[~young], diff[~young], s=90, color=c, alpha=al, edgecolor=SURFACE, lw=1, zorder=3,
                       label=f"{m}, ≥ {YOUNG} y")
            ax.scatter(mean[young], diff[young], s=110, marker="D", color=c, alpha=al, edgecolor=INK, lw=1,
                       zorder=4, label=f"{m}, < {YOUNG} y")
            ax.axhline(bias, color=c, lw=2.5, alpha=al)
            for lim in (bias - 1.96 * sd, bias + 1.96 * sd):
                ax.axhline(lim, color=c, lw=1.5, ls="--", alpha=al)
            ax.text(0.02, 0.97 - 0.08 * models.index(m), f"{m}: {bias:+.0f} [{bias - 1.96 * sd:+.0f}, "
                    f"{bias + 1.96 * sd:+.0f}]", transform=ax.transAxes, va="top", fontsize=14, color=c,
                    fontweight="bold", bbox=dict(fc=SURFACE, ec="none", alpha=0.9), zorder=6)
            if m == args.final:
                pts.append(f"{name}: bias {bias:+.0f} HU, 95 % limits of agreement {bias - 1.96 * sd:+.0f} to "
                           f"{bias + 1.96 * sd:+.0f} HU")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(name, loc="left")
        ax.set_xlabel("Mean of CT and pseudo-CT (HU)")
    axes[0].set_ylabel("Pseudo-CT − CT (HU)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    header(fig, "Agreement per exam (Bland–Altman): small bias for the whole body, systematic errors in lungs and bone",
           "One marker per Utrecht exam: mean HU of the tissue in the real CT and in the pseudo-CT. Solid line = mean "
           "bias, dashed = 95 % limits of agreement (bias ± 1.96 SD).\nNumbers: bias [lower, upper limit]. "
           "A narrow band means the error is predictable and could be corrected.")
    fig.tight_layout(rect=(0, 0.07, 1, 1 - 1.69 / fig.get_size_inches()[1]))
    save(fig, args, "27_utrecht_bland_altman.png", "Bland–Altman agreement of mean HU per exam.", pts)


def fig_youngest(hu, tab, ages, args):
    pred_root = Path(args.pred_dir, "utrecht_pred")
    subs = sorted((s for s in tab.get(args.final, {}) if ages.get(s, 99) < 5), key=lambda s: ages[s])
    if not subs:
        return
    fig = plt.figure(figsize=(22, 6.2 * len(subs) + 2))
    gs = fig.add_gridspec(len(subs), 6, width_ratios=[1, 1, 1.2, 1.2, 1.2, 2.2], wspace=0.08, hspace=0.15)
    for i, s in enumerate(subs):
        case = load_h5(args.utrecht_h5_dir, s, keys=("ct", "mask"))
        ct, mask = case["ct"], case["mask"]
        body = mask > 0
        pr = load_pred(pred_root, args.final, "pet_only", s)
        y = int(np.argmax(body.sum(axis=(0, 2))))
        z = _levels(mask).get("Thorax", ct.shape[2] // 2)
        cc = _crop2d(coronal(body, y))
        ca = _crop2d(axial(body, z))
        imgs = [(coronal(ct, y)[cc], "Real CT, coronal"), (coronal(pr, y)[cc], f"{args.final}, coronal"),
                (axial(ct, z)[ca], "Real CT, mid-lung"), (axial(pr, z)[ca], f"{args.final}, mid-lung"),
                (axial(np.where(body, np.abs(pr - ct), 0), z)[ca], "|Error|, mid-lung")]
        for j, (im, t) in enumerate(imgs):
            ax = fig.add_subplot(gs[i, j])
            if j == 4:
                ax.imshow(im, cmap=ERR_CMAP, vmin=0, vmax=500)
            else:
                ax.imshow(im, cmap="gray", vmin=-1000, vmax=1500)
            ax.set_title(t, fontsize=14)
            clean_axes(ax)
            if j == 0:
                ax.set_ylabel(f"{ages[s]} y, {short_id(s).split(' ')[0]}\nbody MAE {tab[args.final][s]['mae_body']:.0f} HU",
                              fontsize=16, fontweight="bold", color=INK)
                ax.yaxis.label.set_visible(True)
    # Lung volume, all exams.
    ax = fig.add_subplot(gs[:, 5])
    pts = []
    for m in ("v18", args.final):
        keys = [k for k in hu if k.startswith(m + "|") and "lungvol" in hu[k]]
        if not keys:
            continue
        a = np.array([hu[k]["lungvol"]["ct"] for k in keys])
        b = np.array([hu[k]["lungvol"]["pct"] for k in keys])
        yng = np.array([ages.get(k.split("|")[1], 99) < 5 for k in keys])
        c = FINAL_C if m == args.final else BASE_C
        ax.scatter(a[~yng], b[~yng], s=80, color=c, alpha=0.9 if m == args.final else 0.5, edgecolor=SURFACE,
                   label=m, zorder=3)
        ax.scatter(a[yng], b[yng], s=220, marker="*", color=c, edgecolor=INK, lw=1, zorder=4,
                   label=f"{m}, 2 and 4 y")
        pts.append(f"{m}: predicted / real lung volume = {np.median(b / a):.2f} (median); "
                   f"2 and 4 y: {', '.join(f'{x:.2f}' for x in b[yng] / a[yng])}")
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color=INK2, ls="--", lw=1.5)
    ax.set_xlabel("Real lung volume (L, lung slab)")
    ax.set_ylabel(f"Predicted lung volume (L, pseudo-CT < {LUNG_HU} HU)")
    ax.set_title("Lung volume, all 25 exams", loc="left")
    ax.legend(loc="upper left", fontsize=12)
    fig.text(0.01, 0.995, "The two youngest children (2 and 4 y): lungs blurred and partly filled, small bones missing",
             ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.01, 0.995 - 0.65 / fig.get_size_inches()[1],
             "These children are smaller than any training subject; both sit in the same fold, so their model saw no "
             "child under 9 years.\nRight: lung volume measured in the same slab of the real CT and the pseudo-CT "
             "(dashed = perfect). Stars = 2- and 4-year-olds.", ha="left", va="top", fontsize=14, color=INK2,
             linespacing=1.4)
    fig.subplots_adjust(top=1 - 1.6 / fig.get_size_inches()[1], bottom=0.06, left=0.04, right=0.98)
    save(fig, args, "28_utrecht_youngest_children.png",
         "Close-up of the 2- and 4-year-olds and predicted vs real lung volume for all exams.", pts)



# ----------------------------------------------------------------------------- 29 baseline
def fig_baseline(hu, tab, ages, args):
    models = [m for m in ("water", "v18", args.final) if m in tab]
    if "water" not in models or not any(k.startswith("water|") and "acf" in hu[k] for k in hu):
        return
    col = {"water": "#8a8984", "v18": BASE_C, args.final: FINAL_C}
    name = {"water": "Water\n(no learning)", "v18": "v18\nzero-shot", args.final: f"{args.final}\nfine-tuned"}
    subs = sorted(set.intersection(*[set(tab[m]) for m in models]))
    fig = plt.figure(figsize=(22, 13))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42)
    top = outer[0].subgridspec(1, 5, wspace=0.35)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1, 1.6], wspace=0.25)
    metrics = [("mae_body", "Whole body", "MAE (HU)"), ("mae_soft", "Soft tissue", "MAE (HU)"),
               ("mae_air", "Lungs / internal air", "MAE (HU)"), ("mae_bone", "Bone", "MAE (HU)"),
               ("mae_thorax", "Thorax", "MAE (HU)")]
    pts = []
    for j, (k, title, unit) in enumerate(metrics):
        ax = fig.add_subplot(top[0, j])
        vals = [np.mean([tab[m][s][k] for s in subs]) for m in models]
        ax.bar(range(len(models)), vals, 0.7, color=[col[m] for m in models])
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=14,
                        fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m if m != "water" else "water" for m in models], fontsize=13)
        ax.set_title(title, loc="left", fontsize=16)
        ax.set_ylabel(unit if j == 0 else "")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="x", visible=False)
        a = np.array([tab["water"][s][k] for s in subs])
        b = np.array([tab[args.final][s][k] for s in subs])
        pts.append(f"{title}: water {a.mean():.0f}, v18 {vals[1]:.0f}, {args.final} {b.mean():.0f} HU "
                   f"({args.final} better in {int((b < a).sum())}/{len(subs)}, {p_text(wilcoxon(a, b))})")
    ax = fig.add_subplot(bot[0, 0])
    rng = np.random.default_rng(0)
    acf = {}
    for i, m in enumerate(models):
        v = np.array([hu[f"{m}|{s}"]["acf"]["ap"]["mean_abs"] for s in subs if f"{m}|{s}" in hu])
        acf[m] = v
        ax.bar(i, v.mean(), 0.65, color=col[m])
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=26, color=INK, alpha=0.4, lw=0, zorder=3)
        ax.annotate(f"{v.mean():.1f} %", (i, v.mean()), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([name[m] for m in models], fontsize=13)
    ax.set_ylabel("Mean |ACF error| per exam (%)")
    ax.set_title("Attenuation-correction proxy", loc="left", fontsize=16)
    ax.grid(axis="x", visible=False)
    a, b = acf["water"], acf[args.final]
    pts.append(f"ACF error: water {a.mean():.1f} %, v18 {acf['v18'].mean():.1f} %, {args.final} {b.mean():.1f} % "
               f"({args.final} better in {int((b < a).sum())}/{len(b)}, {p_text(wilcoxon(a, b))})")
    ax = fig.add_subplot(bot[0, 1])
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        means = []
        for k, _ in HU_CLASSES:
            r = [100 * (hu[key][k]["mu_pct"] - hu[key][k]["mu_ct"]) / hu[key][k]["mu_ct"]
                 for key in hu if key.startswith(m + "|") and k in hu[key] and "mu_ct" in hu[key][k]]
            means.append(np.mean(r))
        xs = np.arange(len(HU_CLASSES)) + (i - (len(models) - 1) / 2) * w
        ax.bar(xs, means, w * 0.95, color=col[m], label=name[m].replace("\n", " "))
        for xx, mm in zip(xs, means):
            ax.annotate(f"{mm:+.0f}", (xx, mm), xytext=(0, 5 if mm >= 0 else -17), textcoords="offset points",
                        ha="center", fontsize=12)
    ax.axhline(0, color=INK2, lw=1.2)
    ax.set_xticks(range(len(HU_CLASSES)))
    ax.set_xticklabels([n.split(" (")[0] for _, n in HU_CLASSES])
    ax.set_ylabel("Attenuation μ vs CT (%)")
    ax.set_title("Attenuation per tissue (mean over exams)", loc="left", fontsize=16)
    ax.legend(loc="upper right", fontsize=13)
    ax.grid(axis="x", visible=False)
    header(fig, f"Only the fine-tuned model beats a water baseline on attenuation "
                f"({acf[args.final].mean():.1f} vs {acf['water'].mean():.1f} %), mainly through the lungs",
           "Utrecht, n = 25, PET only. Water baseline = 0 HU inside the NAC-PET body contour (the same contour the "
           "network uses), air outside: no lungs, no bone.\nWhole-body MAE favours the baseline because soft tissue "
           "dominates the voxel count; the attenuation proxy weights tissues by how much they attenuate.")
    fig.subplots_adjust(top=1 - 1.69 / fig.get_size_inches()[1], bottom=0.06, left=0.05, right=0.98)
    save(fig, args, "29_utrecht_vs_water_baseline.png",
         "Comparison of the final model with a uniform-water baseline from the NAC-PET body contour.", pts)


# ----------------------------------------------------------------------------- 30 field of view
FOV_BINS = np.arange(-7.0, 3.01, 0.25)   # position in lung lengths from the lung base (+ = towards the head)


def fov_stats(args, models, subs):
    """Per exam: slice-wise |error| binned by position in lung lengths from the base of the lungs."""
    cache = Path(args.cache_dir) / "cache_fov.json"
    old = json.loads(cache.read_text()) if cache.is_file() else {}
    pred_root = Path(args.pred_dir, "utrecht_pred")
    changed = False
    for s in subs:
        need = [m for m in models if f"{m}|{s}" not in old and (pred_root / m / "pet_only" / s / "ct.nii.gz").is_file()]
        if not need:
            continue
        case = load_h5(args.utrecht_h5_dir, s, keys=("ct", "mask"))
        ct, mask = case["ct"].astype(np.float32), case["mask"]
        body = mask > 0
        lung = lung_z_range(mask, contiguous=True)
        if not lung:
            continue
        lo, hi = lung
        pos = (np.arange(ct.shape[2]) - lo) / (hi - lo)
        idx = np.digitize(pos, FOV_BINS) - 1
        ok = (idx >= 0) & (idx < len(FOV_BINS) - 1)
        n_z = body.sum(axis=(0, 1)).astype(float)
        for m in need:
            pr = load_pred(pred_root, m, "pet_only", s)
            e_z = np.where(body, np.abs(pr - ct), 0).sum(axis=(0, 1))
            old[f"{m}|{s}"] = {"err": np.bincount(idx[ok], weights=e_z[ok], minlength=len(FOV_BINS) - 1).tolist(),
                               "n": np.bincount(idx[ok], weights=n_z[ok], minlength=len(FOV_BINS) - 1).tolist(),
                               "pos_err": float((e_z * (pos < args.fov_below)).sum()), "pos_n": float((n_z * (pos < args.fov_below)).sum()),
                               "in_err": float((e_z * (pos >= args.fov_below)).sum()), "in_n": float((n_z * (pos >= args.fov_below)).sum())}
            changed = True
            print(f"    FOV {m} {s}", flush=True)
    if changed:
        cache.write_text(json.dumps(old))
    return old


def fig_fov(tab, ages, args):
    fov_file = Path(EVAL, "bigmac_fov_vs_lungs.json")
    if not fov_file.is_file() or args.final not in tab:
        return
    args.fov_below = -json.loads(fov_file.read_text())["below"]
    models = [m for m in ("water", "v18", args.final) if m in tab]
    subs = sorted(tab[args.final])
    st = fov_stats(args, models, subs)
    col = {"water": "#8a8984", "v18": BASE_C, args.final: FINAL_C}
    lab = {"water": "Water baseline", "v18": "v18 zero-shot", args.final: f"{args.final} fine-tuned"}
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    xc = (FOV_BINS[:-1] + FOV_BINS[1:]) / 2
    for m in models:
        keys = [f"{m}|{s}" for s in subs if f"{m}|{s}" in st]
        err = np.array([st[k]["err"] for k in keys])
        n = np.array([st[k]["n"] for k in keys])
        prof = np.where(n > 0, err / np.maximum(n, 1), np.nan)
        cover = (n > 0).sum(0) >= 10
        ax.plot(xc[cover], np.nanmean(prof, 0)[cover], color=col[m], lw=3.5, label=lab[m])
    ax.axvspan(FOV_BINS[0], args.fov_below, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axvspan(0, 1, color="#cde2fb", alpha=0.5, lw=0, zorder=0)
    ymax = ax.get_ylim()[1]
    ax.text((FOV_BINS[0] + args.fov_below) / 2, ymax * 0.97, "never seen in\nBIC-MAC (legs)", ha="center", va="top",
            fontsize=14, color=INK2, fontweight="bold")
    ax.text(0.5, ymax * 0.97, "lungs", ha="center", va="top", fontsize=14, color=INK2, fontweight="bold")
    ax.set_xlabel("Position along the body, in lung lengths from the base of the lungs (+ = towards the head)")
    ax.set_ylabel("Mean |error| in the slice (HU)")
    ax.set_title("Error along the body, anatomy aligned on the lungs", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.36, 0.99))
    ax = axes[1]
    pts = []
    for i, m in enumerate(models):
        keys = [s for s in subs if f"{m}|{s}" in st and st[f"{m}|{s}"]["pos_n"] > 1000]
        a = np.array([st[f"{m}|{s}"]["in_err"] / st[f"{m}|{s}"]["in_n"] for s in keys])
        b = np.array([st[f"{m}|{s}"]["pos_err"] / st[f"{m}|{s}"]["pos_n"] for s in keys])
        for ai, bi in zip(a, b):
            ax.plot([i - 0.17, i + 0.17], [ai, bi], color=col[m], alpha=0.3, lw=1)
        ax.scatter(np.full(len(a), i - 0.17), a, s=30, color=col[m], zorder=3)
        ax.scatter(np.full(len(b), i + 0.17), b, s=30, color=col[m], marker="s", zorder=3)
        ax.plot([i - 0.3, i - 0.04], [a.mean()] * 2, color=INK, lw=3)
        ax.plot([i + 0.04, i + 0.3], [b.mean()] * 2, color=INK, lw=3)
        pts.append(f"{lab[m]}: {a.mean():.0f} HU inside the BIC-MAC field of view, {b.mean():.0f} HU in the legs below it "
                   f"(n = {len(keys)} exams, {p_text(wilcoxon(a, b))})")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([lab[m].replace(" ", "\n", 1) for m in models])
    ax.set_ylabel("Body MAE (HU)")
    ax.set_title("Per exam: inside (circles) vs below (squares)", loc="left")
    ax.grid(axis="x", visible=False)
    header(fig, "The legs, never seen in the adult data, are not worse; the error peaks in the lungs and at the head",
           f"Utrecht, PET only. BIC-MAC scans cover from the head to {-args.fov_below:.1f} lung lengths below the lungs "
           "(median of 75 adults); Utrecht exams are whole-body.\nEach slice is placed by its distance to the lung base "
           "in units of that child's lung length, so children and teenagers align. Black bars = means.")
    save(fig, args, "30_utrecht_field_of_view.png",
         "Error inside and outside the anatomical range covered by the adult BIC-MAC scans.", pts)


# ----------------------------------------------------------------------------- 31 per organ
# TotalSegmentator (task "total", --ml) label groups.
ORGANS = [("Brain", [90]), ("Skull", [91]), ("Lungs", [10, 11, 12, 13, 14]), ("Heart", [51]),
          ("Liver", [5]), ("Spleen", [1]), ("Kidneys", [2, 3]), ("Stomach + bowel", [6, 18, 19, 20]),
          ("Bladder", [21]), ("Muscle (back, gluteal, psoas)", list(range(80, 90))),
          ("Vertebrae", list(range(26, 51))), ("Ribs + sternum", list(range(92, 117))),
          ("Pelvis + sacrum", [25, 77, 78]), ("Femur", [75, 76]), ("Humerus + scapula", [69, 70, 71, 72])]


def organ_stats(args, models, subs):
    cache = Path(args.cache_dir) / "cache_organs.json"
    old = json.loads(cache.read_text()) if cache.is_file() else {}
    pred_root = Path(args.pred_dir, "utrecht_pred")
    seg_root = Path(EVAL, "totalseg")
    changed = False
    for s in subs:
        seg_file = seg_root / f"{s}.nii.gz"
        need = [m for m in models if f"{m}|{s}" not in old and (pred_root / m / "pet_only" / s / "ct.nii.gz").is_file()]
        if not need or not seg_file.is_file():
            continue
        seg = np.asarray(nib.as_closest_canonical(nib.load(str(seg_file))).dataobj).astype(np.int16)
        case = load_h5(args.utrecht_h5_dir, s, keys=("ct", "mask"))
        ct = case["ct"].astype(np.float32)
        if seg.shape != ct.shape:
            print(f"    skip {s}: segmentation {seg.shape} vs CT {ct.shape}")
            continue
        body = case["mask"] > 0
        sel = {name: np.isin(seg, labels) & body for name, labels in ORGANS}
        mu_ct = hu_to_mu(ct, KVP.get(s, 120))
        for m in need:
            pr = load_pred(pred_root, m, "pet_only", s)
            mu_pr = hu_to_mu(pr, KVP.get(s, 120))
            d = {}
            for name, msk in sel.items():
                if msk.sum() < 200:
                    continue
                d[name] = {"ct": float(ct[msk].mean()), "pct": float(pr[msk].mean()),
                           "mae": float(np.abs(pr[msk] - ct[msk]).mean()),
                           "mu": float(100 * (mu_pr[msk].mean() - mu_ct[msk].mean()) / mu_ct[msk].mean()),
                           "n": int(msk.sum())}
            old[f"{m}|{s}"] = d
            changed = True
            print(f"    organs {m} {s}", flush=True)
    if changed:
        cache.write_text(json.dumps(old))
    return old


def fig_organs(tab, ages, args):
    models = [m for m in ("water", "v18", args.final) if m in tab]
    subs = sorted(tab.get(args.final, {}))
    st = organ_stats(args, models, subs)
    if not any(k.startswith(args.final + "|") for k in st):
        return
    col = {"water": "#8a8984", "v18": BASE_C, args.final: FINAL_C}
    lab = {"water": "Water baseline", "v18": "v18 zero-shot", args.final: f"{args.final} fine-tuned"}
    names = [n for n, _ in ORGANS if sum(n in st.get(f"{args.final}|{s}", {}) for s in subs) >= 10]
    fig, axes = plt.subplots(1, 2, figsize=(21, 0.62 * len(names) + 3.6), sharey=True)
    y = np.arange(len(names))
    pts = []
    for ax, (key, xlabel, title) in zip(axes, (("mae", "MAE (HU)", "Error"),
                                             ("mu", "Attenuation μ, pseudo-CT vs CT (%)", "Attenuation bias"))):
        for i, m in enumerate(models):
            means, lo, hi = [], [], []
            for n in names:
                v = np.array([st[f"{m}|{s}"][n][key] for s in subs if n in st.get(f"{m}|{s}", {})])
                means.append(v.mean())
                lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            yy = y + (i - 1) * 0.25
            ax.errorbar(means, yy, xerr=[np.array(means) - lo, np.array(hi) - means], fmt="o", ms=9, color=col[m],
                        ecolor=col[m], elinewidth=2, capsize=0, label=lab[m])
            if m == args.final and key == "mu":
                pts += [f"{n}: μ bias {v:+.0f} % (water {np.mean([st[f'water|{s}'][n]['mu'] for s in subs if n in st.get(f'water|{s}', {})]):+.0f} %)"
                        for n, v in zip(names, means)] if "water" in models else []
        if key == "mu":
            ax.axvline(0, color=INK2, lw=1.2)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=14)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    header(fig, "Per organ: soft organs are close to the CT; bone, lungs and skull carry the attenuation error",
           "Utrecht, n = 25, PET only. Organs segmented on the real CT with TotalSegmentator; dot = mean over exams, "
           "bar = middle 50 %.\nAttenuation bias = mean 511 keV μ of the pseudo-CT minus that of the CT, relative to "
           "the CT, inside the organ.")
    save(fig, args, "31_utrecht_per_organ.png", "Per-organ MAE and attenuation bias of water baseline, v18 and the final model.",
         pts)


# ----------------------------------------------------------------------------- 32 feature space
def fig_features(args):
    f = Path(args.cache_dir) / "cache_features.npz"
    if not f.is_file():
        return
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    data = {k.replace("BigMAC", "BIC-MAC"): v for k, v in np.load(f).items()}
    models = sorted({k.split("|")[0] for k in data})
    groups = [("BIC-MAC", ADULT_C, "o"), ("BIC-MAC + aug", "#86b6ef", "^"), ("Utrecht", CHILD_C, "s")]
    fig, axes = plt.subplots(1, len(models), figsize=(9.5 * len(models), 9), squeeze=False)
    pts = []
    for ax, m in zip(axes[0], models):
        X = np.concatenate([data[f"{m}|{g}"] for g, _, _ in groups])
        lab = np.concatenate([[g] * len(data[f"{m}|{g}"]) for g, _, _ in groups])
        Z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(
            StandardScaler().fit_transform(X))
        for g, c, mk in groups:
            sel = lab == g
            ax.scatter(Z[sel, 0], Z[sel, 1], s=22, color=c, marker=mk, alpha=0.7, lw=0, label=g)
        acc = {}
        for src in ("BIC-MAC", "BIC-MAC + aug"):
            Xs = np.concatenate([data[f"{m}|{src}"], data[f"{m}|Utrecht"]])
            ys = np.r_[np.zeros(len(data[f"{m}|{src}"])), np.ones(len(data[f"{m}|Utrecht"]))]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
            acc[src] = cross_val_score(clf, Xs, ys, cv=5, scoring="balanced_accuracy").mean()
        ax.set_title(f"{m}: {MODEL_INFO.get(m, '')}", loc="left", fontsize=16)
        ax.text(0.02, 0.02, f"Cohort separable (BIC-MAC vs Utrecht): {100 * acc['BIC-MAC']:.0f} %\n"
                f"after augmentation (BIC-MAC + aug vs Utrecht): {100 * acc['BIC-MAC + aug']:.0f} %",
                transform=ax.transAxes, fontsize=14, va="bottom", bbox=dict(fc=SURFACE, ec=GRID))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        pts.append(f"{m}: balanced accuracy of a linear classifier telling BIC-MAC from Utrecht = "
                   f"{100 * acc['BIC-MAC']:.0f} %; with augmented BIC-MAC = {100 * acc['BIC-MAC + aug']:.0f} % (50 % = indistinguishable)")
    axes[0][0].legend(loc="upper left", fontsize=14, markerscale=2)
    header(fig, "Inside the network: how far apart adult and paediatric NAC-PET patches are",
           "t-SNE of bottleneck features (after the coordinate FiLM, averaged over the patch) for random body patches; "
           "Utrecht patches are from exams the model did not train on.\nThe box gives the 5-fold balanced accuracy of a "
           "linear classifier separating the cohorts: 50 % = indistinguishable, 100 % = fully separable.")
    save(fig, args, "32_feature_space.png", "Bottleneck feature embedding of BIC-MAC and Utrecht patches for v21 and v23.", pts)


# ----------------------------------------------------------------------------- 00 summary
def fig_summary(hu, tab, ages, args):
    if "v18" not in tab or args.final not in tab:
        return
    subs = sorted(tab[args.final])
    a = np.array([tab["v18"][s]["mae_body"] for s in subs])
    b = np.array([tab[args.final][s]["mae_body"] for s in subs])
    air = [np.mean([tab[m][s]["mae_air"] for s in subs]) for m in ("v18", args.final)]
    yng = [s for s in subs if ages.get(s, 99) < YOUNG]
    acf = [hu[k]["acf"]["ap"]["mean_abs"] for k in hu if k.startswith(args.final + "|") and "acf" in hu[k]]
    bone = [hu[k]["bone"]["pct"] - hu[k]["bone"]["ct"] for k in hu if k.startswith(args.final + "|") and "bone" in hu[k]]
    picks = dict(example_picks(tab, ages, args))
    s_ex = picks.get("Median error")
    fig = plt.figure(figsize=(22, 11.5))
    cols = [("1  The problem", "#e4e3de"), ("2  What we did", "#cde2fb"), ("3  Result", "#cdeee0"),
            ("4  Limits and next step", "#fbe3c4")]
    for i, (t, c) in enumerate(cols):
        x0 = 0.01 + i * 0.2475
        fig.patches.append(plt.Rectangle((x0, 0.03), 0.235, 0.8, transform=fig.transFigure, color=c, zorder=-10))
        fig.text(x0 + 0.012, 0.80, t, fontsize=21, fontweight="bold", va="top")
    txt = [
        "Goal: a CT-like attenuation\nmap for paediatric PET,\npredicted from the NAC-PET\nalone (no CT, no MRI).\n\n"
        "Our model learned on 75 adults\nwith PET + MRI (BIC-MAC).\nUtrecht: 25 exams, 2–20 years,\n"
        "6 under 13, no MRI.",
        "• Same 3D U-Net (1.6 M param.)\n• Train without MRI (dropout)\n• Fine-tune on Utrecht, 5 folds\n"
        "• Oversample lungs + young\n• Make adult PET look like\n   Utrecht PET (domain aug.)\n• Tried: shrunk adults,\n"
        "   synthetic children,\n   ensembles → no extra gain",
        "",
        f"• Bone still predicted too\n   soft ({np.mean(bone):+.0f} HU)\n• Under 13: {np.mean([tab['v18'][s]['mae_body'] for s in yng]):.0f} → "
        f"{np.mean([tab[args.final][s]['mae_body'] for s in yng]):.0f} HU;\n   2- and 4-year-olds not improved\n"
        f"• ACF error ≈ {np.mean(acf):.0f} % (proxy)\n\nNext: more paediatric exams\n(especially < 6 y), PET\n"
        "reconstruction with pCT,\nbone-specific loss.",
    ]
    for i, t in enumerate(txt):
        fig.text(0.01 + i * 0.2475 + 0.012, 0.73, t, fontsize=16, va="top", linespacing=1.45)
    # Result column: big numbers.
    x0 = 0.01 + 2 * 0.2475 + 0.012
    fig.text(x0, 0.73, f"{a.mean():.0f} → {b.mean():.0f} HU", fontsize=34, fontweight="bold", va="top", color=FINAL_C)
    fig.text(x0, 0.655, "body MAE (adult model → final)", fontsize=14, va="top", color=INK2)
    fig.text(x0, 0.61, f"{int((b < a).sum())} / {len(subs)} exams better\nlungs {air[0]:.0f} → {air[1]:.0f} HU "
             f"({100 * (air[1] - air[0]) / air[0]:+.0f} %)", fontsize=17, va="top", linespacing=1.4)
    if s_ex:
        case = load_h5(args.utrecht_h5_dir, s_ex, keys=("ct", "mask"))
        ct, mask = case["ct"], case["mask"]
        body = mask > 0
        y = int(np.argmax(body.sum(axis=(0, 2))))
        pr = load_pred(Path(args.pred_dir, "utrecht_pred"), args.final, "pet_only", s_ex)
        crop = _crop2d(coronal(body, y))
        for j, (im, t) in enumerate(((coronal(ct, y)[crop], "Real CT"), (coronal(pr, y)[crop], "Pseudo-CT"))):
            ax = fig.add_axes([x0 + j * 0.108, 0.06, 0.1, 0.42], zorder=5)
            ax.imshow(im, cmap="gray", vmin=-1000, vmax=1500)
            ax.set_title(t, fontsize=14)
            clean_axes(ax)
    fig.text(0.01, 0.985, "Pseudo-CT from NAC-PET alone for children: summary", fontsize=26, fontweight="bold",
             va="top")
    fig.text(0.01, 0.925, f"Final model {args.final}; all numbers out-of-fold on 25 UMC Utrecht exams "
             "(PET only, no MRI).", fontsize=16, color=INK2, va="top")
    save(fig, args, "00_summary.png", "One-page summary of the problem, approach, result and limits.",
         [f"Body MAE {a.mean():.1f} → {b.mean():.1f} HU, better in {int((b < a).sum())}/{len(subs)} exams."])


# ----------------------------------------------------------------------------- animations
def anim_axial_sweep(tab, ages, args, which="median"):
    """which = "median" (median body MAE of the final model) or "best" (lowest body MAE)."""
    import matplotlib.animation as animation
    if which == "best":
        s = min(tab.get(args.final, {}), key=lambda k: tab[args.final][k]["mae_body"], default=None)
        label, out_name = "best case", "anim_axial_sweep_best"
    else:
        s = dict(example_picks(tab, ages, args)).get("Median error")
        label, out_name = "median error", "anim_axial_sweep"
    if not s:
        return
    case = load_h5(args.utrecht_h5_dir, s, keys=("pet", "ct", "mask"))
    pet, ct, mask = case["pet"], case["ct"], case["mask"]
    body = mask > 0
    pr = load_pred(Path(args.pred_dir, "utrecht_pred"), args.final, "pet_only", s)
    v18 = load_pred(Path(args.pred_dir, "utrecht_pred"), "v18", "pet_only", s)
    zb = np.where(body.any(axis=(0, 1)))[0]
    zs = list(range(zb[-1], zb[0], -4))
    r, c = np.where(body.any(axis=2))      # r = x (columns of the axial view), c = y (rows, flipped by rot90)
    ny = body.shape[1]
    crop = (slice(max(ny - 1 - c.max() - 8, 0), ny - c.min() + 8), slice(max(r.min() - 8, 0), r.max() + 9))
    y = int(np.argmax(body.sum(axis=(0, 2))))
    fig = plt.figure(figsize=(18, 5.2), dpi=80)
    gs = fig.add_gridspec(1, 6, width_ratios=[0.3, 1, 1, 1, 1, 1], wspace=0.04)
    loc = fig.add_subplot(gs[0, 0])
    loc.imshow(coronal(ct, y), cmap="gray", vmin=-1000, vmax=1500, aspect="auto")
    line = loc.axhline(0, color=BAD_C, lw=2)
    clean_axes(loc)
    loc.set_title("Position", fontsize=13)
    axs = [fig.add_subplot(gs[0, j]) for j in range(1, 6)]
    titles = ["NAC-PET (input)", "Real CT", "v18 (adult)", f"{args.final} (final)", f"|Error| {args.final}"]
    ims = []
    for ax, t in zip(axs, titles):
        ims.append(ax.imshow(np.zeros((crop[0].stop - crop[0].start, crop[1].stop - crop[1].start)), cmap="gray"))
        ax.set_title(t, fontsize=14)
        clean_axes(ax)
    ims[0].set_cmap("gray_r")
    ims[0].set_clim(0, np.percentile(pet[body], 99.5))
    for im in ims[1:4]:
        im.set_clim(-1000, 1500)
    ims[4].set_cmap(ERR_CMAP)
    ims[4].set_clim(0, 500)
    test = " (test set)" if tab[args.final][s]["h5"] == "test.h5" else ""
    fig.suptitle(f"Head-to-feet sweep, {ages.get(s, '?')}-year-old, {label}{test}: "
                 f"body MAE v18 {tab['v18'][s]['mae_body']:.0f} HU, {args.final} {tab[args.final][s]['mae_body']:.0f} HU",
                 fontsize=16, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.02)
    err = np.where(body, np.abs(pr - ct), 0)

    def frame(i):
        z = zs[i]
        for im, v in zip(ims, (pet, ct, v18, pr, err)):
            im.set_data(axial(v, z)[crop])
        line.set_ydata([ct.shape[2] - 1 - z] * 2)
        return ims + [line]

    ani = animation.FuncAnimation(fig, frame, frames=len(zs), blit=False)
    out = Path(args.out_dir) / out_name
    ani.save(str(out) + ".gif", writer=animation.PillowWriter(fps=10), dpi=60)
    try:
        ani.save(str(out) + ".mp4", writer=animation.FFMpegWriter(fps=12, bitrate=3000))
    except Exception as e:  # noqa: BLE001
        print(f"  (mp4 skipped: {e})")
    plt.close(fig)
    GUIDE.append((f"{out_name}.gif / .mp4", f"Axial sweep from head to feet of the {label} exam ({short_id(s)}): "
                  f"NAC-PET, CT, v18, {args.final} and error.", ["Use the .mp4 in PowerPoint; the .gif for web/Teams."]))
    print(f"  {out}.gif")


def anim_versions(tab, ages, args):
    import matplotlib.animation as animation
    pred_root = Path(args.pred_dir, "utrecht_pred")
    versions = [m for m in ("v18", "v20", "v21", "v22", "v23", "v24", "v25") if m in tab]
    cands = [s for s in tab.get(args.final, {}) if all((pred_root / m / "pet_only" / s / "ct.nii.gz").is_file()
                                                        for m in versions)]
    if not cands:
        return
    # Exam with the largest v18 -> final gain among those saved for every version.
    s = max(cands, key=lambda s: tab["v18"][s]["mae_body"] - tab[args.final][s]["mae_body"])
    case = load_h5(args.utrecht_h5_dir, s, keys=("ct", "mask"))
    ct, mask = case["ct"], case["mask"]
    body = mask > 0
    y = int(np.argmax(body.sum(axis=(0, 2))))
    crop = _crop2d(coronal(body, y))
    preds = {m: load_pred(pred_root, m, "pet_only", s) for m in versions}
    fig, axs = plt.subplots(1, 3, figsize=(14, 8.5), dpi=80)
    a0 = axs[0].imshow(coronal(ct, y)[crop], cmap="gray", vmin=-1000, vmax=1500)
    axs[0].set_title("Real CT", fontsize=16)
    a1 = axs[1].imshow(coronal(preds[versions[0]], y)[crop], cmap="gray", vmin=-1000, vmax=1500)
    a2 = axs[2].imshow(coronal(np.where(body, np.abs(preds[versions[0]] - ct), 0), y)[crop], cmap=ERR_CMAP, vmin=0, vmax=500)
    for ax in axs:
        clean_axes(ax)
    t1 = axs[1].set_title("", fontsize=16)
    t2 = axs[2].set_title("|Error|", fontsize=16)
    sub = fig.text(0.01, 0.02, "", fontsize=15, color=INK)
    fig.suptitle(f"Model versions on one exam: {ages.get(s, '?')}-year-old ({short_id(s).split(' ')[0]}"
                 f"{', test set' if tab[args.final][s]['h5'] == 'test.h5' else ''})", fontsize=17, fontweight="bold",
                 x=0.01, ha="left")
    frames = [m for m in versions for _ in range(6)]

    def frame(i):
        m = frames[i]
        a1.set_data(coronal(preds[m], y)[crop])
        a2.set_data(coronal(np.where(body, np.abs(preds[m] - ct), 0), y)[crop])
        t1.set_text(f"{m}: body MAE {tab[m][s]['mae_body']:.0f} HU")
        sub.set_text(f"{m}: {MODEL_INFO.get(m, '')}")
        return [a1, a2, t1, sub]

    ani = animation.FuncAnimation(fig, frame, frames=len(frames), blit=False)
    out = Path(args.out_dir) / "anim_model_versions"
    ani.save(str(out) + ".gif", writer=animation.PillowWriter(fps=4))
    try:
        ani.save(str(out) + ".mp4", writer=animation.FFMpegWriter(fps=4, bitrate=3000))
    except Exception as e:  # noqa: BLE001
        print(f"  (mp4 skipped: {e})")
    plt.close(fig)
    GUIDE.append(("anim_model_versions.gif / .mp4", f"The same exam ({short_id(s)}) predicted by every version, "
                  "1.5 s per version.", [", ".join(f"{m} {tab[m][s]['mae_body']:.0f} HU" for m in versions)]))
    print(f"  {out}.gif")


# ----------------------------------------------------------------------------- guide
def write_guide(args):
    lines = [f"# Figure guide (final model: {args.final})\n",
             "Generated by `utils/make_presentation_figures.py`. One entry per figure: caption, then talking points "
             "with the numbers.\n"]
    for name, caption, points in sorted(GUIDE):
        lines.append(f"## {name}\n\n{caption}\n")
        lines += [f"- {p}" for p in points]
        lines.append("")
    lines.append("## Model versions\n")
    lines += [f"- **{m}**: {d}" for m, d in MODEL_INFO.items()]
    (Path(args.out_dir) / "figure_guide.md").write_text("\n".join(lines) + "\n")
    print(f"  {Path(args.out_dir) / 'figure_guide.md'}")


def main():
    p = argparse.ArgumentParser("Presentation figures (English)")
    p.add_argument("--out_dir", default=f"{EVAL}/figures_final")
    p.add_argument("--final", default="v23", help="Model presented as final")
    p.add_argument("--models", nargs="*", default=["v18", "v20", "v21", "v22", "v23", "v24", "v25", "v18+v23"],
                   help="Models in the progression figure (missing ones are skipped)")
    p.add_argument("--oof_csv", default=f"{EVAL}/bigmac_oof.csv")
    p.add_argument("--b4_csv", default=f"{EVAL}/utrecht_B4.csv")
    p.add_argument("--zero_shot_csv", default=f"{EVAL}/utrecht_zero_shot.csv")
    p.add_argument("--final_config", default="config/mutimodal_v23_utrecht_ft.yaml",
                   help="Config of the final model (architecture figure)")
    p.add_argument("--cohorts_csv", default=f"{EVAL}/figuras/cohortes.csv")
    p.add_argument("--pred_dir", default=EVAL)
    p.add_argument("--bigmac_h5_dir", default="/mnt/c/Users/edyA0/Proyectos/data_multimask")
    p.add_argument("--utrecht_h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    p.add_argument("--ped_h5_dir", default="/mnt/d/Data/007_pseudoCT/bigmac_ped_hdf5")
    p.add_argument("--participants", default="/mnt/d/Data/007_pseudoCT/utrecht_bigmac_space/participants.csv")
    p.add_argument("--only", nargs="*", default=None, help="Only these figure numbers (e.g. 08 09)")
    p.add_argument("--paper", action="store_true",
                   help="Paper versions: no title/subtitle, shorter labels, cropped; written to figures_paper")
    p.add_argument("--cache_dir", default=f"{EVAL}/figures_final", help="Where the computed caches live")
    p.add_argument("--b4_ts_csv", default=f"{EVAL}/utrecht_B4_ts.csv")
    args = p.parse_args()
    global PAPER
    PAPER = args.paper
    if PAPER and args.out_dir == f"{EVAL}/figures_final":
        args.out_dir = f"{EVAL}/figures_paper"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run = lambda k: args.only is None or k in args.only

    oof = read_csv(args.oof_csv)
    b4 = read_csv(args.b4_csv)
    coh = read_cohorts(args.cohorts_csv)
    ages = read_ages(args.participants)
    part = read_participants(args.participants)
    tab = b4_table(b4, set(args.models) | {args.final, "v18", "water"})
    # v26 was evaluated only on the anatomical-mask HDF5s; its body MAE uses the same body mask as the others.
    tab.update(b4_table(read_csv(args.b4_ts_csv), {"v26"}))
    print("Figures:")
    zs_rows = read_csv(args.zero_shot_csv)
    steps = [("01", lambda: fig_cohorts(coh, args)), ("02", lambda: fig_bigmac_dataset(coh, oof, args)),
             ("03", lambda: fig_architecture(args)), ("04", lambda: fig_training_scheme(args)),
             ("05", lambda: fig_bigmac_inputs(oof, args)), ("06", lambda: fig_bigmac_tissues(oof, args)),
             ("07", lambda: fig_bigmac_example(oof, args)), ("08", lambda: fig_pet_domain(args)),
             ("09", lambda: fig_synthetic(args)), ("10", lambda: fig_versions(tab, ages, zs_rows, args)),
             ("11", lambda: fig_progress(tab, ages, args)), ("12", lambda: fig_change_effects(tab, ages, args)),
             ("13", lambda: fig_per_patient(tab, ages, part, args)),
             ("14", lambda: fig_model_variation(tab, ages, part, args)), ("15", lambda: fig_tissues(tab, ages, args)),
             ("16", lambda: fig_regions(tab, args))]
    for k, f in steps:
        if run(k):
            f()
    ages_global.update(ages)
    hu = {}
    if any(run(k) for k in ("00", "17", "19", "20", "22", "24", "27", "28", "29")):
        subs = sorted(tab.get(args.final, {}))
        hu, hists, edges = hu_stats(args, ["v18", args.final, "water"], subs)
        if run("17"):
            fig_along_body(hu, ages, args)
    if run("18"):
        fig_error_vs_size(tab, ages, coh, args)
    if run("19"):
        fig_hu_tissue(hu, tab, ages, args)
    if run("20"):
        fig_hu_hist(hists, edges, args)
    if run("21"):
        fig_training_curves(tab, args)
    if run("22"):
        rows = per_patient_rows(tab, hu, ages, part, args)
        write_patient_table(rows, args)
        fig_patient_table(rows, args)
    if run("23"):
        fig_examples(tab, ages, args)
    if run("24"):
        fig_attenuation(hu, tab, ages, args)
    if run("25"):
        fig_axial(tab, ages, args)
    if run("26"):
        fig_uncertainty(tab, args)
    if run("27"):
        fig_bland_altman(hu, ages, args)
    if run("28"):
        fig_youngest(hu, tab, ages, args)
    if run("29"):
        fig_baseline(hu, tab, ages, args)
    if run("30"):
        fig_fov(tab, ages, args)
    if run("31"):
        fig_organs(tab, ages, args)
    if run("32"):
        fig_features(args)
    if run("00"):
        fig_summary(hu, tab, ages, args)
    if run("anim"):
        anim_versions(tab, ages, args)
        anim_axial_sweep(tab, ages, args)
    if run("anim_best") or run("anim"):
        anim_axial_sweep(tab, ages, args, which="best")
    if args.only is None:
        write_guide(args)


if __name__ == "__main__":
    main()
