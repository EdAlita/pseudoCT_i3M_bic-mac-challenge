"""
utils/deform_to_pediatric.py

Tarea T3 (prototipo): deformar un adulto de BigMAC (NAC-PET, MRI, CT, máscara) hacia la
anatomía de un niño de Utrecht, para obtener tripletes PET/MRI/CT con forma pediátrica
(Utrecht no tiene MRI y UMC no puede dar más datos con MRI).

Pasos (todo en la malla común de los HDF5: 1.5234 x 1.5234 x 2.0 mm, ejes x, y, z):
  1. Landmarks desde las máscaras: losa pulmonar (tramo continuo) y extensión del cuerpo
     en el corte central de los pulmones. Dan una escala por eje y un centro.
  2. Recorte del niño a la región equivalente al FOV de torso de BigMAC.
  3. Registro CT adulto -> CT niño con SimpleITK: afín inicializado con los landmarks y
     después B-spline (MeanSquares sobre el CT recortado a [-1000, 1500] HU, en el cuerpo).
  4. La misma transformación se aplica a NAC-PET, MRI (2 canales), CT y máscara.
  5. Control de calidad: Dice de cuerpo / pulmón / hueso y MAE de CT frente al niño, después
     de cada etapa (landmarks, afín, B-spline), y figura de cortes coronales.

Solo cambia la FORMA: la intensidad del PET sigue siendo la del adulto. Un par deformado
hacia un niño del fold k no debe usarse para entrenar el modelo del fold k (copia la forma
de un paciente de su validación); se guarda en qa.json como "target_split".

Ejemplo:
    python -m utils.deform_to_pediatric --adult sub-082 --child sub-PUMP543171-20241216

Owner: Edwing Ulin
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from utils.body_coords import lung_z_range

SPACING = (1.5234375, 1.5234375, 2.0)
OUT_ROOT = "/mnt/d/Data/007_pseudoCT/deform_proto"


# ----------------------------------------------------------------------------- datos
def load_subject(h5_dir, sub_id, keys):
    for f in sorted(Path(h5_dir).glob("*.h5")):
        with h5py.File(f, "r") as hf:
            if sub_id in hf:
                return {k: hf[sub_id][k][:] for k in keys if k in hf[sub_id]}, f.stem
    raise KeyError(f"{sub_id} no está en {h5_dir}")


def to_sitk(arr, origin=(0.0, 0.0, 0.0)):
    """Array (x, y, z) -> imagen SimpleITK con índice (x, y, z) y el espaciado de la malla."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(np.asarray(arr).transpose(2, 1, 0)))
    img.SetSpacing(SPACING)
    img.SetOrigin(origin)
    return img


def to_numpy(img):
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0)


# ----------------------------------------------------------------------------- landmarks
def landmarks(mask):
    """Centro (mm) y tamaño (mm) por eje: x, y del cuerpo en el centro pulmonar; z de la losa pulmonar."""
    z0, z1 = lung_z_range(mask, contiguous=True)
    zm = (z0 + z1) // 2
    body = mask[:, :, zm] > 0
    xs = np.where(body.sum(1) > 0)[0]
    ys = np.where(body.sum(0) > 0)[0]
    lo = np.array([xs[0], ys[0], z0], float)
    hi = np.array([xs[-1] + 1, ys[-1] + 1, z1], float)
    sp = np.array(SPACING)
    return (lo + hi) / 2 * sp, (hi - lo) * sp


def landmark_affine(child_mask, adult_mask):
    """Afín niño -> adulto (convención de SimpleITK: del espacio fijo al móvil)."""
    c_child, s_child = landmarks(child_mask)
    c_adult, s_adult = landmarks(adult_mask)
    scale = s_adult / s_child
    t = sitk.AffineTransform(3)
    t.SetMatrix(np.diag(scale).ravel().tolist())
    t.SetCenter(c_child.tolist())
    t.SetTranslation((c_adult - c_child).tolist())
    return t, scale, c_child, c_adult


def child_crop(child_shape, adult_shape, scale, c_child, c_adult):
    """Cortes z del niño que corresponden al FOV de torso del adulto."""
    z_mm = lambda za: (za - c_adult[2]) / scale[2] + c_child[2]
    z0 = int(np.floor(z_mm(0.0) / SPACING[2]))
    z1 = int(np.ceil(z_mm(adult_shape[2] * SPACING[2]) / SPACING[2]))
    return max(z0, 0), min(z1, child_shape[2])


# ----------------------------------------------------------------------------- registro
def ct_for_metric(ct):
    return sitk.Clamp(sitk.Cast(ct, sitk.sitkFloat32), lowerBound=-1000.0, upperBound=1500.0)


def register(fixed_ct, moving_ct, fixed_mask, init, mesh, iters_affine, iters_bspline):
    fixed = ct_for_metric(fixed_ct)
    moving = ct_for_metric(moving_ct)
    fmask = sitk.BinaryDilate(sitk.Cast(fixed_mask > 0, sitk.sitkUInt8), [5, 5, 3])

    def base(shrink, smooth):
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMeanSquares()
        r.SetMetricFixedMask(fmask)
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(0.05, seed=42)
        r.SetInterpolator(sitk.sitkLinear)
        r.SetShrinkFactorsPerLevel(shrink)
        r.SetSmoothingSigmasPerLevel(smooth)
        r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        return r

    # Afín: parte de los landmarks.
    affine = sitk.AffineTransform(init)
    r = base([8, 4], [2, 1])
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4,
                                               numberOfIterations=iters_affine, gradientMagnitudeTolerance=1e-6)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetInitialTransform(affine, inPlace=True)
    r.Execute(fixed, moving)
    affine_metric = r.GetMetricValue()

    # B-spline sobre el afín.
    bspline = sitk.BSplineTransformInitializer(fixed, mesh)
    r = base([4, 2], [2, 1])
    r.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iters_bspline,
                           maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000)
    r.SetMovingInitialTransform(affine)
    r.SetInitialTransform(bspline, inPlace=True)
    r.Execute(fixed, moving)
    composite = sitk.CompositeTransform(affine)
    composite.AddTransform(bspline)       # se aplica primero el B-spline, después el afín
    return affine, composite, affine_metric, r.GetMetricValue()


def register_demons(fixed_ct, moving_ct, base, shrinks=(4, 2), iters=(80, 40), sigma=2.0):
    """
    Refinamiento no rígido diffeomórfico (sin plegado) sobre una transformación base (afín o
    afín + B-spline): Demons multiresolución entre el CT del niño y el CT del adulto ya
    llevado al espacio del niño con la base.
    """
    norm = lambda im: sitk.RescaleIntensity(ct_for_metric(im), 0.0, 1.0)
    fixed = norm(fixed_ct)
    moving = norm(sitk.Resample(moving_ct, fixed_ct, base, sitk.sitkLinear, -1024.0, sitk.sitkFloat32))
    field = None
    for shrink, n in zip(shrinks, iters):
        f = sitk.Shrink(fixed, [shrink] * 2 + [max(shrink // 2, 1)])
        m = sitk.Shrink(moving, [shrink] * 2 + [max(shrink // 2, 1)])
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(n)
        demons.SetSmoothDisplacementField(True)
        demons.SetStandardDeviations(sigma)
        demons.SetSmoothUpdateField(True)
        demons.SetUpdateFieldStandardDeviations(1.0)
        if field is None:
            field = demons.Execute(f, m)
        else:
            init = sitk.Resample(field, f, sitk.Transform(), sitk.sitkLinear, 0.0, field.GetPixelID())
            field = demons.Execute(f, m, init)
    field = sitk.Resample(field, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, field.GetPixelID())
    composite = sitk.CompositeTransform(base)
    composite.AddTransform(sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64)))
    return composite


def warp(img, ref, transform, interp, default):
    return sitk.Resample(img, ref, transform, interp, default, img.GetPixelID())


# ----------------------------------------------------------------------------- QA
def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def qa(warped_mask, warped_ct, child_mask, child_ct):
    body = child_mask > 0
    return {
        "dice_body": dice(warped_mask > 0, body),
        "dice_lung": dice(warped_mask == 3, child_mask == 3),
        "dice_bone": dice(warped_mask >= 5, child_mask >= 5),
        "mae_ct_body": float(np.abs(warped_ct - child_ct)[body].mean()),
    }


def jacobian_stats(transform, ref, body):
    """Determinante del jacobiano de la deformación en el cuerpo: <= 0 indica plegado del tejido."""
    field = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat64, ref.GetSize(),
                                              ref.GetOrigin(), ref.GetSpacing(), ref.GetDirection())
    jac = to_numpy(sitk.DisplacementFieldJacobianDeterminant(field))[body]
    return {"jac_min": float(jac.min()), "jac_p1": float(np.percentile(jac, 1)),
            "jac_p99": float(np.percentile(jac, 99)), "jac_fold_frac": float((jac <= 0).mean())}


def figure(out_png, child, adult, warped, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.make_figures import SURFACE

    def cor(v, y):
        return np.rot90(v[:, y, :])

    yc = int(np.median(np.where(child["mask"] > 0)[1]))
    ya = int(np.median(np.where(adult["mask"] > 0)[1]))
    ct_kw = dict(cmap="gray", vmin=-1000, vmax=1500)
    panels = [
        ("Adulto BigMAC: CT", cor(adult["ct"], ya), ct_kw, None),
        ("Niño Utrecht: CT (objetivo)", cor(child["ct"], yc), ct_kw, None),
        ("Adulto deformado: CT", cor(warped["ct"], yc), ct_kw, cor(child["mask"], yc)),
        ("Adulto deformado: NAC-PET", cor(warped["pet"], yc),
         dict(cmap="gray_r", vmax=np.percentile(warped["pet"][warped["mask"] > 0], 99.5)), cor(child["mask"], yc)),
        ("Adulto deformado: MRI (in-phase)", cor(warped["mri"][0], yc),
         dict(cmap="gray", vmax=np.percentile(warped["mri"][0][warped["mask"] > 0], 99.5)), cor(child["mask"], yc)),
        ("Niño Utrecht: NAC-PET", cor(child["pet"], yc),
         dict(cmap="gray_r", vmax=np.percentile(child["pet"][child["mask"] > 0], 99.5)), None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 5.2))
    for ax, (name, img, kw, contour) in zip(axes, panels):
        ax.imshow(img, aspect=SPACING[2] / SPACING[0], **kw)
        if contour is not None:
            ax.contour(contour > 0, [0.5], colors="#eb6834", linewidths=0.8)
            ax.contour(contour == 3, [0.5], colors="#1baf7a", linewidths=0.8)
        ax.set_title(name, fontsize=9.5)
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser("Deformar un adulto de BigMAC hacia la anatomía de un niño de Utrecht")
    p.add_argument("--adult", required=True)
    p.add_argument("--child", required=True)
    p.add_argument("--bigmac_h5_dir", default="/mnt/c/Users/edyA0/Proyectos/data_multimask")
    p.add_argument("--utrecht_h5_dir", default="/mnt/d/Data/007_pseudoCT/utrecht_hdf5")
    p.add_argument("--out_root", default=OUT_ROOT)
    p.add_argument("--mesh", type=int, nargs=3, default=[6, 6, 8], help="Nodos del B-spline por eje")
    p.add_argument("--iters_affine", type=int, default=150)
    p.add_argument("--iters_bspline", type=int, default=60)
    p.add_argument("--method", choices=["bspline", "demons", "bspline_demons", "adaptive"], default="adaptive",
                   help="Refinamiento no rígido: B-spline grueso, Demons diffeomórfico, B-spline + Demons, o "
                        "adaptive (B-spline 6x6x8 + Demons; si pliega, 4x4x6 + Demons; si aún pliega, solo Demons)")
    p.add_argument("--max_fold", type=float, default=0.001, help="adaptive: fracción máxima de plegado aceptada")
    p.add_argument("--demons_sigma", type=float, default=1.5, help="Suavizado del campo (vóxeles)")
    args = p.parse_args()
    adult, adult_split = load_subject(args.bigmac_h5_dir, args.adult, ["pet", "mri", "ct", "mask"])
    child, child_split = load_subject(args.utrecht_h5_dir, args.child, ["pet", "ct", "mask"])
    warped, report, child = deform_pair(adult, child, args, verbose=True)
    report.update(adult=args.adult, adult_split=adult_split, child=args.child, target_split=child_split)

    out = Path(args.out_root) / f"{args.adult}_to_{args.child}"
    out.mkdir(parents=True, exist_ok=True)
    aff = np.diag(list(SPACING) + [1.0])
    for k, v in warped.items():
        if k == "mri":
            for c, name in enumerate(["mri_in_phase", "mri_out_phase"][: v.shape[0]]):
                nib.save(nib.Nifti1Image(v[c].astype(np.float32), aff), str(out / f"{name}.nii.gz"))
        else:
            dtype = np.int16 if k == "ct" else np.uint8 if k == "mask" else np.float32
            nib.save(nib.Nifti1Image(v.astype(dtype), aff), str(out / f"{k}.nii.gz"))
    (out / "qa.json").write_text(json.dumps(report, indent=2))
    figure(out / "qa_coronal.png", child, adult, warped, qa_title(args.adult, args.child, report))
    print(f"Guardado en {out} ({report['seconds']} s)")


def qa_title(adult_id, child_id, report):
    st = report["stages"]["no_rigido"]
    return (f"{adult_id} (BigMAC) deformado hacia {child_id} (Utrecht); Dice cuerpo {st['dice_body']:.2f}, "
            f"pulmón {st['dice_lung']:.2f} — naranja: cuerpo del niño, verde: sus pulmones")


def deform_pair(adult, child, args, verbose=False):
    """
    Deforma un adulto (dict pet, mri, ct, mask) hacia un niño (dict pet, ct, mask).
    Devuelve (warped, report, child recortado). `args` lleva mesh, iters_*, method, demons_sigma.
    """
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    init, scale, c_child, c_adult = landmark_affine(child["mask"], adult["mask"])
    z0, z1 = child_crop(child["ct"].shape, adult["ct"].shape, scale, c_child, c_adult)
    log(f"Escala adulto/niño (x, y, z): {np.round(1 / scale, 2)}; recorte del niño z {z0}-{z1}", flush=True)
    child = {k: v[..., z0:z1] for k, v in child.items()}
    origin = (0.0, 0.0, z0 * SPACING[2])            # mismo espacio físico que el volumen completo
    fixed_ct, fixed_mask = to_sitk(child["ct"], origin), to_sitk(child["mask"], origin)
    moving = {k: to_sitk(adult[k]) for k in ("ct", "pet", "mask")}
    moving_mri = [to_sitk(adult["mri"][c]) for c in range(adult["mri"].shape[0])]

    body = child["mask"] > 0
    if args.method == "adaptive":
        # De más calidad a más seguro; se queda con el primero que no pliega el tejido.
        candidates = [("bspline6+demons", args.mesh), ("bspline4+demons", [4, 4, 6]), ("demons", None)]
    else:
        candidates = [(args.method, args.mesh)]
    for method, mesh in candidates:
        iters = 0 if method == "demons" else args.iters_bspline
        affine, composite, _, _ = register(fixed_ct, moving["ct"], fixed_mask, init, mesh or args.mesh,
                                           args.iters_affine, iters)
        if method == "demons":
            composite = register_demons(fixed_ct, moving["ct"], affine, shrinks=(8, 4, 2), iters=(100, 80, 40),
                                        sigma=args.demons_sigma)
        elif method.endswith("demons"):
            composite = register_demons(fixed_ct, moving["ct"], composite, sigma=args.demons_sigma)
        fold = jacobian_stats(composite, fixed_ct, body)["jac_fold_frac"]
        log(f"Registro ({method}) en {time.time() - t0:.0f} s, plegado {100 * fold:.2f} %", flush=True)
        if args.method != "adaptive" or fold <= getattr(args, "max_fold", 0.001):
            break

    report = {"method": method, "scale_child_over_adult": (1 / scale).round(3).tolist(),
              "child_crop_z": [z0, z1], "stages": {}}
    for stage, tr in (("landmarks", init), ("afin", affine), ("no_rigido", composite)):
        w_ct = to_numpy(warp(moving["ct"], fixed_ct, tr, sitk.sitkLinear, -1024.0))
        w_mask = to_numpy(warp(moving["mask"], fixed_mask, tr, sitk.sitkNearestNeighbor, 0))
        report["stages"][stage] = qa(w_mask, w_ct, child["mask"], child["ct"])
        if stage == "no_rigido":
            report["stages"][stage].update(jacobian_stats(tr, fixed_ct, child["mask"] > 0))
        log(f"  {stage:9s} " + ", ".join(f"{k} {v:.0f} HU" if k.startswith("mae") else f"{k} {v:.3f}"
                                         for k, v in report["stages"][stage].items()), flush=True)

    warped = {
        "ct": w_ct,
        "mask": w_mask,
        "pet": to_numpy(warp(moving["pet"], fixed_ct, composite, sitk.sitkLinear, 0.0)),
        "mri": np.stack([to_numpy(warp(m, fixed_ct, composite, sitk.sitkLinear, 0.0)) for m in moving_mri]),
    }
    report["seconds"] = round(time.time() - t0)
    return warped, report, child


if __name__ == "__main__":
    main()
