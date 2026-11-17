"""
prepare_utrecht.py

Pipeline para la cohorte pediátrica de UMC Utrecht: DICOM (export XNAT) -> NIfTI
nativo -> NIfTI en el espacio de datos de BigMAC, con la misma estructura de
carpetas que BigMAC para reutilizar generate_hdf5_utrecht.py e inference.py.

Espacio de datos de BigMAC (medido en /mnt/d/bic-mac-data/train/*/ct-label/ct.nii.gz):
  - Malla del CT: 512 x 512 vóxeles en el plano, spacing 1.5234375 x 1.5234375 x 2.0 mm.
  - Orientación en disco LAS; generate_hdf5*.py la pasa a RAS con as_closest_canonical.
  - El NAC-PET está remuestreado a la malla del CT (su malla nativa es 3.2 x 3.2 x 1.65 mm).
  - CT en int16 (HU), NAC-PET en float32.

Qué se hace con cada examen de Utrecht:
  1. Lectura DICOM con SimpleITK (orden por ImagePositionPatient, rescale por corte) de las
     series CT_LD, TS_PET_NAC y TS_PET_AC_EARL2 -> <out>/<sub>/native/*.nii.gz.
     No se usa dcm2niix: en algunos PET ordena los cortes por FrameReferenceTime (falla con
     "InstanceNumberOrderIsNotSpatial") y toma 1.65 mm de espaciado en z donde las posiciones
     reales dan 1.6456 mm (~5 mm de desajuste PET/CT en un total-body).
  2. Malla destino (RAS): x/y = 512 vóxeles de 1.5234375 mm centrados en el centro del FOV del
     CT (como BigMAC); z = 2.0 mm cubriendo el solape de los FOV de CT y PET, recortado a la
     extensión del cuerpo (mayor componente conexa del CT > -300 HU, que excluye la camilla) + margen. El número de cortes es variable: Utrecht es
     total-body (FOV axial de ~1.8 m) y dataset.py / inference.py no necesitan 531 cortes.
  3. Remuestreo en espacio físico (antialias al reducir la resolución del CT), CT recortado
     a [-1024, 3071] y relleno con aire (-1024 HU) fuera de su FOV, y escritura en LAS como BigMAC:
        <out>/<sub>/ct-label/ct.nii.gz          (int16)
        <out>/<sub>/features/nacpet.nii.gz      (float32)
        <out>/<sub>/features/acpet.nii.gz       (float32, solo para evaluación)
  4. participants.csv con serie usada, tamaño de la malla y (si pydicom está instalado)
     edad, sexo y peso.

Entrada: carpeta con los zips PUMP-*.zip o con los exports ya descomprimidos
(PUMP-xxx/PUMP-xxx/scans/<serie>/resources/DICOM/files/*.dcm).

Uso:
    python -m data.prepare_utrecht --input /mnt/d/Data/007_pseudoCT/utrech_data \
        --out_dir /mnt/d/Data/007_pseudoCT/utrecht_bigmac_space

Owner: Edwing Ulin
Version: v1.0.0
"""

import argparse
import csv
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torchio as tio

from scipy.ndimage import label as cc_label

from utils.logger import logger_gen

LOGGER = logger_gen('prepare_utrecht')

try:
    import pydicom
except ImportError:  # edad/sexo/peso quedan vacíos en participants.csv
    pydicom = None

# Espacio de BigMAC (malla del CT)
BIGMAC_SPACING = (1.5234375, 1.5234375, 2.0)
BIGMAC_INPLANE = 512
BIGMAC_AXCODES = ("L", "A", "S")

CT_MIN_HU, CT_MAX_HU = -1024, 3071
BODY_HU = -300          # umbral de cuerpo en el CT (como hu_body_min en generate_hdf5_utrecht.py)
Z_MARGIN_MM = 10.0      # margen en z alrededor del cuerpo

# nombre de salida nativo -> regex sobre el nombre de la serie (ej. "5-TS_PET_NAC")
SERIES = {
    "ct": re.compile(r"CT_LD"),
    "nacpet": re.compile(r"^\d+-TS_PET_NAC$"),
    "acpet": re.compile(r"^\d+-TS_PET_AC_EARL2$"),
}
REQUIRED = ("ct", "nacpet")


# --------------------------------------------------------------------------- #
# Paso 1: localizar series y convertir DICOM -> NIfTI
# --------------------------------------------------------------------------- #
def series_number(name: str) -> int:
    return int(name.split("-", 1)[0])


def sub_id_from_name(name: str) -> str:
    """PUMP-108861_20241205_24197 -> sub-PUMP108861-20241205"""
    pump_id, date = name.split("_")[:2]
    return f"sub-{pump_id.replace('-', '')}-{date}"


def exam_name_from_zip(zpath: Path) -> Optional[str]:
    """
    Nombre del examen dentro del zip (carpeta PUMP-* que contiene scans/). El nombre del zip
    lleva la fecha de DESCARGA (ej. PUMP-117115_20250526_...) y no la del examen (20250522).
    """
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            m = re.search(r"(PUMP-[^/]+)/scans/", name)
            if m:
                return m.group(1)
    return None


def pick_series(all_series: List[str], sub_id: str) -> Dict[str, str]:
    chosen = {}
    for key, pattern in SERIES.items():
        matches = sorted((s for s in all_series if pattern.search(s)), key=series_number)
        if not matches:
            level = LOGGER.error if key in REQUIRED else LOGGER.warning
            level(f"{sub_id}: no hay serie para '{key}' (series: {all_series})")
            continue
        if len(matches) > 1:
            LOGGER.warning(f"{sub_id}: varias series para '{key}': {matches} -> uso {matches[0]} (revisar)")
        chosen[key] = matches[0]
    return chosen


def dicom_to_nifti(dicom_dir: Path, dest: Path) -> str:
    """
    Convierte una serie DICOM a .nii.gz con SimpleITK (cortes ordenados por posición,
    rescale slope/intercept de cada corte, salida float32). Devuelve las unidades (0054,1001).
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No hay series DICOM en {dicom_dir}")
    files = max((reader.GetGDCMSeriesFileNames(str(dicom_dir), sid) for sid in series_ids), key=len)
    if len(series_ids) > 1:
        LOGGER.warning(f"{dicom_dir}: {len(series_ids)} series UID; se usa la de más cortes ({len(files)})")

    reader.SetFileNames(files)
    reader.SetOutputPixelType(sitk.sitkFloat32)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    units = reader.GetMetaData(0, "0054|1001").strip() if reader.HasMetaDataKey(0, "0054|1001") else ""

    dest.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(dest))
    return units


def read_demographics(dicom_dir: Path) -> Dict[str, str]:
    if pydicom is None:
        return {}
    first = next(dicom_dir.glob("*.dcm"), None)
    if first is None:
        return {}
    ds = pydicom.dcmread(first, stop_before_pixels=True)
    return {
        "age": str(getattr(ds, "PatientAge", "")),
        "sex": str(getattr(ds, "PatientSex", "")),
        "weight_kg": str(getattr(ds, "PatientWeight", "")),
    }


def convert_folder(scans_dir: Path, native_dir: Path, sub_id: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Export ya descomprimido: scans/<serie>/resources/DICOM/files/*.dcm"""
    all_series = sorted((d.name for d in scans_dir.iterdir() if d.is_dir()), key=series_number)
    chosen = pick_series(all_series, sub_id)
    info = {}
    for key, series in chosen.items():
        dicom_dir = scans_dir / series / "resources" / "DICOM" / "files"
        units = dicom_to_nifti(dicom_dir, native_dir / f"{key}.nii.gz")
        if key == "nacpet":
            info = {"nac_units": units, **read_demographics(dicom_dir)}
    return chosen, info


def convert_zip(zpath: Path, native_dir: Path, sub_id: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Zip XNAT: extrae solo las series necesarias a un temporal y las convierte."""
    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as tmp:
        names = zf.namelist()
        all_series = sorted(
            {m.group(1) for n in names if (m := re.search(r"scans/([^/]+)/", n))},
            key=series_number,
        )
        chosen = pick_series(all_series, sub_id)
        info = {}
        for key, series in chosen.items():
            members = [n for n in names if f"scans/{series}/" in n and n.lower().endswith(".dcm")]
            zf.extractall(tmp, members)
            dicom_dir = Path(tmp) / Path(members[0]).parent
            units = dicom_to_nifti(dicom_dir, native_dir / f"{key}.nii.gz")
            if key == "nacpet":
                info = {"nac_units": units, **read_demographics(dicom_dir)}
    return chosen, info


# --------------------------------------------------------------------------- #
# Paso 2: llevar al espacio de BigMAC
# --------------------------------------------------------------------------- #
def world_z_range(img: tio.ScalarImage) -> Tuple[float, float]:
    """Extensión en z (RAS, mm) de los centros de vóxel de una imagen."""
    shape = np.array(img.spatial_shape)
    corners = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)])
    world = corners @ img.affine[:3, :3].T + img.affine[:3, 3]
    return float(world[:, 2].min()), float(world[:, 2].max())


def fov_center(img: tio.ScalarImage) -> np.ndarray:
    shape = np.array(img.spatial_shape)
    return img.affine[:3, :3] @ ((shape - 1) / 2) + img.affine[:3, 3]


def make_grid(ct: tio.ScalarImage, pet: tio.ScalarImage) -> Tuple[Tuple[int, int, int], np.ndarray]:
    """Malla RAS de BigMAC: x/y centrados en el FOV del CT, z = solape de los FOV de CT y PET."""
    ct_z, pet_z = world_z_range(ct), world_z_range(pet)
    z0, z1 = max(ct_z[0], pet_z[0]), min(ct_z[1], pet_z[1])
    if z1 <= z0:
        raise ValueError(f"Los FOV de CT {ct_z} y PET {pet_z} no se solapan en z")
    sx, sy, sz = BIGMAC_SPACING
    nz = int(np.floor((z1 - z0) / sz)) + 1
    center = fov_center(ct)
    affine = np.diag([sx, sy, sz, 1.0])
    affine[0, 3] = center[0] - sx * (BIGMAC_INPLANE - 1) / 2
    affine[1, 3] = center[1] - sy * (BIGMAC_INPLANE - 1) / 2
    affine[2, 3] = z0
    return (BIGMAC_INPLANE, BIGMAC_INPLANE, nz), affine


def resample(img: tio.ScalarImage, grid, antialias: bool, pad_value: float = 0.0) -> np.ndarray:
    """
    Remuestrea en espacio físico. TorchIO rellena con 0 fuera del FOV de origen,
    así que se desplaza la imagen para que ese relleno equivalga a pad_value
    (p. ej. -1024 HU en el CT; sin esto, fuera del FOV del CT quedaría agua).
    """
    if pad_value != 0.0:
        img = tio.ScalarImage(tensor=img.data - pad_value, affine=img.affine)
    out = tio.Resample(target=grid, image_interpolation="linear", antialias=antialias)(img)
    return out.data[0].numpy() + pad_value


def body_z_crop(ct: np.ndarray) -> Tuple[int, int]:
    """
    Cortes [k0, k1) con cuerpo, con margen. Cuerpo = mayor componente conexa del CT
    por encima de BODY_HU (el mismo criterio que la máscara de generate_hdf5_utrecht.py):
    así se excluye la camilla, que es un objeto separado. El NAC-PET no sirve aquí:
    su halo y su ruido de fondo ocupan todos los cortes.
    """
    step = 2  # componentes conexas sobre el volumen submuestreado, por memoria y tiempo
    labeled, n = cc_label(ct[::step, ::step, ::step] > BODY_HU)
    if n == 0:
        raise ValueError("No se encontró cuerpo en el CT remuestreado")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    body_slices = np.where((labeled == sizes.argmax()).any(axis=(0, 1)))[0]
    margin = int(round(Z_MARGIN_MM / BIGMAC_SPACING[2]))
    k0 = int(body_slices[0]) * step - margin
    k1 = (int(body_slices[-1]) + 1) * step + margin
    return max(0, k0), min(ct.shape[2], k1)


def save_las(data: np.ndarray, ras_affine: np.ndarray, path: Path, dtype) -> None:
    """Guarda en orientación LAS, como los NIfTI de BigMAC."""
    img = nib.Nifti1Image(data.astype(dtype), ras_affine)
    transform = nib.orientations.ornt_transform(
        nib.io_orientation(ras_affine), nib.orientations.axcodes2ornt(BIGMAC_AXCODES)
    )
    las = img.as_reoriented(transform)
    las.set_qform(las.affine, code=1)
    las.set_sform(las.affine, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(las, path)


def to_bigmac_space(native_dir: Path, sub_dir: Path) -> Dict[str, object]:
    ct_img = tio.ScalarImage(native_dir / "ct.nii.gz")
    nac_img = tio.ScalarImage(native_dir / "nacpet.nii.gz")
    grid_shape, grid_affine = make_grid(ct_img, nac_img)

    ct_img = tio.ScalarImage(tensor=ct_img.data.clamp(CT_MIN_HU, CT_MAX_HU), affine=ct_img.affine)
    ct = resample(ct_img, (grid_shape, grid_affine), antialias=True, pad_value=CT_MIN_HU)
    ct = np.clip(ct, CT_MIN_HU, CT_MAX_HU)
    k0, k1 = body_z_crop(ct)
    affine = grid_affine.copy()
    affine[2, 3] += k0 * BIGMAC_SPACING[2]
    final_grid = ((BIGMAC_INPLANE, BIGMAC_INPLANE, k1 - k0), affine)

    save_las(np.round(ct[:, :, k0:k1]), affine, sub_dir / "ct-label" / "ct.nii.gz", np.int16)
    del ct, ct_img

    nac = resample(nac_img, final_grid, antialias=False)
    save_las(nac, affine, sub_dir / "features" / "nacpet.nii.gz", np.float32)
    del nac, nac_img

    if (native_dir / "acpet.nii.gz").exists():
        ac = resample(tio.ScalarImage(native_dir / "acpet.nii.gz"), final_grid, antialias=False)
        save_las(ac, affine, sub_dir / "features" / "acpet.nii.gz", np.float32)

    return {"grid_shape": f"{BIGMAC_INPLANE}x{BIGMAC_INPLANE}x{k1 - k0}", "z_length_mm": (k1 - k0) * BIGMAC_SPACING[2]}


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def find_exams(input_dir: Path) -> List[Tuple[str, Path, str]]:
    """Devuelve (sub_id, ruta, tipo) para cada examen; prioriza la carpeta descomprimida sobre el zip."""
    exams: Dict[str, Tuple[Path, str]] = {}
    for zpath in sorted(input_dir.glob("PUMP-*.zip")):
        if "raw_data" in zpath.name or "List_mode" in zpath.name:
            continue
        exam_name = exam_name_from_zip(zpath)
        if exam_name is None:
            LOGGER.warning(f"{zpath.name}: sin carpeta scans/, se omite (revisar a mano)")
            continue
        exams[sub_id_from_name(exam_name)] = (zpath, "zip")
    for scans_dir in sorted(input_dir.glob("PUMP-*/PUMP-*/scans")):
        exams[sub_id_from_name(scans_dir.parent.name)] = (scans_dir, "folder")
    return [(sub_id, path, kind) for sub_id, (path, kind) in sorted(exams.items())]


def process_exam(sub_id: str, path: Path, kind: str, out_dir: Path, keep_native: bool) -> Optional[Dict[str, object]]:
    sub_dir = out_dir / sub_id
    native_dir = sub_dir / "native"
    if (sub_dir / "features" / "nacpet.nii.gz").exists() and (sub_dir / "ct-label" / "ct.nii.gz").exists():
        LOGGER.info(f"{sub_id}: ya procesado, se omite (borra la carpeta para rehacerlo)")
        return None

    LOGGER.info(f"{sub_id}: convirtiendo DICOM desde {kind} {path.name}")
    if kind == "zip":
        chosen, info = convert_zip(path, native_dir, sub_id)
    else:
        chosen, info = convert_folder(path, native_dir, sub_id)
    missing = [k for k in REQUIRED if k not in chosen]
    if missing:
        raise RuntimeError(f"faltan series obligatorias {missing}")

    LOGGER.info(f"{sub_id}: remuestreando al espacio de BigMAC")
    grid = to_bigmac_space(native_dir, sub_dir)
    if not keep_native:
        shutil.rmtree(native_dir)

    row = {"sub_id": sub_id, "patient": sub_id.split("-")[1], "source": path.name,
           "ct_series": chosen.get("ct", ""), "nac_series": chosen.get("nacpet", ""),
           "ac_series": chosen.get("acpet", ""), **info, **grid}
    LOGGER.info(f"{sub_id}: OK {grid}")
    return row


def main(input_dir: Path, out_dir: Path, only: Optional[List[str]], keep_native: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    exams = find_exams(input_dir)
    if only:
        exams = [e for e in exams if e[0] in only or e[0].split("-")[1] in only]
    LOGGER.info(f"{len(exams)} exámenes a procesar")
    if pydicom is None:
        LOGGER.warning("pydicom no está instalado: participants.csv no tendrá edad/sexo/peso (pip install pydicom)")

    rows, failed = [], []
    for sub_id, path, kind in exams:
        try:
            row = process_exam(sub_id, path, kind, out_dir, keep_native)
            if row:
                rows.append(row)
        except Exception as e:
            LOGGER.error(f"{sub_id}: {e}")
            failed.append(sub_id)

    fields = ["sub_id", "patient", "source", "ct_series", "nac_series", "ac_series", "nac_units",
              "grid_shape", "z_length_mm", "age", "sex", "weight_kg"]
    csv_path = out_dir / "participants.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(f"Terminado: {len(rows)} procesados, {len(failed)} fallidos {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("UMC Utrecht: DICOM -> NIfTI en el espacio de BigMAC")
    parser.add_argument("--input", type=Path, default=Path("/mnt/d/Data/007_pseudoCT/utrech_data"),
                        help="Carpeta con PUMP-*.zip y/o exports descomprimidos PUMP-*/PUMP-*/scans")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Procesar solo estos sub_id o pacientes (ej. PUMP108861)")
    parser.add_argument("--keep_native", action="store_true",
                        help="Conservar los NIfTI nativos (geometría DICOM original) en <sub>/native/")
    args = parser.parse_args()
    main(args.input, args.out_dir, args.only, args.keep_native)
