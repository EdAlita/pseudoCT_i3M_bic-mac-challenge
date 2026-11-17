"""
dataset.py

Este módulo define las clases de Dataset para el entrenamiento y validación
de síntesis de pseudo-CT multimodal usando archivos HDF5.

Owner: Edwing Ulin
Version: v1.0.0
"""

import csv
import re

import torch
import h5py
import numpy as np
import yacs.config
import torchio as tio
from scipy import ndimage as ndi
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple
from utils.logger import logger_gen
from data.preprocessing import normalize_pet, normalize_mri, normalize_ct, patch_coords

logger = logger_gen(__name__)

class MultimodalpCTDataset(Dataset):
    """
    Dataset optimizado para síntesis de pseudo-CT a partir de PET, MRI y Topogramas.

    Soporta "MRI dropout": con probabilidad cfg.DATA.MRI_DROPOUT_PROB (train) /
    cfg.DATA.MRI_DROPOUT_PROB_VAL (val), el canal de MRI se pone a cero para que
    el modelo aprenda a sintetizar CT únicamente a partir de PET, cubriendo el
    caso de cohortes sin MRI disponible (ej. pacientes pediátricos de UMC Utrecht).
    """

    def __init__(
        self,
        h5_paths: List[str],
        cfg: yacs.config.CfgNode,
        mode: str = 'train',
        transforms: Optional = None
    ):
        """
        Args:
            h5_paths: Lista de rutas a archivos .h5 (ej. los 4 folds de train).
            cfg: Configuración de YACS.
            mode: 'train' para parches aleatorios, 'val' para parches fijos o volumen.
            transforms: Transformaciones (Data Augmentation).
        """
        self.cfg = cfg
        self.mode = mode
        self.transforms = transforms
        self.h5_files = h5_paths
        
        # Cohorte extra (solo train): se muestrea con probabilidad EXTRA_TRAIN_FRAC por parche.
        self.extra_files = list(getattr(cfg.DATA, "EXTRA_TRAIN_FILES", [])) if mode == 'train' else []
        self.extra_frac = float(getattr(cfg.DATA, "EXTRA_TRAIN_FRAC", 0.0)) if self.extra_files else 0.0
        # Cohorte pediátrica deformada (solo train), con su propia probabilidad por parche.
        self.ped_files = list(getattr(cfg.DATA, "PED_TRAIN_FILES", [])) if mode == 'train' else []
        self.ped_frac = float(getattr(cfg.DATA, "PED_TRAIN_FRAC", 0.0)) if self.ped_files else 0.0

        self.subjects_data = []
        self._load_metadata()
        self.main_idx = [i for i, m in enumerate(self.subjects_data) if m["cohort"] == "main"]
        self.extra_idx = [i for i, m in enumerate(self.subjects_data) if m["cohort"] == "extra"]
        self.ped_idx = [i for i, m in enumerate(self.subjects_data) if m["cohort"] == "ped"]
        if self.ped_files and not self.ped_idx:
            logger.warning("Cohorte pediátrica sin parejas válidas: PED_TRAIN_FRAC pasa a 0.")
            self.ped_frac = 0.0
        if self.extra_idx or self.ped_idx:
            logger.info(f"Muestreo mixto: {len(self.main_idx)} sujetos principales, "
                        f"{len(self.extra_idx)} extra con p={self.extra_frac:.2f}, "
                        f"{len(self.ped_idx)} pediátricos deformados con p={self.ped_frac:.2f} por parche.")
        self._setup_targeted_sampling()

        # Escala pediátrica: se lee una región mayor (parche / factor) y se reduce al tamaño
        # del parche, así la anatomía adulta se ve más pequeña sin bordes de relleno.
        self.shrink_prob = float(getattr(cfg.DATA, "EXTRA_SHRINK_PROB", 0.0)) if self.extra_idx else 0.0
        if self.shrink_prob > 0:
            lo, hi = cfg.DATA.EXTRA_SHRINK_SCALE
            logger.info(f"Escala pediátrica en la cohorte extra: p={self.shrink_prob:.2f}, factor {lo}-{hi}.")

        self.domain_prob = float(getattr(cfg.DATA, "PET_DOMAIN_PROB", 0.0)) if mode == 'train' else 0.0
        ped_prob = float(getattr(cfg.DATA, "PET_DOMAIN_PROB_PED", -1.0))
        self.domain_prob_ped = (ped_prob if ped_prob >= 0 else self.domain_prob) if mode == 'train' else 0.0
        if self.domain_prob > 0 or self.domain_prob_ped > 0:
            logger.info(f"Augmentation de dominio del NAC-PET: p={self.domain_prob:.2f} en extra, "
                        f"p={self.domain_prob_ped:.2f} en pediátrica.")

    def _setup_targeted_sampling(self):
        """Muestreo dirigido de la cohorte principal en train: tórax y sujetos jóvenes."""
        train = self.mode == 'train'
        self.thorax_frac = float(getattr(self.cfg.DATA, "THORAX_FRAC", 0.0)) if train else 0.0
        self.young_frac = float(getattr(self.cfg.DATA, "YOUNG_FRAC", 0.0)) if train else 0.0
        self.young_idx, self.old_idx = [], []

        if self.thorax_frac > 0:
            from utils.body_coords import lung_z_range
            thorax_pool = list(self.main_idx or range(len(self.subjects_data))) + self.ped_idx
            for i in thorax_pool:
                meta = self.subjects_data[i]
                with h5py.File(meta["path"], 'r') as hf:
                    mask = hf[meta["sub_id"]]["mask"][:]
                lung_z = lung_z_range(mask, contiguous=getattr(self.cfg.DATA, "LUNG_CONTIGUOUS", False))
                if lung_z is not None:
                    # Caja de la clase 3 dentro de la losa pulmonar (los tres ejes).
                    air = mask[:, :, lung_z[0]:lung_z[1]] == 3
                    lo, hi = [], []
                    for ax in range(2):
                        idx = np.where(air.any(axis=tuple(a for a in range(3) if a != ax)))[0]
                        lo.append(int(idx[0]))
                        hi.append(int(idx[-1]) + 1)
                    meta["lung_box"] = (lo + [lung_z[0]], hi + [lung_z[1]])
                del mask
            n = sum("lung_box" in self.subjects_data[i] for i in thorax_pool)
            logger.info(f"Muestreo de tórax: p={self.thorax_frac:.2f}, losa pulmonar en {n} sujetos.")

        if self.young_frac > 0:
            path = getattr(self.cfg.DATA, "PARTICIPANTS_CSV", "")
            ages = {}
            for r in csv.DictReader(open(path)):
                m = re.match(r"(\d+)Y", r.get("age", "") or "")
                if m:
                    ages[r["sub_id"]] = int(m.group(1))
            pool = self.main_idx or list(range(len(self.subjects_data)))
            self.young_idx = [i for i in pool if ages.get(self.subjects_data[i]["sub_id"], 99) < self.cfg.DATA.YOUNG_AGE]
            self.old_idx = [i for i in pool if i not in self.young_idx]
            if not self.young_idx or not self.old_idx:
                self.young_frac = 0.0
            logger.info(f"Muestreo por edad: {len(self.young_idx)} sujetos < {self.cfg.DATA.YOUNG_AGE} años "
                        f"con p={self.young_frac:.2f}, {len(self.old_idx)} mayores.")

    def _load_metadata(self):
        """Mapea los sujetos disponibles en los archivos HDF5 sin cargarlos a RAM."""
        for path in list(self.h5_files) + self.extra_files + self.ped_files:
            if path in self.h5_files:
                cohort = "main"
            else:
                cohort = "extra" if path in self.extra_files else "ped"
            with h5py.File(path, 'r') as hf:
                for sub_id in hf.keys():
                    # Solo se leen atributos: el volumen sigue sin cargarse a RAM.
                    attrs = hf[sub_id].attrs
                    if cohort == "ped" and not bool(attrs.get("qa_ok", True)):
                        logger.info(f"Pareja deformada {sub_id} sin qa_ok: excluida.")
                        continue
                    if cohort == "ped" and float(attrs.get("child_age", 0)) >= self.cfg.DATA.PED_MAX_AGE:
                        continue
                    bbox = None
                    if "body_lo" in attrs:
                        bbox = (np.asarray(attrs["body_lo"]), np.asarray(attrs["body_hi"]))
                    elif self.cfg.DATA.COORDS_MODE == "body":
                        raise KeyError(
                            f"{sub_id} en {path} no tiene body_lo/body_hi: ejecuta utils/add_body_bbox.py"
                        )

                    self.subjects_data.append({
                        "path": path,
                        "sub_id": sub_id,
                        "has_mri": bool(attrs.get("has_mri", True)),    # BigMAC no tiene el atributo -> True
                        "has_topo": bool(attrs.get("has_topo", True)),
                        "bbox": bbox,
                        "cohort": cohort,
                        "extra": cohort == "extra",
                    })

        logger.info(f"Dataset {self.mode} cargado con {len(self.subjects_data)} sujetos.")

    def __len__(self) -> int:
        if self.mode == 'train':
            return self.cfg.TRAIN.BATCHES_PER_EPOCH * self.cfg.TRAIN.BATCH_SIZE
        else:
            return self.cfg.TRAIN.VAL_PATCHES

    @staticmethod
    def _resize_patches(size, pet, mri, ct, mask):
        """Reduce los parches leídos a `size`: trilineal en imágenes, vecino más cercano en la máscara."""
        F = torch.nn.functional

        def img(a):
            t = torch.from_numpy(np.asarray(a, dtype=np.float32))
            t = t[None, None] if t.dim() == 3 else t[None]
            return F.interpolate(t, size=size, mode="trilinear", align_corners=False)[0].numpy()

        m = torch.from_numpy(np.asarray(mask).astype(np.float32))[None, None]
        mask_r = F.interpolate(m, size=size, mode="nearest")[0, 0].numpy().astype(np.asarray(mask).dtype)
        return img(pet)[0], img(mri), img(ct)[0], mask_r

    def _pet_domain_aug(self, pet, body):
        """
        NAC-PET con aspecto de Utrecht: centro más oscuro y halo de scatter fuera del cuerpo.

        La distancia a la piel se calcula en el parche (a media resolución); el borde del parche
        se extiende por replicación para que no cuente como piel.
        """
        if not body.any() or body.all():
            return pet

        def U(key):
            lo, hi = getattr(self.cfg.DATA, key)
            return lo + (hi - lo) * torch.rand(1).item()

        dark, depth_mm, rim_mm = U("PET_DOMAIN_DARK"), U("PET_DOMAIN_DEPTH_MM"), U("PET_DOMAIN_RIM_MM")
        halo, halo_mm = U("PET_DOMAIN_HALO"), U("PET_DOMAIN_HALO_MM")
        vox = 2 * float(self.cfg.DATA.VOXEL_SIZES[0])
        b2 = body[::2, ::2, ::2]
        pad = int(np.ceil(max(rim_mm + depth_mm, halo_mm) / vox)) + 1
        bp = np.pad(b2, pad, mode="edge")
        crop = tuple(slice(pad, pad + n) for n in b2.shape)
        d_in = ndi.distance_transform_edt(bp)[crop] * vox
        d_out = ndi.distance_transform_edt(~bp)[crop] * vox

        def up(a):
            a = a.repeat(2, 0).repeat(2, 1).repeat(2, 2)
            return a[:pet.shape[0], :pet.shape[1], :pet.shape[2]]

        d_in, d_out = up(d_in), up(d_out)
        ref = float(np.percentile(pet[body], 95))
        gain = np.where(body, 1.0 - dark * np.clip((d_in - rim_mm) / depth_mm, 0.0, 1.0), 1.0)
        add = np.where(body, 0.0, halo * ref * np.exp(-d_out / halo_mm))
        return (pet * gain + add).astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            if self.extra_idx or self.ped_idx or self.young_frac > 0:
                # Mixto: primero la cohorte (y el grupo de edad), después un sujeto al azar.
                pool = self.main_idx or list(range(len(self.subjects_data)))
                r = torch.rand(1).item()
                if self.ped_idx and r < self.ped_frac:
                    pool = self.ped_idx
                elif self.extra_idx and r < self.ped_frac + self.extra_frac:
                    pool = self.extra_idx
                elif self.young_frac > 0:
                    pool = self.young_idx if torch.rand(1).item() < self.young_frac else self.old_idx
                sub_idx = pool[torch.randint(0, len(pool), (1,)).item()]
            else:
                sub_idx = index % len(self.subjects_data)

            meta = self.subjects_data[sub_idx]
            h5_path, sub_id = meta["path"], meta["sub_id"]

            with h5py.File(h5_path, 'r') as hf:
                group = hf[sub_id]

                pet = group['pet']
                mri = group['mri'] if meta["has_mri"] else None      # Utrecht no guarda MRI
                if self.mode == 'train' and self.cfg.DATA.MRI_DROPOUT_PROB >= 1.0:
                    mri = None      # se descartaría siempre: no se lee (2 canales, el más lento)
                ct = group['ct']
                mask = group['mask']
                topo = group['topo'][:] if meta["has_topo"] else None

                depth, height, width = pet.shape
                out_size = tuple(self.cfg.TRAIN.PATCH_SIZE)
                p_d, p_h, p_w = out_size
                if self.shrink_prob > 0 and meta["extra"] and torch.rand(1).item() < self.shrink_prob:
                    lo, hi = self.cfg.DATA.EXTRA_SHRINK_SCALE
                    factor = lo + (hi - lo) * torch.rand(1).item()
                    # Tamaño de lectura (parche / factor); al final se reduce a out_size.
                    p_d, p_h, p_w = (min(int(round(p / factor)), n)
                                     for p, n in zip(out_size, (depth, height, width)))
                # Tórax: el centro del parche dentro de la caja pulmonar en los tres ejes.
                lung_box = meta.get("lung_box")
                in_thorax = (self.mode == 'train' and lung_box is not None
                             and torch.rand(1).item() < self.thorax_frac)

                def start_in(ax, size, patch):
                    """Inicio aleatorio con el centro del parche dentro de la caja pulmonar."""
                    lo = min(max(lung_box[0][ax] - patch // 2, 0), size - patch)
                    hi = min(max(lung_box[1][ax] - patch // 2, lo), size - patch)
                    return torch.randint(lo, hi + 1, (1,)).item()

                for attempt in range(15):
                    if self.mode == 'train' and in_thorax:
                        z, y, x = start_in(0, depth, p_d), start_in(1, height, p_h), start_in(2, width, p_w)
                    elif self.mode == 'train':
                        z = torch.randint(0, depth - p_d + 1, (1,)).item()
                        y = torch.randint(0, height - p_h + 1, (1,)).item()
                        x = torch.randint(0, width - p_w + 1, (1,)).item()
                    else:
                        rng = np.random.default_rng(index+attempt) 
                        z = rng.integers(0, depth - p_d + 1)
                        y = rng.integers(0, height - p_h + 1)
                        x = rng.integers(0, width - p_w + 1)

                    # Primero la máscara (la más barata): el resto solo para el parche aceptado.
                    mask_patch = mask[z:z+p_d, y:y+p_h, x:x+p_w]
                    if np.mean(mask_patch > 0) > 0.05:
                        break

                pet_patch = pet[z:z+p_d, y:y+p_h, x:x+p_w]
                if mri is not None:
                    mri_patch = mri[:, z:z+p_d, y:y+p_h, x:x+p_w]
                else:
                    mri_patch = np.zeros((2, p_d, p_h, p_w), dtype=np.float32)
                ct_patch = ct[z:z+p_d, y:y+p_h, x:x+p_w]

                if (p_d, p_h, p_w) != out_size:
                    pet_patch, mri_patch, ct_patch, mask_patch = self._resize_patches(
                        out_size, pet_patch, mri_patch, ct_patch, mask_patch)

                p_dom = {"extra": self.domain_prob, "ped": self.domain_prob_ped}.get(meta["cohort"], 0.0)
                if p_dom > 0 and torch.rand(1).item() < p_dom:
                    pet_patch = self._pet_domain_aug(np.asarray(pet_patch, dtype=np.float32),
                                                     np.asarray(mask_patch) > 0)
                
                bbox = meta["bbox"] if self.cfg.DATA.COORDS_MODE == "body" else None
                coords = torch.from_numpy(
                    patch_coords((z, y, x), (p_d, p_h, p_w), (depth, height, width), bbox)
                )
            
            pet_patch_norm = normalize_pet(pet_patch)
            mri_patch_norm = normalize_mri(mri_patch)
            ct_patch_norm = normalize_ct(ct_patch, self.cfg)

            if self.mode == 'train':
                mri_dropped = torch.rand(1).item() < self.cfg.DATA.MRI_DROPOUT_PROB
                topo_dropped = torch.rand(1).item() < self.cfg.DATA.TOPO_DROPOUT_PROB
            else:
                # RNG independiente del muestreo de parches (que usa index+attempt),
                # para que la decisión de dropout sea estable entre épocas. El primer
                # sorteo es el del MRI, igual que antes de añadir el del topograma.
                rng_drop = np.random.default_rng(index)
                mri_dropped = rng_drop.random() < getattr(self.cfg.DATA, "MRI_DROPOUT_PROB_VAL", 0.0)
                topo_dropped = rng_drop.random() < getattr(self.cfg.DATA, "TOPO_DROPOUT_PROB_VAL", 0.0)

            # Una modalidad ausente en el sujeto cuenta siempre como "dropped".
            mri_dropped = mri_dropped or not meta["has_mri"]
            topo_dropped = topo_dropped or not meta["has_topo"]

            if mri_dropped:
                mri_patch_norm = np.zeros_like(mri_patch_norm)

            subject = tio.Subject(
                pet=tio.ScalarImage(tensor=torch.from_numpy(pet_patch_norm).unsqueeze(0).float()),
                mri=tio.ScalarImage(tensor=torch.from_numpy(mri_patch_norm).float()),
                target=tio.ScalarImage(tensor=torch.from_numpy(ct_patch_norm).unsqueeze(0).float()),
                batch_mask=tio.LabelMap(tensor=torch.from_numpy(mask_patch).unsqueeze(0).long()),
            )

            if self.transforms:
                subject = self.transforms(subject)
                applied = [t.name for t in getattr(subject, "history", [])]
                if any("Flip" in name for name in applied):
                    axis = self.cfg.DATA.FLIP_AXIS
                    coords[axis] = 1.0 - coords[axis]

            if topo_dropped:
                topo_clean = np.zeros(self.cfg.DATA.TOPO_SHAPE, dtype=np.float32)
            else:
                topo_clean = np.squeeze(topo)

            clean_mask = subject["batch_mask"].data.float()

            data = {
                "pet": subject["pet"].data,
                "mri": subject["mri"].data,
                "target": subject["target"].data,
                "topo": torch.from_numpy(topo_clean).unsqueeze(0).float(),
                "coords": coords,
                "batch_mask": clean_mask,
                "mri_dropped": torch.tensor(float(mri_dropped)),
                "topo_dropped": torch.tensor(float(topo_dropped)),
            }

            return data