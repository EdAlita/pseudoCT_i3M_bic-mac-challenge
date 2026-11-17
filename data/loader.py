"""
loader.py

Este módulo configura los DataLoaders de PyTorch para el entrenamiento y validación.
Incluye visualización para depuración de parches.

Owner: Edwing Ulin
Version: v1.3.0
"""

from torch.utils.data import DataLoader
import yacs.config
import torchio as tio

from data import dataset as dset
from utils.logger import logger_gen
from config.defaults import get_cfg_defaults

logger = logger_gen(__name__)

def get_dataloader(cfg: yacs.config.CfgNode, mode: str, fold_paths: list):
    """
    Configura el DataLoader basado en el modo (train/val).
    
    Args:
        cfg: Nodo de configuración YACS.
        mode: 'train' o 'val'.
        fold_paths: Lista de rutas a los archivos .h5 para este proceso.
    """
    assert mode in ['train', 'val'], f'Modo de dataloader inválido: {mode}'

    # Definición de qué canales reciben qué transformaciones
    include_all = ["pet", "mri", "target", "batch_mask"]
    include_images = ["pet", "mri"]

    if mode == 'train':
        apply_augs = cfg.DATA.APPLY_AUGMENTATION
        shuffle = True
        logger.info(f"Configurando {mode.upper()} dataloader con {len(fold_paths)} folds.")

        if apply_augs:
            logger.info("-> Data Augmentation: ACTIVADO")
            # 1. Transformaciones Geométricas
            scaling = tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=0,
                isotropic=True,
                image_interpolation="linear",
                include=include_all,
            )

            rotation = tio.RandomAffine(
                scales=(1.0, 1.0),
                degrees=10,
                image_interpolation="linear",
                include=include_all
            )

            # DATA.FLIP_AXIS: 0 = izquierda-derecha; 2 (original) volteaba cabeza-pies.
            flip = tio.RandomFlip(
                axes=(cfg.DATA.FLIP_AXIS,),
                flip_probability=0.5,
                include=include_all
            )

            # 2. Transformaciones de Intensidad
            bias_field = tio.RandomBiasField(
                coefficients=0.5, 
                order=3, 
                include=include_images,
            )

            noise = tio.RandomNoise(
                mean=0,
                std=(0.01, 0.05),
                include=include_images
            )

            geometric = tio.OneOf([flip, scaling, rotation], p=0.5)
            intensity = tio.OneOf([bias_field, noise], p=0.3)
            transform = tio.Compose([geometric, intensity])
        else:
            logger.info("-> Data Augmentation: DESACTIVADO (Modo Raw)")
            transform = None # Sin aumentación

        dataset = dset.MultimodalpCTDataset(h5_paths=fold_paths, cfg=cfg, mode='train', transforms=transform)
    
    else:
        shuffle = False
        logger.info(f"Configurando {mode.upper()} dataloader para validación.")
        dataset = dset.MultimodalpCTDataset(h5_paths=fold_paths, cfg=cfg, mode='val', transforms=None)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return dataloader

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    cfg = get_cfg_defaults()
    test_folds = ["/mnt/d/Data/007_pseudoCT/data/fold_0.h5"] 
    
    loader = get_dataloader(cfg, mode='val', fold_paths=test_folds)
    batch = next(iter(loader))
    print("Loader length:", len(loader))
    print("Batch keys:", batch.keys())
    print(f"Batch Size: {batch['pet'].shape[0]} samples")
    print(f"Batch PET shape: {batch['pet'].shape}")
    print(f"Batch MRI shape: {batch['mri'].shape}")
    print(f"Batch Mask shape: {batch['batch_mask'].shape}")
    print(f"Batch CT Target shape: {batch['target'].shape}")
    print(f"Batch Coords: {batch['coords']}")

    batch_size = batch['pet'].shape[0]
    

    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 3.5 * batch_size), constrained_layout=True)
    
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(batch_size):
        pet_vol = batch['pet'][i, 0].numpy()
        mri_vol = batch['mri'][i, 0].numpy()
        ct_vol = batch['target'][i, 0].numpy()
        mask_vol = batch['batch_mask'][i, 0].numpy()
        topo_img = batch['topo'][i, 0].squeeze().numpy() 

        mid_z = pet_vol.shape[0] // 2
        pet_slice = pet_vol[mid_z, :, :]
        mri_slice = mri_vol[mid_z, :, :]
        ct_slice = ct_vol[mid_z, :, :]
        mask_slice = mask_vol[mid_z, :, :]
        
        z_coord, y_coord, x_coord = batch['coords'][i].numpy()
        coord_str = f"[{z_coord:.2f}, {y_coord:.2f}, {x_coord:.2f}]"
        
        im0 = axes[i, 0].imshow(pet_slice, cmap='hot')
        axes[i, 0].set_title(f"B{i} PET | Coords: {coord_str}")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        axes[i, 0].axis('off')
        
        im1 = axes[i, 1].imshow(mri_slice, cmap='gray')
        axes[i, 1].set_title(f"B{i} MRI (Z={mid_z})")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        axes[i, 1].axis('off')
        
        im2 = axes[i, 2].imshow(ct_slice, cmap='gray')
        axes[i, 2].set_title(f"B{i} CT Target")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        axes[i, 2].axis('off')

        im3 = axes[i, 3].imshow(mask_slice, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(f"B{i} Mask")
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
        axes[i, 3].axis('off')

        print(f"Batch {i} - PET min/max: {pet_vol.min():.3f}/{pet_vol.max():.3f}")
        print(f"Batch {i} - MRI min/max: {mri_vol.min():.3f}/{mri_vol.max():.3f}")
        print(f"Batch {i} - CT Target min/max: {ct_vol.min():.3f}/{ct_vol.max():.3f}")
        print(f"Batch {i} - Batch Mask min/max: {mask_vol.min():.3f}/{mask_vol.max():.3f}")
    
    plt.show()