"""
defaults.py

This module sets up the default configuration for training and evaluating 3D multimodal
pseudo-CT synthesis models using the yacs library.

Dependencies:
    - yacs.config: For hierarchical configuration management.

Owner:
    Edwing Ulin

Version:
    v1.2.0
"""
from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Name of model
_C.MODEL.MODEL_NAME = "MultimodalPCT"

# Loss function: 'l1' or 'multimodal_pct'
_C.MODEL.LOSS_FUNC = "multimodal_pct"

# Weights for MultimodalPCTLoss
_C.MODEL.L1_WEIGHT = 1.0
_C.MODEL.PROJ_WEIGHT = 0.1  # Weight for 2D Projection consistency
_C.MODEL.GRAD_WEIGHT = 0.0  # Weight for spatial gradient consistency
_C.MODEL.SSIM_WEIGHT = 0.0  # Weight for structural similarity index
_C.MODEL.SSIM_WINDOW_SIZE = 11  # Tamaño de ventana para cálculo de SSIM
_C.MODEL.MAGNITUDE_WEIGHT_K = 3.0  # Peso adicional por magnitud de intensidad real

# Architecture parameters
_C.MODEL.NUM_FILTERS = 32     # Base filters for encoders
_C.MODEL.TOPO_CHANNELS = 1    # 1 channel since topogram is [1, 512, 531]
_C.MODEL.COORD_DIM = 3        # x, y, z normalized coordinates
_C.MODEL.INTERPOLATION_MODE = "trilinear" # Interpolación de red

# Number of input channels for 3D modalities (PET, MRI)
_C.MODEL.NUM_CHANNELS = 1

_C.MODEL.KERNEL_D = 3
_C.MODEL.KERNEL_H = 3
_C.MODEL.KERNEL_W = 3

_C.MODEL.STRIDE_CONV = 1
_C.MODEL.STRIDE_POOL = 2
_C.MODEL.NUM_GROUPS = 8

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 4       # Aumentado para mayor estabilidad con parches
_C.TRAIN.LOG_INTERVAL = 20

# Resume & Fine-tuning options
_C.TRAIN.RESUME = False
_C.TRAIN.RESUME_EXPR_NUM = ""
_C.TRAIN.FINE_TUNE = False
_C.TRAIN.WARM_START_CKPT = ""   # Ruta a un .pkl concreto para fine-tuning (ej. Best_training_state.pkl); vacío = último Epoch_*
_C.TRAIN.CHECKPOINT_PERIOD = 10 # Frecuencia de guardado de respaldos

_C.TRAIN.NUM_EPOCHS = 200     # Ajustado para entrenamiento multimodal complejo

# Patch-based training settings
_C.TRAIN.PATCH_TRAINING = True
_C.TRAIN.PATCH_SIZE = [64, 128, 128]    # Tamaño optimizado para GPU memory (D, H, W)
_C.TRAIN.PATCH_STRIDE = [32, 64, 64]    # Solapamiento para consistencia
_C.TRAIN.BATCHES_PER_EPOCH = 100        # Número de parches aleatorios por época
_C.TRAIN.VAL_PATCHES = 200          # Número de parches para validación
# Mixed Precision Training
_C.TRAIN.AMP = True           # Automatic Mixed Precision para ahorrar memoria
_C.TRAIN.GRAD_SSIM_WARMUP_EPOCHS = 30
_C.TRAIN.VAL_EVERY_N_EPOCHS = 1
_C.TRAIN.PATIENCE = 40        # Early stopping: épocas sin mejora en val MAE
# Media móvil exponencial (EMA) de los pesos; 0 = desactivada. Si está activa, la
# validación y los checkpoints usan los pesos EMA (0.999 ≈ 1000 iteraciones de memoria).
_C.TRAIN.EMA_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #
_C.DATA = CN()

_C.DATA.DATA_PATH = "./data/challenge_dataset"

# Input sizes for full volumes (Depth, Height, Width)
_C.DATA.SIZES = [346, 391, 391]
_C.DATA.TARGET_SIZE = [346, 391, 391]
_C.DATA.VOXEL_SIZES = [2.0644, 2.0644, 2.0644]
_C.DATA.RESIZE_MODE = "trilinear"
_C.DATA.NOISE_STD = 0.0

# Normalization values (HU units are typically -1000 to 3000)
_C.DATA.CT_MIN = -1000
_C.DATA.CT_MAX = 3000

# En config/defaults.py, dentro de la sección _C.DATA
_C.DATA.APPLY_AUGMENTATION = False  # Por defecto activo

# Probabilidad de poner a cero el input de MRI durante entrenamiento/validación,
# para que el modelo aprenda a sintetizar CT usando solo PET cuando no hay MRI
# disponible (caso UMC Utrecht pediátrico). 0.0 = comportamiento original (MRI
# siempre presente, sin cambios de comportamiento para configs existentes).
_C.DATA.MRI_DROPOUT_PROB = 0.0
_C.DATA.MRI_DROPOUT_PROB_VAL = 0.0

# Igual que el MRI dropout pero para el topograma: la cohorte de Utrecht no
# tiene scout, así que el modelo debe aprender a funcionar sin él.
_C.DATA.TOPO_DROPOUT_PROB = 0.0
_C.DATA.TOPO_DROPOUT_PROB_VAL = 0.0
_C.DATA.TOPO_SHAPE = [512, 531]   # Forma del topograma tras squeeze; se usa para rellenar cuando falta

# "grid" (v12): coords = inicio del parche / tamaño de la malla.
# "body": centro del parche relativo al bbox del cuerpo (calculado desde NAC-PET),
# para que las coordenadas signifiquen lo mismo en adultos y niños.
_C.DATA.COORDS_MODE = "grid"

# Cohorte extra SOLO para entrenamiento (B3: BigMAC mientras los folds rotan sobre
# Utrecht). run_trainning.py la rellena desde --extra_train_dir. Cada parche de
# train sale de la cohorte extra con probabilidad EXTRA_TRAIN_FRAC.
_C.DATA.EXTRA_TRAIN_FILES = []
_C.DATA.EXTRA_TRAIN_FRAC = 0.3

# Muestreo dirigido de la cohorte principal (B3 v21, solo train):
#   THORAX_FRAC  fracción de parches con el centro en la losa pulmonar (clase 3 de la máscara)
#   YOUNG_FRAC   fracción de parches de sujetos con edad < YOUNG_AGE (edad de PARTICIPANTS_CSV)
_C.DATA.THORAX_FRAC = 0.0
_C.DATA.YOUNG_FRAC = 0.0
_C.DATA.YOUNG_AGE = 13
_C.DATA.PARTICIPANTS_CSV = ""
# Losa pulmonar solo con el tramo continuo de aire alrededor de su máximo (sin gas intestinal).
# False = definición original (v21, v22); ver utils/body_coords.lung_z_range.
_C.DATA.LUNG_CONTIGUOUS = False

# Escala "pediátrica" (B3 v22, solo train): con probabilidad EXTRA_SHRINK_PROB, un parche
# de la cohorte extra (BigMAC, adultos) se encoge con un factor isotrópico en
# EXTRA_SHRINK_SCALE para parecerse a la anatomía de un niño.
_C.DATA.EXTRA_SHRINK_PROB = 0.0
_C.DATA.EXTRA_SHRINK_SCALE = [0.65, 0.9]

# Eje del flip de la augmentation (ejes del volumen: 0 = izquierda-derecha, 1 = AP,
# 2 = cabeza-pies). 2 = comportamiento original (v18-v22, volteaba cabeza-pies).
_C.DATA.FLIP_AXIS = 2

# Augmentation de dominio del NAC-PET (B3 v23, solo train, cohortes extra y pediátrica):
# imita el PET de Utrecht sobre el de BigMAC (medido con el perfil frente a la distancia a la
# piel en 25 exámenes de Utrecht y 10 de BigMAC: centro ~0.67x relativo a los 3 cm externos y
# cola del halo fuera del cuerpo +50 %; con estos rangos el aug reproduce esos cambios).
#   ganancia dentro = 1 - DARK * clip((d_piel - RIM_MM) / DEPTH_MM, 0, 1)   (los RIM_MM externos
#                     no cambian: en las dos cohortes miden lo mismo)
#   halo fuera      = HALO * p95(PET en el cuerpo) * exp(-d_piel / HALO_MM)
# Cada parámetro se sortea uniforme en su rango.
_C.DATA.PET_DOMAIN_PROB = 0.0
_C.DATA.PET_DOMAIN_DARK = [0.3, 0.5]
_C.DATA.PET_DOMAIN_DEPTH_MM = [30.0, 60.0]
_C.DATA.PET_DOMAIN_RIM_MM = [20.0, 30.0]
_C.DATA.PET_DOMAIN_HALO = [0.02, 0.05]
_C.DATA.PET_DOMAIN_HALO_MM = [60.0, 100.0]

# Cohorte pediátrica deformada (B3 v24: adultos de BigMAC deformados a niños de Utrecht,
# utils/deform_batch.py), solo train. run_trainning.py la rellena desde --ped_train_dir
# SIN el fold_k.h5 del fold de validación (sus parejas copian la forma de esos niños).
# Cada parche sale de esta cohorte con probabilidad PED_TRAIN_FRAC.
_C.DATA.PED_TRAIN_FILES = []
_C.DATA.PED_TRAIN_FRAC = 0.0
# v25: solo las parejas cuyo niño tiene menos de PED_MAX_AGE años (atributo child_age) y
# probabilidad propia de la augmentation de dominio en esta cohorte (< 0 = PET_DOMAIN_PROB).
_C.DATA.PED_MAX_AGE = 99.0
_C.DATA.PET_DOMAIN_PROB_PED = -1.0

# ---------------------------------------------------------------------------- #
# DataLoader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()

_C.OPTIMIZER.OPTIMIZING_METHOD = "adamw"
_C.OPTIMIZER.BASE_LR = 1e-4

# Learning rate scheduler
_C.OPTIMIZER.LR_SCHEDULER = "cosineWarmRestarts"
_C.OPTIMIZER.ETA_MIN = 1e-6
_C.OPTIMIZER.T_ZERO = 20
_C.OPTIMIZER.T_MULT = 2
_C.OPTIMIZER.POLY_POWER = 0.9
_C.OPTIMIZER.MOMENTUM = 0.99
_C.OPTIMIZER.NESTEROV = True

# Regularization
_C.OPTIMIZER.WEIGHT_DECAY = 1e-2

# Fine-tuning por grupos (B3). Parámetros cuyo nombre empieza por un prefijo de
# FREEZE_PREFIXES no se entrenan; los de LR_MULT_PREFIXES[i] usan
# BASE_LR * LR_MULT_VALUES[i]; el resto, BASE_LR.
_C.OPTIMIZER.FREEZE_PREFIXES = []
_C.OPTIMIZER.LR_MULT_PREFIXES = []
_C.OPTIMIZER.LR_MULT_VALUES = []

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.NUM_GPUS = 1
_C.LOG_DIR = "./experiments"
_C.EXPR_NUM = "PCT_v1"
_C.SUMMARY_PATH = ""          # Ruta dinámica para TensorBoard
_C.RNG_SEED = 42

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()