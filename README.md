# pseudoCT_i3M_bic-mac-challenge

Multimodal pseudo-CT synthesis for the **BIC-MAC Challenge**. A 3D U-Net (`MultimodalPCTUNet`) predicts a CT volume in HU from:

- **NAC-PET**: required
- **MRI Dixon**, in-phase and out-phase: optional at inference if the model was trained with MRI dropout
- **Topogram (scout)**: optional at inference if the model was trained with topogram dropout
- normalised patch coordinates

Training uses patches and 5-fold cross-validation. Inference uses sliding-window reconstruction (Hann blending) and averages the 5 fold models.

---

## 1. Requirements

- Linux or WSL2 with an NVIDIA GPU and CUDA. The code uses AMP and was developed with PyTorch 2.4 and CUDA 12.1.
- Python ≥ 3.10. The cached bytecode in the repo is from Python 3.12.
- Enough RAM to hold full volumes: the BIC-MAC volumes are about 512 × 512 × 531.

### Install

```bash
python -m venv .venv
source .venv/bin/activate

# Install the PyTorch build that matches your CUDA version first, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt` covers training and inference on BIC-MAC (`torch`, `torchio`, `nibabel`, `h5py`, `yacs`, `scikit-learn`, `scipy`, `tensorboard`, …). The UMC Utrecht pediatric pipeline needs two more packages:

```bash
pip install SimpleITK pydicom
```

### Python path (important)

The model code lives in `utils/models/`, but it is imported as the top-level package `models` (`from models.networks import build_model`). Run every command **from the repository root** with both the root and `utils/` on the path:

```bash
export PYTHONPATH="$(pwd):$(pwd)/utils"
```

---

## 2. Dataset preparation

### 2.1 Expected raw layout (BIC-MAC)

Put one folder per subject (`sub-*`) in a root folder (e.g. `train/`):

```
train/
├── sub-001/
│   ├── features/
│   │   ├── nacpet.nii.gz
│   │   ├── mri_combined_in_phase.nii.gz
│   │   ├── mri_combined_out_phase.nii.gz
│   │   └── topogram.nii.gz
│   └── ct-label/
│       ├── ct.nii.gz          # target CT (HU)
│       ├── organ_seg.nii.gz   # organ segmentation (TotalSegmentator-style labels)
│       └── body_seg.nii.gz    # body mask
├── sub-002/
│   └── ...
```

Only folders whose name starts with `sub-` are used. All 3D volumes of a subject must be on the same grid, which is the CT grid in BIC-MAC. Volumes are reoriented to RAS with `nib.as_closest_canonical` when they are loaded.

### 2.2 Generate the HDF5 folds

```bash
python -m data.generate_hdf5 \
    --dataset_path /path/to/bic-mac-data/train \
    --output_dir   /path/to/hdf5_bigmac \
    --folds 5 --seed 42 --workers 4
```

This step:

- Shuffles the subjects and splits them with `KFold` into `fold_0.h5 … fold_4.h5`.
- Stores `pet`, `mri` (2 channels, in/out phase), `ct`, `topo` and a weighting `mask` for each subject, with gzip compression.
- Builds the `mask` from `body_seg` and `organ_seg` for the loss weighting:

| Class | Tissue |
|---|---|
| 0 | outside the body |
| 2 | body |
| 3 | internal air (lungs) |
| 4 | muscles, vessels, heart and critical organs |
| 5 | standard bones (ribs, sternum, humerus, scapula, …) |
| 6 | dense bone (vertebrae and sacrum) |

## 3. Training

Training is driven by `run_trainning.py`. It runs K-fold cross-validation over every `fold_*.h5` file in `--hdf5_dir`. For fold *k*, `fold_k.h5` is the validation set and the other folds are the training set.

1. Choose or copy a config in `config/`. The shipped weights were trained with `config/mutimodal_docker.yaml` (`EXPR_NUM: Multimodal_pCT_v12_torso`).
2. Set `LOG_DIR` in the YAML, or override it with `--log_dir`.

```bash
python run_trainning.py \
    --config_path config/mutimodal_docker.yaml \
    --hdf5_dir    /path/to/hdf5_bigmac \
    --log_dir     ./experiments \
    --gpus 0
```

### Useful options

| Flag | Description |
|---|---|
| `--gpus 0,1` | Sets `CUDA_VISIBLE_DEVICES` and `NUM_GPUS` |
| `--omitted 1 3` | Skip these fold indices |
| `--resume 0 2` | Resume these folds from their last `Epoch_*` checkpoint, including optimizer, scheduler and epoch |
| `--warm_start_expr NAME` | Fine-tune: load only the model weights of `NAME_Fold_k` |
| `--warm_start_dir checkpoints` | With `--warm_start_expr`, use `Best_training_state.pkl` instead of the last epoch |
| `--extra_train_dir DIR` | Add the `fold_*.h5` files from another cohort to training only. Their share of patches is set by `DATA.EXTRA_TRAIN_FRAC` |
| `--ped_train_dir DIR` | Add the deformed pediatric cohort to training only. `fold_k.h5` is excluded for fold *k* |

Example: fine-tune on Utrecht from the shipped BIC-MAC weights, with BIC-MAC as an extra training cohort.

```bash
python run_trainning.py \
    --config_path     config/mutimodal_v26_utrecht_ft.yaml \
    --hdf5_dir        /path/to/hdf5_utrecht \
    --extra_train_dir /path/to/hdf5_bigmac \
    --warm_start_expr Multimodal_pCT_v12_torso \
    --warm_start_dir  checkpoints \
    --gpus 0
```

### Outputs

```
<LOG_DIR>/
├── checkpoints/<EXPR_NUM>_Fold_<k>/
│   ├── Best_training_state.pkl   # best validation MAE
│   └── Epoch_*.pkl               # every TRAIN.CHECKPOINT_PERIOD epochs
├── summary/<EXPR_NUM>_Fold_<k>/  # TensorBoard
└── comparisons/
```

Monitor training with:

```bash
tensorboard --logdir <LOG_DIR>/summary
```

Main hyperparameters are in the YAML. `config/defaults.py` lists every key with comments. Key settings include `TRAIN.PATCH_SIZE`, `TRAIN.PATCH_STRIDE`, `TRAIN.BATCH_SIZE`, `TRAIN.NUM_EPOCHS`, `TRAIN.PATIENCE`, the loss weights (`MODEL.*_WEIGHT`), and `DATA.MRI_DROPOUT_PROB` / `DATA.TOPO_DROPOUT_PROB` for PET-only models.

---

## 4. Inference

### 4.1 Local (batch of patients)

The input is a root folder with one sub-folder per patient. Each patient needs at least `features/nacpet.nii.gz`:

```
input_dir/
├── sub-101/features/{nacpet, mri_combined_in_phase, mri_combined_out_phase, topogram}.nii.gz
└── sub-102/features/...
```

```bash
python inference.py \
    --config_path     config/mutimodal_docker.yaml \
    --checkpoints_dir checkpoints \
    --version         v12_torso \
    --input_dir       /path/to/val \
    --output_dir      outputs/inference_v12 \
    --device cuda
```

- Every `*<version>*/Best_training_state.pkl` under `--checkpoints_dir` is loaded, and their predictions are averaged. The repo ships the 5 folds of `Multimodal_pCT_v12_torso` in `checkpoints/`.
- Results are written to `<output_dir>/<case>/ct.nii`.
- If MRI or the topogram is missing, zeros are used in its place. Only do this with a checkpoint trained with MRI or topogram dropout.
- `--clamp_outside_body` sets voxels outside the dilated NAC-PET body mask to −1000 HU. This is recommended for Utrecht.
- Use the same YAML the checkpoints were trained with, because the architecture, patch size and `DATA.COORDS_MODE` must match.

### 4.2 Docker (challenge submission)

`inferencia_docker.py` uses fixed paths. It reads `/data/features/` (read-only) and writes `/data/output/ct.nii.gz`. The config and weights are built into the image, so it runs with `--network none`.

```bash
docker build -t bicmac-inference .

docker run --rm --gpus all --network none \
    -v /path/to/sub-004/features:/data/features:ro \
    -v /path/to/output/sub-004:/data/output \
    bicmac-inference
```

You can also edit the volumes in `docker-compose.yml` and run `docker compose up`.

Optional environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `CONFIG_PATH` | `/app/config/mutimodal_docker.yaml` | Config file |
| `CHECKPOINTS_DIR` | `/app/checkpoints` | Folder with the fold checkpoints |
| `MODEL_VERSION` | all | Pattern to select checkpoints |
| `INFERENCE_BATCH_SIZE` | `8` | Patches per batch |
| `INFERENCE_PATCH_STRIDE` | from config | Patch overlap, e.g. `20,40,40` |
| `MAX_ENSEMBLE_MODELS` | all | Use only the first N folds |

The Docker script needs all four input modalities.

---

## License

See [LICENSE](LICENSE).
