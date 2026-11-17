"""
run_training.py

Orquestador de 5-Fold Cross-Validation para el proyecto Multimodal pCT.
Este script entrena 5 modelos independientes rotando los archivos HDF5,
asegurando que cada fold tenga sus propios logs y checkpoints.

Owner: Edwing Ulin
Version: v1.2.0
"""

import os
import argparse
from pathlib import Path

from config.defaults import get_cfg_defaults
from train import Trainer
from utils.logger import logger_gen

logger = logger_gen('cross_val_orchestrator')

def run_cross_validation(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus_available = len(args.gpus.split(','))
    logger.info(f"GPUs configuradas: {args.gpus} (Disponibles: {num_gpus_available})")
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)

    cfg.NUM_GPUS = num_gpus_available
    
    if args.log_dir: 
        cfg.LOG_DIR = args.log_dir
    
    h5_folder = Path(args.hdf5_dir)
    all_folds = sorted(list(h5_folder.glob("fold_*.h5")))

    if args.extra_train_dir:
        # Cohorte extra solo para entrenar (B3: BigMAC); los folds rotan sobre --hdf5_dir.
        cfg.DATA.EXTRA_TRAIN_FILES = [str(p) for p in sorted(Path(args.extra_train_dir).glob("fold_*.h5"))]
        if not cfg.DATA.EXTRA_TRAIN_FILES:
            logger.error(f"No se encontraron fold_*.h5 en --extra_train_dir {args.extra_train_dir}")
            return
        logger.info(f"Cohorte extra (solo train, p={cfg.DATA.EXTRA_TRAIN_FRAC}): "
                    f"{[Path(f).name for f in cfg.DATA.EXTRA_TRAIN_FILES]}")

    ped_files = []
    if args.ped_train_dir:
        # Cohorte pediátrica deformada: fold_k.h5 tiene las parejas deformadas hacia los niños
        # del fold k de Utrecht, así que se excluye al entrenar el modelo del fold k.
        ped_files = sorted(Path(args.ped_train_dir).glob("fold_*.h5"))
        if not ped_files:
            logger.error(f"No se encontraron fold_*.h5 en --ped_train_dir {args.ped_train_dir}")
            return
        logger.info(f"Cohorte pediátrica deformada (solo train, p={cfg.DATA.PED_TRAIN_FRAC}): "
                    f"{[p.name for p in ped_files]}")

    if len(all_folds) == 0:
        logger.error(f"No se encontraron archivos fold_*.h5 en {args.hdf5_dir}")
        return

    logger.info(f"Iniciando Cross-Validation con {len(all_folds)} folds encontrados.")

    for fold_idx in range(len(all_folds)):
        if fold_idx in args.omitted:
            logger.info(f"🔹 Omitiendo Fold {fold_idx} por configuración.")
            continue
        logger.info("\n" + "="*60)
        logger.info(f"INICIANDO ENTRENAMIENTO DEL FOLD {fold_idx} ")
        logger.info("="*60)

        val_fold = [str(all_folds[fold_idx])]
        train_folds = [str(f) for i, f in enumerate(all_folds) if i != fold_idx]

        logger.info(f"Archivos Train: {[Path(f).name for f in train_folds]}")
        logger.info(f"Archivo Val:  {[Path(f).name for f in val_fold]}")

        fold_cfg = cfg.clone()
        if ped_files:
            fold_cfg.DATA.PED_TRAIN_FILES = [str(p) for p in ped_files
                                             if p.name != Path(all_folds[fold_idx]).name]
            logger.info(f"Pediátrica en train: {[Path(f).name for f in fold_cfg.DATA.PED_TRAIN_FILES]} "
                        f"(excluido {Path(all_folds[fold_idx]).name})")
        
        base_expr = fold_cfg.EXPR_NUM
        fold_cfg.EXPR_NUM = f"{base_expr}_Fold_{fold_idx}"

        if fold_idx in args.resume:
            logger.info(f"🔄 Resumiendo Fold {fold_idx} desde el último checkpoint disponible.")
            fold_cfg.TRAIN.RESUME = True
            fold_cfg.TRAIN.RESUME_EXPR_NUM = fold_cfg.EXPR_NUM
            # Reanudar el MISMO experimento: optimizer, scheduler, época y mejor MAE
            # vienen del último Epoch_*. Con FINE_TUNE (de los YAML de fine-tuning)
            # se reiniciaban a 0 / inf y el primer val sobrescribía Best_training_state.
            fold_cfg.TRAIN.FINE_TUNE = False
            fold_cfg.TRAIN.WARM_START_CKPT = ""
        elif args.warm_start_expr:
            warm_start_name = f"{args.warm_start_expr}_Fold_{fold_idx}"
            logger.info(
                f"🌱 Fine-tuning Fold {fold_idx}: cargando SOLO los pesos de "
                f"'{warm_start_name}' (sin optimizer/scheduler, época reiniciada)."
            )
            fold_cfg.TRAIN.RESUME = True
            fold_cfg.TRAIN.RESUME_EXPR_NUM = warm_start_name
            fold_cfg.TRAIN.FINE_TUNE = True
            if args.warm_start_dir:
                fold_cfg.TRAIN.WARM_START_CKPT = os.path.join(
                    args.warm_start_dir, warm_start_name, "Best_training_state.pkl"
                )
                if not os.path.isfile(fold_cfg.TRAIN.WARM_START_CKPT):
                    raise FileNotFoundError(fold_cfg.TRAIN.WARM_START_CKPT)
                logger.info(f"Warm start desde el MEJOR checkpoint: {fold_cfg.TRAIN.WARM_START_CKPT}")
        else:
            fold_cfg.TRAIN.RESUME = False
            fold_cfg.TRAIN.RESUME_EXPR_NUM = None
        
        fold_cfg.SUMMARY_PATH = os.path.join(fold_cfg.LOG_DIR, "summary", fold_cfg.EXPR_NUM)

        try:
            trainer_inst = Trainer(cfg=fold_cfg)
            trainer_inst.run(train_folds=train_folds, val_fold=val_fold)
            logger.info(f"Fold {fold_idx} completado exitosamente.")
            
        except Exception as e:
            logger.error(f"❌ Error crítico en el Fold {fold_idx}: {e}")
            logger.exception("Detalles del error:")
            continue 

    logger.info("\n🎉 5-Fold Cross-Validation completado. Todos los modelos han sido guardados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation Orchestrator")
    
    parser.add_argument("--config_path", type=str, required=True, 
                        help="Ruta al archivo .yaml de configuración (ej. config/MultimodalPCT.yaml)")
    
    parser.add_argument("--hdf5_dir", type=str, required=True, 
                        help="Carpeta que contiene fold_0.h5, fold_1.h5, etc.")
    
    parser.add_argument("--log_dir", type=str, default=None, 
                        help="Opcional: sobrescribir la ruta base de logs y experimentos")
    
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPUs a utilizar (ej. 0,1,2,3)")
    
    parser.add_argument("--omitted", type=int, nargs='*', default=[],
                        help="Índices de folds a omitir (ej. --omitted 1 3)")
    
    parser.add_argument("--resume", type=int, nargs='*', default=[],
                        help="Índices de folds a resumir (ej. --resume 0 2)")

    parser.add_argument("--warm_start_expr", type=str, default=None,
                        help="Nombre base de un experimento previo (ej. "
                             "'Multimodal_pCT_v12_torso') del cual cargar SOLO los "
                             "pesos del modelo por fold para fine-tuning (activa "
                             "TRAIN.FINE_TUNE, no carga optimizer/scheduler ni "
                             "contador de épocas). Se combina con --resume en "
                             "folds específicos si quieres priorizar resumir esos.")

    parser.add_argument("--extra_train_dir", type=str, default=None,
                        help="Carpeta con fold_*.h5 de otra cohorte que se añade SOLO al "
                             "entrenamiento de cada fold (ej. BigMAC en el fine-tuning de "
                             "Utrecht); proporción de parches: DATA.EXTRA_TRAIN_FRAC.")

    parser.add_argument("--ped_train_dir", type=str, default=None,
                        help="Carpeta con fold_*.h5 de la cohorte pediátrica deformada "
                             "(utils/deform_batch.py). Se añade SOLO al entrenamiento y sin el "
                             "fold_k.h5 del fold de validación; proporción: DATA.PED_TRAIN_FRAC.")

    parser.add_argument("--warm_start_dir", type=str, default=None,
                        help="Carpeta con <warm_start_expr>_Fold_k/Best_training_state.pkl "
                             "(ej. 'checkpoints'). Si se da, el warm start usa el MEJOR "
                             "checkpoint de cada fold en vez del último Epoch_* de LOG_DIR.")

    args = parser.parse_args()
    
    run_cross_validation(args)