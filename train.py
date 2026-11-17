"""
train.py

Main training orchestrator for Multimodal pseudo-CT Synthesis.
Integrates AMP (Mixed Precision), MAE tracking, and NIfTI comparison logging.

Owner: Edwing Ulin
Version: v1.2.0
"""

import os
from pathlib import Path
import time
from typing import Union

import numpy as np
import torch
from torch.cuda import amp
import yacs.config
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import loader
from models.losses import get_loss_func
from models.networks import build_model
from models.optimizer import get_optimizer
from utils.logger import logger_gen
from utils import checkpoint as cp
from utils.lr_scheduler import get_lr_scheduler
from utils.meters import Meter
from utils.misc import update_num_steps

logger = logger_gen(__name__)

torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(self, cfg: yacs.config.CfgNode):
        """
        Initializes the training environment, model, loss, and devices.
        """
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        os.makedirs(cfg.LOG_DIR, exist_ok=True)

        os.makedirs(cfg.SUMMARY_PATH, exist_ok=True)

        os.makedirs(os.path.join(cfg.LOG_DIR, "comparisons"), exist_ok=True)


        self.checkpoint_dir = cp.create_checkpoint_dir(cfg.LOG_DIR, cfg.EXPR_NUM)
        logger.info(f"Training initialized with config:\n{cfg.dump()}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = build_model(cfg).to(self.device)
        self.loss_func = get_loss_func(cfg)
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.TRAIN.AMP)
        
        self.metric_name = "MAE_HU"
        self.ema = None   # se crea en run(), después de cargar los pesos de partida

    def train(
        self,
        train_loader: loader.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: any,
        train_meter: Meter,
        epoch: int,
    ) -> None:
        """
        Executes one training epoch.
        """
        self.model.train()
        logger.info(f"--- Epoch {epoch} Training Started ---")
        epoch_start = time.time()

        if epoch == 1:
            torch.cuda.reset_peak_memory_stats()
        
        last_pred, last_target = None, None

        for curr_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            pet = batch["pet"].to(self.device)
            mri = batch["mri"].to(self.device)
            topo = batch["topo"].to(self.device)
            mask = batch["batch_mask"].to(self.device)
            coords = batch["coords"].to(self.device)
            target = batch["target"].to(self.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.cfg.TRAIN.AMP):
                pred = self.model(pet, mri, topo, coords)
                loss_total = self.loss_func(pred, target, mask=mask)
            
            if torch.isnan(loss_total):
                logger.warning(f"NaN detected in loss at epoch {epoch}, iter {curr_iter}. Skipping this batch.")
                continue

            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.scaler.step(optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update_parameters(self.model)

            if scheduler is not None:
                scheduler.step()

            train_meter.update_stats(pred, target, loss_total, mask=mask)
            train_meter.log_iter(curr_iter, epoch)
            
            lr = optimizer.param_groups[0]['lr']
            train_meter.write_summary(
                loss_total,
                [lr],
                loss_components=getattr(self.loss_func, "last_components", None),
                grad_norm=grad_norm.item()
                )
            
            if curr_iter == 20:
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"Peak VRAM tras 20 iters: {peak_gb:.2f} GB de 16 GB disponibles")

            last_pred = pred
            last_target = target

        train_meter.log_epoch(epoch, batch=batch, pred=last_pred, target=last_target)
        logger.info(f"Training epoch {epoch} finished in {time.time() - epoch_start:.2f}s")

    @torch.no_grad()
    def eval(
        self, 
        val_loader: loader.DataLoader, 
        val_meter: Meter, 
        epoch: int
    ) -> float:
        """
        Evaluates the model and calculates the Mean Absolute Error (MAE) in HU.
        """
        logger.info(f"--- Evaluating model at epoch {epoch} ---")
        # Con EMA se valida (y se guarda) el modelo promediado.
        model = self.ema.module if self.ema is not None else self.model
        model.eval()
        val_start = time.time()
        
        last_pred, last_target = None, None
        # MAE por parche separada según si el MRI estaba presente o anulado,
        # para vigilar que el fine-tuning sin MRI no degrade el caso con MRI.
        mae_by_regime = {"con_mri": [], "sin_mri": []}

        for curr_iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            pet = batch["pet"].to(self.device)
            mri = batch["mri"].to(self.device)
            topo = batch["topo"].to(self.device)
            mask = batch["batch_mask"].to(self.device)
            coords = batch["coords"].to(self.device)
            target = batch["target"].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.cfg.TRAIN.AMP):
                pred = model(pet, mri, topo, coords)
                loss_total = self.loss_func(pred, target, mask=mask)

            with torch.no_grad():
                pred_hu = pred * (self.cfg.DATA.CT_MAX - self.cfg.DATA.CT_MIN) + self.cfg.DATA.CT_MIN
                target_hu = target * (self.cfg.DATA.CT_MAX - self.cfg.DATA.CT_MIN) + self.cfg.DATA.CT_MIN
                logger.info(f"Target HU range: [{target_hu.min().item():.1f}, {target_hu.max().item():.1f}]")
                logger.info(f"Pred HU range:   [{pred_hu.min().item():.1f}, {pred_hu.max().item():.1f}]")

                # float32: con AMP pred viene en fp16 y la suma de ~1M vóxeles desborda.
                valid = (torch.round(mask) > 0).float()
                abs_err = (pred_hu.float() - target_hu.float()).abs() * valid
                per_sample = abs_err.flatten(1).sum(1) / (valid.flatten(1).sum(1) + 1e-8)
                dropped = batch["mri_dropped"].to(per_sample.device).bool()
                mae_by_regime["sin_mri"] += per_sample[dropped].tolist()
                mae_by_regime["con_mri"] += per_sample[~dropped].tolist()

            val_meter.update_stats(pred, target, loss_total, mask=mask)
            val_meter.log_iter(curr_iter, epoch)
            val_meter.write_summary(
                loss_total,
                lr=[0.0],
                loss_components=getattr(self.loss_func, "last_components", None)
            )

            if curr_iter == 0:
                viz_val_batch = batch
                viz_val_pred = pred
                viz_val_target = target

        avg_mae = np.mean(val_meter.epoch_maes)

        for regime, values in mae_by_regime.items():
            if values:
                regime_mae = float(np.mean(values))
                logger.info(f"Val MAE {regime}: {regime_mae:.2f} HU (n={len(values)})")
                val_meter.writer.add_scalar(f"Val/MAE_{regime}", regime_mae, epoch)

        val_meter.log_epoch(epoch, batch=viz_val_batch, pred=viz_val_pred, target=viz_val_target)
        
        logger.info(f"Validation finished in {time.time() - val_start:.2f}s. Avg MAE: {avg_mae:.4f} HU")
        return avg_mae

    def run(self, train_folds: list, val_fold: list):
        """
        Main loop managing data loading, training, evaluation, and checkpoints.
        """

        train_loader = loader.get_dataloader(self.cfg, "train", fold_paths=train_folds)
        val_loader = loader.get_dataloader(self.cfg, "val", fold_paths=val_fold)

        update_num_steps(train_loader, self.cfg)
        optimizer = get_optimizer(self.model, self.cfg)
        scheduler = get_lr_scheduler(optimizer, self.cfg)

        start_epoch = 0
        best_mae = float('inf')
        patience = getattr(self.cfg.TRAIN, "PATIENCE", 40)
        epochs_without_improvement = 0
        warmup_epochs = self.cfg.TRAIN.GRAD_SSIM_WARMUP_EPOCHS if hasattr(self.cfg.TRAIN, "GRAD_SSIM_WARMUP_EPOCHS") else 30
        val_every = getattr(self.cfg.TRAIN, "VAL_EVERY_N_EPOCHS", 1)

        # Valores OBJETIVO fijos, leídos UNA sola vez del config original —
        # nunca se reasignan sobre sí mismos, evita el bug de ramp compuesto.
        target_grad_weight = self.cfg.MODEL.GRAD_WEIGHT
        target_ssim_weight = self.cfg.MODEL.SSIM_WEIGHT

        if self.cfg.TRAIN.WARM_START_CKPT:
            checkpoint_paths = [self.cfg.TRAIN.WARM_START_CKPT]
        else:
            checkpoint_paths = cp.get_checkpoint_path(self.cfg.LOG_DIR, self.cfg.TRAIN.RESUME_EXPR_NUM)
        if self.cfg.TRAIN.RESUME and checkpoint_paths:
            checkpoint_path = checkpoint_paths.pop()
            start_epoch, best_mae = cp.load_from_checkpoint(
                checkpoint_path, self.model, optimizer, scheduler,
                fine_tune=self.cfg.TRAIN.FINE_TUNE
            )
            if self.cfg.TRAIN.FINE_TUNE:
                # Fine-tuning arranca una fase nueva: solo nos interesan los pesos
                # del checkpoint, no su contador de épocas ni su mejor métrica
                # (que no es comparable bajo el nuevo régimen de MRI dropout).
                logger.info(
                    f"Fine-tuning desde pesos preentrenados (época original {start_epoch}). "
                    f"Reiniciando contador de épocas y mejor métrica."
                )
                start_epoch, best_mae = 0, float('inf')
            logger.info(f"Resuming from epoch {start_epoch} with Best MAE: {best_mae:.4f}")

        ema_decay = float(getattr(self.cfg.TRAIN, "EMA_DECAY", 0.0))
        if ema_decay > 0:
            self.ema = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
            logger.info(f"EMA de los pesos activada (decay {ema_decay}).")

        writer = SummaryWriter(os.path.join(self.cfg.LOG_DIR, "summary", self.cfg.EXPR_NUM))
        train_meter = Meter(
            self.cfg, mode="train", global_step=start_epoch * len(train_loader),
            total_iter=len(train_loader), total_epoch=self.cfg.TRAIN.NUM_EPOCHS, 
            device=self.device, writer=writer
        )
        val_meter = Meter(
            self.cfg, mode="val", global_step=start_epoch * len(val_loader),
            total_iter=len(val_loader), total_epoch=self.cfg.TRAIN.NUM_EPOCHS, 
            device=self.device, writer=writer
        )

        for epoch in range(start_epoch, self.cfg.TRAIN.NUM_EPOCHS):
            epochs_since_start = epoch - start_epoch
            ramp = min(1.0, (epochs_since_start + 1) / warmup_epochs)
            self.loss_func.grad_weight = target_grad_weight * ramp
            self.loss_func.ssim_weight = target_ssim_weight * ramp

            self.train(train_loader, optimizer, scheduler, train_meter, epoch + 1)

            should_validate = ((epoch + 1) % val_every == 0) or ((epoch + 1) == self.cfg.TRAIN.NUM_EPOCHS)

            if not should_validate:
                continue

            current_mae = self.eval(val_loader, val_meter, epoch + 1)

            is_best = current_mae < best_mae
            if is_best:
                best_mae = current_mae
                epochs_without_improvement = 0
                logger.info(f"New Best MAE reached: {best_mae:.4f}. Saving best model...")
            else:
                epochs_without_improvement += val_every
                logger.info(f"No improvement in MAE. Current: {current_mae:.4f}, Best: {best_mae:.4f}. Patience: {epochs_without_improvement}/{patience}")

            if is_best or (epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                if self.ema is not None:
                    model_to_save = self.ema.module
                else:
                    model_to_save = self.model.module if self.cfg.NUM_GPUS > 1 else self.model

                cp.save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    epoch=epoch + 1,
                    best_metric=best_mae,
                    num_gpus=self.cfg.NUM_GPUS,
                    cfg=self.cfg,
                    model=model_to_save,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best=is_best
                )

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        writer.add_hparams(
            {
                "l1_weight": self.cfg.MODEL.L1_WEIGHT,
                "grad_weight": target_grad_weight,
                "ssim_weight": target_ssim_weight,
                "base_lr": self.cfg.OPTIMIZER.BASE_LR,
                "model_name": self.cfg.MODEL.MODEL_NAME,
            },
            {"hparam/best_mae": best_mae},
        )
        writer.close()
        logger.info("Training process completed.")