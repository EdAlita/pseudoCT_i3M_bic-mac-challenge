"""
meters.py

Módulo para la gestión de logs y resúmenes de TensorBoard específicos para
la síntesis de pseudo-CT multimodal.

Owner: Edwing Ulin
Version: v1.1.0
"""

import os
from typing import Any, Optional, Dict
import torch
import yacs.config
import numpy as np

from utils.logger import logger_gen
from utils.misc import plot_nifti_batch_comparison, plot_batch_inputs 

logger = logger_gen(__name__)

class Meter:
    """
    Utility class for logging and writing training summaries for pCT synthesis.
    """

    def __init__(self,
                 cfg: yacs.config.CfgNode,
                 mode: str,
                 global_step: int = 0,
                 total_iter: Optional[int] = None,
                 total_epoch: Optional[int] = None,
                 device: Optional[Any] = None,
                 writer: Optional[Any] = None
    ):
        self.cfg = cfg
        self.mode = mode # 'train' o 'val'
        self.global_step = global_step
        self.total_iter = total_iter
        self.total_epoch = total_epoch
        self.device = device
        self.writer = writer
        
        # Almacén temporal para promedios de época
        self.epoch_losses = []
        self.epoch_maes = []
        self.epoch_maes_by_class: Dict[int, list] = {}

        self.class_names = {
            2 : "Body",
            3 : "Lungs",
            4 : "Critical Organs",
            5 : "Bone",
            6 : "Dense Bone",
        }

    def update_stats(self, pred: torch.Tensor, target: torch.Tensor, loss: torch.Tensor, mask: torch.Tensor = None) -> None:
        """Actualiza las estadísticas del batch actual."""
        with torch.no_grad():
            pred_hu = pred * (self.cfg.DATA.CT_MAX - self.cfg.DATA.CT_MIN) + self.cfg.DATA.CT_MIN
            target_hu = target * (self.cfg.DATA.CT_MAX - self.cfg.DATA.CT_MIN) + self.cfg.DATA.CT_MIN
            abs_diff = torch.abs(pred_hu - target_hu)

            if mask is not None:
                mask_idx = torch.round(mask).long()
                valid_voxels = (mask_idx > 0)
                valid_mask_float = valid_voxels.float()
                
                clinical_mae = (abs_diff * valid_mask_float).sum() / (valid_mask_float.sum() + 1e-8)
                mae_val = clinical_mae.item()

                for cls in self.class_names.keys():
                    cls_mask = (mask_idx == cls).float()
                    cls_voxel_count  = cls_mask.sum()
                    if cls_voxel_count > 0:
                        cls_mae = (abs_diff * cls_mask).sum() / cls_voxel_count
                        self.epoch_maes_by_class.setdefault(cls, []).append(cls_mae.item())

            else:
                mae_val = torch.mean(torch.abs(pred_hu - target_hu)).item()
        
        if np.isnan(loss.item()) or np.isnan(mae_val):
            return
        
        self.epoch_losses.append(loss.item())
        self.epoch_maes.append(mae_val)

    def write_summary(self, loss: torch.Tensor, lr: list,
                      loss_components: Optional[Dict[str, float]] = None, 
                      grad_norm: Optional[float] = None
                      ) -> None:
        """Escribe el resumen del batch en TensorBoard."""
        tag = self.mode.capitalize()
        self.writer.add_scalar(f'{tag}/Batch_Loss', loss.item(), self.global_step)

        if loss_components:
            for name, value in loss_components.items():
                self.writer.add_scalar(f'{tag}/Loss_Components/{name}', value, self.global_step)

        if self.mode == 'train':
            self.writer.add_scalar('Train/Learning_Rate', lr[0], self.global_step)
            if grad_norm is not None:
                self.writer.add_scalar('Train/Grad_Norm', grad_norm, self.global_step)
        
        self.global_step += 1

    def log_iter(self, cur_iter: int, cur_epoch: int) -> None:
        """Loguea el progreso dentro de la época."""
        if cur_iter % self.cfg.TRAIN.LOG_INTERVAL == 0:
            avg_mae = np.nanmean(self.epoch_maes[-self.cfg.TRAIN.LOG_INTERVAL:])
            avg_loss = np.nanmean(self.epoch_losses[-self.cfg.TRAIN.LOG_INTERVAL:])
            logger.info(
                f"Epoch [{cur_epoch}/{self.cfg.TRAIN.NUM_EPOCHS}] "
                f"Iter [{cur_iter}/{self.total_iter}] "
                f"MAE: {avg_mae:.2f} HU"
            )
            tag = self.mode.capitalize()
            
            self.writer.add_scalar(f'{tag}/Rolling_MAE', avg_mae, self.global_step)
            self.writer.add_scalar(f'{tag}/Rolling_Loss', avg_loss, self.global_step)

    def log_epoch(self, cur_epoch: int, batch: Optional[dict] = None, pred: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None):
        """Finaliza la época logueando promedios y visualizaciones."""
        avg_loss = np.nanmean(self.epoch_losses)
        avg_mae = np.nanmean(self.epoch_maes)
        tag = self.mode.capitalize()

        # Logs a TensorBoard
        self.writer.add_scalar(f'{tag}/Epoch_Loss', avg_loss, cur_epoch)
        self.writer.add_scalar(f'{tag}/Epoch_MAE', avg_mae, cur_epoch)

        logger.info(f">>> {tag} Epoch {cur_epoch} Summary - Avg Loss: {avg_loss:.4f}, Avg MAE: {avg_mae:.2f} HU")

        # MAE por clase de tejido
        for cls, values in self.epoch_maes_by_class.items():
            if values:
                cls_avg = np.nanmean(values)
                name = self.class_names.get(cls, f"class_{cls}")
                self.writer.add_scalar(f'{tag}/MAE_by_class/{name}', cls_avg, cur_epoch)
                logger.info(f"    {name}: {cls_avg:.2f} HU")

        inputs_dir = os.path.join(self.cfg.LOG_DIR, "comparisons", "inputs")
        preds_dir = os.path.join(self.cfg.LOG_DIR, "comparisons", "predictions")
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(preds_dir, exist_ok=True)

        should_plot = (cur_epoch % 5 == 0) or (cur_epoch == 1)

        if pred is not None and target is not None and should_plot:
            plot_nifti_batch_comparison(
                gt_tensor=target.detach().cpu(),
                denoised_tensor=pred.detach().cpu(),
                writer=self.writer,
                tag=f"{tag}_Comparison",
                step=cur_epoch,
                file_save_name=os.path.join(preds_dir, f"{tag}_epoch_{cur_epoch}.png")
            )
        
        if batch is not None and should_plot:
            plot_batch_inputs(
            batch=batch,
            step=cur_epoch,
            tag=f"{tag}_Inputs",
            writer=self.writer,
            max_samples=self.cfg.TRAIN.BATCH_SIZE,
            save_path=os.path.join(inputs_dir, f"{tag}_inputs_epoch_{cur_epoch}.png")
            )

        self.epoch_losses = []
        self.epoch_maes = []
        self.epoch_maes_by_class = {}
    