"""
losses.py

This module defines custom PyTorch loss functions for multimodal pseudo-CT generation,
focusing on voxel-wise L1 accuracy and 2D projection alignment.

Dependencies:
    - yacs.config: Handles configuration settings.
    - torch: Provides tensor operations and loss computation.
    - torch.nn: Supplies the loss function base class and functional APIs.
    - typing: Supports type hinting for clarity and robustness.

Owner:
    Edwing Ulin

Version:
    v1.1.0
"""

import torch
import yacs.config
from torch import Tensor
from torch.nn import functional as f
from torch.nn.modules.loss import _Loss
from typing import Union
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


class L1Loss(_Loss):
    """
    L1 Loss (Mean Absolute Error) for 3D tensors.
    Computes the absolute difference between predicted and target values.

    Methods
    -------
    forward:
        Computes the L1 loss between predicted output and target for 3D volumes.
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Initializes the L1Loss object.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the L1 loss (MAE) for 3D tensors.
        """
        return f.l1_loss(output, target, reduction=self.reduction)

class GradientLoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor, weight_map: Tensor = None) -> Tensor:
        dx_pred = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        dy_pred = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        dz_pred = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])

        dx_target = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
        dy_target = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        dz_target = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])

        diff_x = torch.abs(dx_pred - dx_target)
        diff_y = torch.abs(dy_pred - dy_target)
        diff_z = torch.abs(dz_pred - dz_target)

        if weight_map is not None:
            if weight_map.dim() == 4:
                weight_map = weight_map.unsqueeze(1)

            wx = weight_map[:, :, :, :, 1:]
            wy = weight_map[:, :, :, 1:, :]
            wz = weight_map[:, :, 1:, :, :]

            loss_x = (diff_x * wx).sum() / (wx.sum() + 1e-8)
            loss_y = (diff_y * wy).sum() / (wy.sum() + 1e-8)
            loss_z = (diff_z * wz).sum() / (wz.sum() + 1e-8)
        else:
            loss_x = diff_x.mean() if self.reduction == 'mean' else diff_x.sum()
            loss_y = diff_y.mean() if self.reduction == 'mean' else diff_y.sum()
            loss_z = diff_z.mean() if self.reduction == 'mean' else diff_z.sum()

        return loss_x + loss_y + loss_z

class FastStochasticSSIM(_Loss):
    def __init__(self, num_slices: int = 4, window_size: int = 7, C1: float = (0.01)**2, C2: float = (0.03)**2):
        super().__init__()
        self.num_slices = num_slices
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
        b, c, d, h, w = pred.shape

        if mask is not None and d > self.num_slices:
            slice_scores = mask.float().sum(dim=(1, 3, 4)).squeeze()  # (b, d) aprox
            if slice_scores.dim() == 1:
                slice_scores = slice_scores.unsqueeze(0)
            probs = torch.softmax(slice_scores[0] + 1e-3, dim=0)  # simplificado a nivel batch
            indices = torch.multinomial(probs, self.num_slices, replacement=False).to(pred.device)
        elif d > self.num_slices:
            indices = torch.randint(0, d, (self.num_slices,), device=pred.device)
        else:
            indices = torch.arange(d, device=pred.device)

        pred_2d = pred[:, :, indices, :, :].reshape(-1, c, h, w)
        target_2d = target[:, :, indices, :, :].reshape(-1, c, h, w)

        pad = self.window_size // 2
        mu1 = f.avg_pool2d(pred_2d, self.window_size, stride=1, padding=pad)
        mu2 = f.avg_pool2d(target_2d, self.window_size, stride=1, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

        sigma1_sq = f.avg_pool2d(pred_2d*pred_2d, self.window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = f.avg_pool2d(target_2d*target_2d, self.window_size, stride=1, padding=pad) - mu2_sq
        sigma12 = f.avg_pool2d(pred_2d*target_2d, self.window_size, stride=1, padding=pad) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + self.C1) * (2*sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return 1.0 - ssim_map.mean()


class MultimodalPCTLoss(_Loss):
    """
    Combines L1 Voxel Loss and Projection Loss for multimodal pseudo-CT synthesis.

    Attributes:
    -----------
    l1_weight : float
        Weight for the voxel-wise MAE.
    proj_weight : float
        Weight for the 2D projection consistency (Topogram alignment).
    grad_weight : float
        Weight for the spatial gradients
    ssim_weight : float
        Weight for the structural similarity index
    """

    def __init__(
        self, 
        l1_weight: float = 1.0, 
        proj_weight: float = 0.1,
        grad_weight: float = 0.0,
        ssim_weight: float = 0.0,
        ssim_window_size: int = 11,
        magnitude_weight_k: float = 3.0,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.proj_weight = proj_weight
        self.grad_weight = grad_weight
        self.ssim_weight = ssim_weight
        self.ssim_window_size = ssim_window_size
        self.magnitude_weight_k = magnitude_weight_k
        self.reduction = reduction

        if self.grad_weight > 0:
            self.grad_loss = GradientLoss(reduction=reduction)
        if self.ssim_weight > 0:
            self.ssim_loss = FastStochasticSSIM(num_slices=4, window_size=self.ssim_window_size)

    def forward(self, output: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for the multimodal loss function.

        Parameters:
        ----------
        output : Tensor
            The predicted pseudo-CT volume (B, 1, D, H, W).
        target : Tensor
            The ground truth CT volume (B, 1, D, H, W).
        mask: Tensor
            Binary mask indicating valid regions for loss calculation (B, 1, D, H, W).

        Returns:
        --------
        Tensor:
            Weighted sum of L1 and Projection loss.
        """
        loss = 0.0
        loss_l1 = torch.tensor(0.0)
        loss_proj = torch.tensor(0.0)
        loss_grad = torch.tensor(0.0)
        loss_ssim = torch.tensor(0.0)

        if mask is not None:
            mask_idx = torch.round(mask).long()
            weight_map = torch.ones_like(output)
            
            weight_map[mask_idx == 0] = 0.0
            weight_map[mask_idx == 2] = 1.0
            weight_map[mask_idx == 3] = 10.0
            weight_map[mask_idx == 4] = 3.0
            weight_map[mask_idx == 5] = 3.0
            weight_map[mask_idx == 6] = 10.0

            if weight_map.dim() == 4:
                weight_map = weight_map.unsqueeze(1)
            
            weight_map = F.max_pool3d(weight_map, kernel_size=3, stride=1, padding=1)
            
            kernel_size = 5
            sigma = 1.0
        
            coords = torch.arange(kernel_size).float() - kernel_size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            kernel = (g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1))
            kernel = kernel / kernel.sum()
            kernel = kernel.repeat(1, 1, 1, 1, 1).to(output.device) # (Out_C, In_C, D, H, W)
            
            weight_map = F.conv3d(weight_map, kernel, padding=kernel_size // 2)
            weight_map = weight_map.squeeze(1) # Volver a (B, D, H, W)
        else:
            weight_map = None

        if self.l1_weight > 0:
                    if weight_map is not None:
                        smooth_diff = torch.nn.functional.smooth_l1_loss(output, target, reduction='none', beta=0.05)

                        magnitude_k = getattr(self, "magnitude_weight_k", 3.0)
                        magnitude_weight = 1.0 + magnitude_k * torch.clamp(target, min=0.0, max=1.0)

                        combined_weight = weight_map * magnitude_weight
                        weighted_diff = smooth_diff * combined_weight

                        loss_l1 = weighted_diff.sum() / (combined_weight.sum() + 1e-8)
                        loss += self.l1_weight * loss_l1
                    else:
                        loss_l1 = torch.nn.functional.smooth_l1_loss(output, target, reduction=self.reduction, beta=0.05)
                        loss += self.l1_weight * loss_l1

        if self.proj_weight > 0:
                    proj_output = torch.mean(output, dim=3) 
                    proj_target = torch.mean(target, dim=3)

                    if weight_map is not None:
                        weight_map_proj = torch.mean(weight_map, dim=3)
                        proj_diff = torch.abs(proj_output - proj_target) * weight_map_proj
                        loss_proj = proj_diff.sum() / (weight_map_proj.sum() + 1e-8)
                    else: 
                        loss_proj = torch.nn.functional.l1_loss(proj_output, proj_target, reduction=self.reduction)
                    
                    loss += self.proj_weight *  loss_proj

        if self.grad_weight > 0:
            loss_grad = self.grad_loss(output, target, weight_map=weight_map)
            loss += self.grad_weight * loss_grad

        if self.ssim_weight > 0:
            loss_ssim = self.ssim_loss(output, target, mask=mask)
            loss += self.ssim_weight * loss_ssim
        
        self.last_components = {
            "l1": float(loss_l1.detach()),
            "proj": float(loss_proj.detach()),
            "grad": float(loss_grad.detach()),
            "ssim": float(loss_ssim.detach()),
        }
        return loss


def get_loss_func(cfg: yacs.config.CfgNode) -> Union[L1Loss, MultimodalPCTLoss]:
    """
    Retrieve the specified loss function from the configuration.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Configuration node containing the selected loss function.

    Returns
    -------
    Union[L1Loss, MultimodalPCTLoss]
        The initialized loss function object.

    Raises
    ------
    NotImplementedError
        If the requested loss function is not supported.
    """
    loss_func_name = cfg.MODEL.LOSS_FUNC

    if loss_func_name == 'l1':
        return L1Loss(reduction='mean')
    elif loss_func_name == 'multimodal_pct':
        return MultimodalPCTLoss(
            l1_weight=getattr(cfg.MODEL, "L1_WEIGHT", 1.0),
            proj_weight=getattr(cfg.MODEL, "PROJ_WEIGHT", 0.1),
            grad_weight=getattr(cfg.MODEL, "GRAD_WEIGHT", 0.0),
            ssim_weight=getattr(cfg.MODEL, "SSIM_WEIGHT", 0.0),
            ssim_window_size=getattr(cfg.MODEL, "SSIM_WINDOW_SIZE", 11),
            magnitude_weight_k=getattr(cfg.MODEL, "MAGNITUDE_WEIGHT_K", 3.0),
            reduction='mean'
        )
    else:
        raise NotImplementedError(f"The loss function '{loss_func_name}' is not supported.")