"""
models.py

This module defines deep learning models for 3D denoising tasks using PyTorch,
including UNet-based and competitive architecture variants. It supports model building
through configuration-driven design and modular encoder-decoder components.

Dependencies:
    - yacs.config: Handles model configuration via hierarchical config files.
    - torch: Provides tensor operations and neural network modules.
    - models.sub_models: Contains encoder, decoder, bottleneck, and auxiliary model blocks.
    - typing: Offers type hinting support for clarity and maintainability.

Owner:
    Edwing Ulin

Version:
    v1.0.1
"""
import yacs.config
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Union
import models.sub_models as sm
from torch.utils.checkpoint import checkpoint

class MultimodalPCT(nn.Module):
    """
    Network Definition for Multimodal Pseudo-CT Synthesis.

    Attributes:
    ----------
    pet_encoder, mri_encoder:
        Encoders for functional and anatomical local 3D data.
    
    cross_attention:
        Integration of global 2D context.

    coord_cond:
        Anatomical position conditioning.

    decoder_block:
        Final 3D reconstruction block.

    Methods:
    -------
    forward:
        Feed forward through the multimodal network.
    """

    def __init__(self, params: Dict) -> None:
        """
        Construct MultimodalPCT object.
        """
        super(MultimodalPCT, self).__init__()

        # Encoders
        self.pet_encoder = sm.MultimodalEncoder({
            "in_channels": params["num_channels"], 
            "base_filters": params["num_filters"]
        })
        self.mri_encoder = sm.MultimodalEncoder({
            "in_channels": 2 * params["num_channels"], 
            "base_filters": params["num_filters"]
        })

        fusion_dim = params["num_filters"] * 2
        
        self.cross_attention = sm.CrossAttentionBlock({
            "topo_channels": params["topo_channels"], 
            "feature_dim": fusion_dim
        })
        
        self.coord_cond = sm.FiLMLayer({
            "coord_dim": params["coord_dim"], 
            "feature_dim": fusion_dim
        })

        # Decoder
        self.decoder_block = sm.PCTDecoder({
            "in_channels": fusion_dim,
            "hidden_channels": params["num_filters"]
        })


    def forward(self, pet: Tensor, mri: Tensor, topo: Tensor, coords: Tensor) -> Tensor:
        """
        Forward pass.
        """
        # Extract features
        f_pet = self.pet_encoder(pet)
        f_mri = self.mri_encoder(mri)
        x = torch.cat([f_pet, f_mri], dim=1)

        # Apply global attention and spatial conditioning
        x = self.cross_attention(x, topo)
        x = self.coord_cond(x, coords)

        # Final reconstruction
        return self.decoder_block(x)
    
class MultimodalPCTUNet_3(nn.Module):
    """
    Network Definition for Multimodal Pseudo-CT Synthesis con arquitectura
    U-Net real (downsampling + skip connections) para mayor receptive field
    frente a MultimodalPCT. Usa GroupNorm en vez de BatchNorm.

    Attributes:
    ----------
    pet_encode1, pet_encode2, pet_encode3, pet_bottleneck:
        Encoder PET de 4 niveles (3 downsamples + bottleneck).

    mri_encode1, mri_encode2, mri_encode3, mri_bottleneck:
        Encoder MRI de 4 niveles, arquitectura simétrica a PET.

    cross_attention:
        Atención espacial cruzada del Topograma 2D sobre el bottleneck fusionado.

    coord_cond:
        Modulación FiLM con coordenadas anatómicas sobre el bottleneck.

    decode1, decode2, decode3:
        Bloques decoder con upsampling y skip connections fusionadas (PET+MRI).

    out_conv:
        Convolución final 1x1x1 a un canal (pseudo-CT).

    Methods:
    -------
    forward:
        Feed forward through the multimodal U-Net.
    """

    def __init__(self, params: Dict):
        """
        Construct MultimodalPCTUNet object.

        Parameters
        ----------
        params : Dict
            Requiere: num_channels, num_filters, topo_channels, coord_dim.
            Opcionales con default: kernel_d/h/w (3), stride_conv (1),
            stride_pool (2), num_groups (8).
        """
        super(MultimodalPCTUNet_3, self).__init__()

        base = params["num_filters"]
        kd = params.get("kernel_d", 3)
        kh = params.get("kernel_h", 3)
        kw = params.get("kernel_w", 3)
        sc = params.get("stride_conv", 1)
        sp = params.get("stride_pool", 2)
        ng = params.get("num_groups", 8)

        def block_params(in_ch, out_ch):
            return {
                "num_channels": in_ch, "num_filters": out_ch,
                "kernel_d": kd, "kernel_h": kh, "kernel_w": kw,
                "stride_conv": sc, "stride_pool": sp, "num_groups": ng,
            }

        # ---- Encoder PET (nivel completo -> 1/2 -> 1/4 -> 1/8 bottleneck) ----
        self.pet_encode1 = sm.MultimodalEncoderBlock(block_params(params["num_channels"], base))
        self.pet_encode2 = sm.MultimodalEncoderBlock(block_params(base, base * 2))
        self.pet_encode3 = sm.MultimodalEncoderBlock(block_params(base * 2, base * 4))
        self.pet_bottleneck = sm.MultimodalConvBlock(block_params(base * 4, base * 8))

        # ---- Encoder MRI (misma profundidad, canales de entrada distintos) ----
        mri_in_ch = 2 * params["num_channels"]
        self.mri_encode1 = sm.MultimodalEncoderBlock(block_params(mri_in_ch, base))
        self.mri_encode2 = sm.MultimodalEncoderBlock(block_params(base, base * 2))
        self.mri_encode3 = sm.MultimodalEncoderBlock(block_params(base * 2, base * 4))
        self.mri_bottleneck = sm.MultimodalConvBlock(block_params(base * 4, base * 8))

        # ---- Fusión + conditioning en el bottleneck ----
        fused_bottleneck = base * 8 * 2   # concat PET + MRI en el bottleneck
        fused_skip1 = base * 2            # concat PET + MRI en skip nivel 1
        fused_skip2 = base * 2 * 2        # concat PET + MRI en skip nivel 2
        fused_skip3 = base * 4 * 2        # concat PET + MRI en skip nivel 3

        self.cross_attention = sm.SpatialCrossAttentionBlock({
            "topo_channels": params["topo_channels"],
            "feature_dim": fused_bottleneck,
        })
        self.coord_cond = sm.FiLMLayer({
            "coord_dim": params["coord_dim"],
            "feature_dim": fused_bottleneck,
        })

        # ---- Decoder ----
        decode3_params = block_params(fused_bottleneck, base * 4)
        decode3_params["num_skip_channels"] = fused_skip3
        self.decode3 = sm.MultimodalDecoderBlock(decode3_params)

        decode2_params = block_params(base * 4, base * 2)
        decode2_params["num_skip_channels"] = fused_skip2
        self.decode2 = sm.MultimodalDecoderBlock(decode2_params)

        decode1_params = block_params(base * 2, base)
        decode1_params["num_skip_channels"] = fused_skip1
        self.decode1 = sm.MultimodalDecoderBlock(decode1_params)

        self.out_conv = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, pet: Tensor, mri: Tensor, topo: Tensor, coords: Tensor) -> Tensor:
        """
        Forward pass a través de la U-Net multimodal, con gradient checkpointing
        en encoders y decoder para reducir memoria pico (VRAM limitada).

        Args:
            pet (Tensor): Volumen PET [N, C, D, H, W].
            mri (Tensor): Volumen MRI [N, 2C, D, H, W].
            topo (Tensor): Topograma 2D [N, C_topo, H_t, W_t].
            coords (Tensor): Coordenadas anatómicas normalizadas [N, coord_dim].

        Returns:
            Tensor: Volumen pseudo-CT reconstruido [N, 1, D, H, W].
        """
        # ---------------- Encoders (con checkpointing) ----------------
        pet_skip1, pet_x = checkpoint(self.pet_encode1, pet, use_reentrant=False)
        pet_skip2, pet_x = checkpoint(self.pet_encode2, pet_x, use_reentrant=False)
        pet_skip3, pet_x = checkpoint(self.pet_encode3, pet_x, use_reentrant=False)
        pet_bottleneck = checkpoint(self.pet_bottleneck, pet_x, use_reentrant=False)

        mri_skip1, mri_x = checkpoint(self.mri_encode1, mri, use_reentrant=False)
        mri_skip2, mri_x = checkpoint(self.mri_encode2, mri_x, use_reentrant=False)
        mri_skip3, mri_x = checkpoint(self.mri_encode3, mri_x, use_reentrant=False)
        mri_bottleneck = checkpoint(self.mri_bottleneck, mri_x, use_reentrant=False)

        # ---------------- Fusión multimodal por nivel ----------------
        skip1 = torch.cat([pet_skip1, mri_skip1], dim=1)
        skip2 = torch.cat([pet_skip2, mri_skip2], dim=1)
        skip3 = torch.cat([pet_skip3, mri_skip3], dim=1)
        x = torch.cat([pet_bottleneck, mri_bottleneck], dim=1)

        # ---------------- Conditioning global ----------------
        x = self.cross_attention(x, topo)
        x = self.coord_cond(x, coords)

        # ---------------- Decoder (con checkpointing) ----------------
        x = checkpoint(self.decode3, x, skip3, use_reentrant=False)
        x = checkpoint(self.decode2, x, skip2, use_reentrant=False)
        x = checkpoint(self.decode1, x, skip1, use_reentrant=False)

        return self.out_conv(x)

class MultimodalPCTUNet(nn.Module):
    """
    Network Definition for Multimodal Pseudo-CT Synthesis con arquitectura
    U-Net real (downsampling + skip connections) para mayor receptive field
    frente a MultimodalPCT. Usa GroupNorm en vez de BatchNorm.

    Attributes:
    ----------
    pet_encode1, pet_encode2, pet_bottleneck:
        Encoder PET de 3 niveles (2 downsamples + bottleneck).

    mri_encode1, mri_encode2, mri_bottleneck:
        Encoder MRI de 3 niveles, arquitectura simétrica a PET.

    cross_attention:
        Atención espacial cruzada del Topograma 2D sobre el bottleneck fusionado.

    coord_cond:
        Modulación FiLM con coordenadas anatómicas sobre el bottleneck.

    decode1, decode2:
        Bloques decoder con upsampling y skip connections fusionadas (PET+MRI).

    out_conv:
        Convolución final 1x1x1 a un canal (pseudo-CT).

    Methods:
    -------
    forward:
        Feed forward through the multimodal U-Net.
    """

    def __init__(self, params: Dict):
        """
        Construct MultimodalPCTUNet object.

        Parameters
        ----------
        params : Dict
            Requiere: num_channels, num_filters, topo_channels, coord_dim.
            Opcionales con default: kernel_d/h/w (3), stride_conv (1),
            stride_pool (2), num_groups (8).
        """
        super(MultimodalPCTUNet, self).__init__()

        base = params["num_filters"]
        kd = params.get("kernel_d", 3)
        kh = params.get("kernel_h", 3)
        kw = params.get("kernel_w", 3)
        sc = params.get("stride_conv", 1)
        sp = params.get("stride_pool", 2)
        ng = params.get("num_groups", 8)

        def block_params(in_ch, out_ch):
            return {
                "num_channels": in_ch, "num_filters": out_ch,
                "kernel_d": kd, "kernel_h": kh, "kernel_w": kw,
                "stride_conv": sc, "stride_pool": sp, "num_groups": ng,
            }

        # ---- Encoder PET (nivel completo -> 1/2 -> 1/4 bottleneck) ----
        self.pet_encode1 = sm.MultimodalEncoderBlock(block_params(params["num_channels"], base))
        self.pet_encode2 = sm.MultimodalEncoderBlock(block_params(base, base * 2))
        self.pet_bottleneck = sm.MultimodalConvBlock(block_params(base * 2, base * 4))

        # ---- Encoder MRI (misma profundidad, canales de entrada distintos) ----
        mri_in_ch = 2 * params["num_channels"]
        self.mri_encode1 = sm.MultimodalEncoderBlock(block_params(mri_in_ch, base))
        self.mri_encode2 = sm.MultimodalEncoderBlock(block_params(base, base * 2))
        self.mri_bottleneck = sm.MultimodalConvBlock(block_params(base * 2, base * 4))

        # ---- Fusión + conditioning en el bottleneck ----
        fused_bottleneck = base * 4 * 2   # concat PET + MRI en el bottleneck
        fused_skip1 = base * 2            # concat PET + MRI en skip nivel 1
        fused_skip2 = base * 2 * 2        # concat PET + MRI en skip nivel 2

        self.cross_attention = sm.SpatialCrossAttentionBlock({
            "topo_channels": params["topo_channels"],
            "feature_dim": fused_bottleneck,
        })
        self.coord_cond = sm.FiLMLayer({
            "coord_dim": params["coord_dim"],
            "feature_dim": fused_bottleneck,
        })

        # ---- Decoder ----
        decode2_params = block_params(fused_bottleneck, base * 2)
        decode2_params["num_skip_channels"] = fused_skip2
        self.decode2 = sm.MultimodalDecoderBlock(decode2_params)

        decode1_params = block_params(base * 2, base)
        decode1_params["num_skip_channels"] = fused_skip1
        self.decode1 = sm.MultimodalDecoderBlock(decode1_params)

        self.out_conv = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, pet: Tensor, mri: Tensor, topo: Tensor, coords: Tensor) -> Tensor:
        """
        Forward pass a través de la U-Net multimodal, con gradient checkpointing
        en encoders y decoder para reducir memoria pico (VRAM limitada).

        Args:
            pet (Tensor): Volumen PET [N, C, D, H, W].
            mri (Tensor): Volumen MRI [N, 2C, D, H, W].
            topo (Tensor): Topograma 2D [N, C_topo, H_t, W_t].
            coords (Tensor): Coordenadas anatómicas normalizadas [N, coord_dim].

        Returns:
            Tensor: Volumen pseudo-CT reconstruido [N, 1, D, H, W].
        """
        # ---------------- Encoders (con checkpointing) ----------------
        pet_skip1, pet_x = checkpoint(self.pet_encode1, pet, use_reentrant=False)
        pet_skip2, pet_x = checkpoint(self.pet_encode2, pet_x, use_reentrant=False)
        pet_bottleneck = checkpoint(self.pet_bottleneck, pet_x, use_reentrant=False)

        mri_skip1, mri_x = checkpoint(self.mri_encode1, mri, use_reentrant=False)
        mri_skip2, mri_x = checkpoint(self.mri_encode2, mri_x, use_reentrant=False)
        mri_bottleneck = checkpoint(self.mri_bottleneck, mri_x, use_reentrant=False)

        # ---------------- Fusión multimodal por nivel ----------------
        skip1 = torch.cat([pet_skip1, mri_skip1], dim=1)
        skip2 = torch.cat([pet_skip2, mri_skip2], dim=1)
        x = torch.cat([pet_bottleneck, mri_bottleneck], dim=1)

        # ---------------- Conditioning global ----------------
        x = self.cross_attention(x, topo)
        x = self.coord_cond(x, coords)

        # ---------------- Decoder (con checkpointing) ----------------
        x = checkpoint(self.decode2, x, skip2, use_reentrant=False)
        x = checkpoint(self.decode1, x, skip1, use_reentrant=False)

        return self.out_conv(x)

_MODELS = {
    "MultimodalPCT": MultimodalPCT,
    "MultimodalPCTUNet": MultimodalPCTUNet,
    "MultimodalPCTUNet_3": MultimodalPCTUNet_3,
}

def build_model(cfg: any) -> Union[MultimodalPCT, MultimodalPCTUNet, MultimodalPCTUNet_3]:
    """
    Build the model from YACS config node.
    """
    assert (
        cfg.MODEL.MODEL_NAME in _MODELS.keys()
    ), f"Model {cfg.MODEL.MODEL_NAME} not supported"

    params = {k.lower(): v for k, v in dict(cfg.MODEL).items()}
    return _MODELS[cfg.MODEL.MODEL_NAME](params)