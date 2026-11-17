"""
sub_models.py

Módulos y bloques de construcción para la red Multimodal pseudo-CT.
Incluye Encoders 3D, Atención Cruzada para el Topograma 2D y modulación FiLM.

Owner: Edwing Ulin
Version: v1.0.0
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

class MultimodalEncoder(nn.Module):
    """
    Encoder 3D estándar para extraer características locales de PET o MRI.
    """
    def __init__(self, params: Dict):
        super().__init__()
        in_ch = params["in_channels"]
        filters = params["base_filters"]
        
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossAttentionBlock(nn.Module):
    """
    Inyecta el contexto global 2D del Topograma en las características 3D
    usando un mecanismo estilo Squeeze-and-Excitation.
    """
    def __init__(self, params: Dict):
        super().__init__()
        topo_ch = params["topo_channels"]
        feat_dim = params["feature_dim"]
        
        self.topo_pool = nn.Sequential(
            nn.Conv2d(topo_ch, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(16, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, x_3d: torch.Tensor, topo_2d: torch.Tensor) -> torch.Tensor:
        topo_feat = self.topo_pool(topo_2d)
        
        topo_vec = topo_feat.view(topo_feat.size(0), -1)
        
        attn_weights = self.attention_mlp(topo_vec)
        
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return x_3d * attn_weights


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Inyecta las coordenadas espaciales (Z, Y, X) en la red modulando las 
    características con escalado (gamma) y desplazamiento (beta).
    """
    def __init__(self, params: Dict):
        super().__init__()
        coord_dim = params["coord_dim"]
        feat_dim = params["feature_dim"]
        
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, feat_dim * 2)
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        film_params = self.mlp(coords)
        
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return (1 + gamma) * x + beta


class PCTDecoder(nn.Module):
    """
    Decodificador final que reduce los canales fusionados al pseudo-CT final.
    """
    def __init__(self, params: Dict):
        super().__init__()
        in_ch = params["in_channels"]
        hidden = params["hidden_channels"]
        
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden // 2, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class MultimodalConvBlock(nn.Module):
    """
    Bloque convolucional doble para encoders/decoders multimodales 3D.
    Usa GroupNorm en vez de BatchNorm para mayor estabilidad con batch
    pequeño (ver EncoderBlock/DenseBlock para el equivalente con BatchNorm).

    Methods:
    -------
    forward:
        Feed forward through the graph.
    """

    def __init__(self, params: Dict):
        """
        Construct MultimodalConvBlock object.

        Parameters
        ----------
        params : Dict
            Dictionary containing the parameters for the block.
        """

        super(MultimodalConvBlock, self).__init__()

        padding = self._compute_padding(params)
        in_size = int(params["num_channels"])
        out_size = int(params["num_filters"])
        num_groups = min(params.get("num_groups", 8), out_size)

        self.conv0 = nn.Conv3d(
            in_channels =in_size,
            out_channels =out_size,
            kernel_size=(params["kernel_d"], params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=padding,
        )

        self.conv1 = nn.Conv3d(
            in_channels=out_size,
            out_channels=out_size,
            kernel_size=(params["kernel_d"], params["kernel_h"], params["kernel_w"]),
            stride=params["stride_conv"],
            padding=padding,
        )

        self.gn0 = nn.GroupNorm(num_groups, out_size)
        self.gn1 = nn.GroupNorm(num_groups, out_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    @staticmethod
    def _compute_padding(params: Dict) -> tuple[int, int, int]:
        """
        Compute padding based on kernel sizes to maintain spatial dimensions.

        Args:
            params (Dict): Configuration dictionary containing kernel sizes.

        Returns:
            tuple: Padding values for each dimension.
        """
        return (
            params["kernel_d"] // 2,
            params["kernel_h"] // 2,
            params["kernel_w"] // 2,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the conv block.

        Args:
            x (Tensor): Input tensor [N, C, D, H, W].

        Returns:
            Tensor: Processed tensor after two conv + GroupNorm + LeakyReLU.
        """
        x1 = self.leaky_relu(self.gn0(self.conv0(x)))
        out = self.leaky_relu(self.gn1(self.conv1(x1)))
        return out
    
class MultimodalEncoderBlock(MultimodalConvBlock):
    """
    Encoder Block = MultimodalConvBlock + downsampling conv (stride 2).
    Devuelve tanto el feature de resolución completa (para skip connection)
    como el feature ya downsampleado (para el siguiente nivel).

    Methods:
    -------
    forward:
        Feed forward through the encoder block.
    """

    def __init__(self, params: Dict):
        """
        Initialize the encoder block.

        Args:
            params (Dict): Configuration dictionary containing necessary parameters.
                Requires the same keys as MultimodalConvBlock plus stride_pool.
        """
        super(MultimodalEncoderBlock, self).__init__(params)

        padding = self._compute_padding(params)
        out_size = int(params["num_filters"])
        num_groups = min(params.get("num_groups", 8), out_size)

        self.down_conv = nn.Conv3d(
            in_channels=out_size,
            out_channels=out_size,
            kernel_size=(params["kernel_d"], params["kernel_h"], params["kernel_w"]),
            stride=params["stride_pool"],
            padding=padding,
        )
        self.down_gn = nn.GroupNorm(num_groups, out_size)
        self.down_act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the encoder block.

        Args:
            x (Tensor): Input tensor from the previous block.

        Returns:
            Tuple[Tensor, Tensor]:
                skip: feature de resolución completa (previo al downsample).
                out: feature downsampleado, entrada del siguiente nivel.
        """
        skip = super(MultimodalEncoderBlock, self).forward(x)
        out = self.down_act(self.down_gn(self.down_conv(skip)))
        return skip, out    
    
class MultimodalDecoderBlock(MultimodalConvBlock):
    """
    Decoder Block usando upsampling por transposed conv + skip connection,
    seguido de un MultimodalConvBlock.

    Methods:
    -------
    forward:
        Feed forward through the decoder block.
    """

    def __init__(self, params: Dict):
        """
        Initialize the decoder block.

        Args:
            params (Dict): Configuration dictionary. Requires num_channels
                (canales de entrada tras upsample), num_skip_channels
                (canales del skip a concatenar), num_filters (canales
                del bloque conv resultante), plus kernel/stride keys.
        """
        up_in = int(params["num_channels"])
        up_out = int(params["num_filters"])

        # El bloque conv final recibe (up_out + skip_channels) como entrada
        conv_params = dict(params)
        conv_params["num_channels"] = up_out + int(params["num_skip_channels"])
        super(MultimodalDecoderBlock, self).__init__(conv_params)

        self.up = nn.ConvTranspose3d(
            in_channels=up_in,
            out_channels=up_out,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        Forward through decoder block: upsample, concat con skip, doble conv.

        Args:
            x (Tensor): Input desde el nivel inferior del decoder.
            skip (Tensor): Skip connection del encoder correspondiente.

        Returns:
            Tensor: Output tensor tras upsample + fusión + conv block.
        """
        x_up = self.up(x)

        # Ajuste defensivo por si el tamaño no calza exacto (ej. patch no divisible entre 4)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = nn.functional.interpolate(
                x_up, size=skip.shape[2:], mode="trilinear", align_corners=True
            )

        x_cat = torch.cat([x_up, skip], dim=1)
        out = super(MultimodalDecoderBlock, self).forward(x_cat)
        return out    
    

class SpatialCrossAttentionBlock(nn.Module):
    """
    Atención cruzada espacial: proyecta el topograma 2D a un mapa de
    atención por canal que preserva correspondencia en (H, W) y se
    expande a lo largo del eje de profundidad (D) del volumen 3D.

    Methods:
    -------
    forward:
        Feed forward through the attention block.
    """

    def __init__(self, params: Dict):
        """
        Construct SpatialCrossAttentionBlock object.

        Parameters
        ----------
        params : Dict
            Requires: topo_channels, feature_dim.
        """
        super(SpatialCrossAttentionBlock, self).__init__()
        topo_ch = params["topo_channels"]
        feat_dim = params["feature_dim"]

        self.topo_encoder = nn.Sequential(
            nn.Conv2d(topo_ch, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, feat_dim, kernel_size=3, stride=1, padding=1),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x_3d: Tensor, topo_2d: Tensor) -> Tensor:
        """
        Forward pass modulando x_3d con el mapa de atención del topograma.

        Args:
            x_3d (Tensor): Feature volumétrico [N, C, D, H, W].
            topo_2d (Tensor): Topograma 2D [N, C_topo, H_t, W_t].

        Returns:
            Tensor: x_3d modulado espacialmente.
        """
        attn_map = self.topo_encoder(topo_2d)

        _, _, d, h, w = x_3d.shape
        attn_map = nn.functional.interpolate(
            attn_map, size=(h, w), mode="bilinear", align_corners=False
        )
        attn_map = self.gate(attn_map)
        attn_map = attn_map.unsqueeze(2).expand(-1, -1, d, -1, -1)

        return x_3d * attn_map

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Inyecta coordenadas espaciales (Z, Y, X) modulando las características
    con escalado (gamma) y desplazamiento (beta).

    Methods:
    -------
    forward:
        Feed forward through the FiLM layer.
    """

    def __init__(self, params: Dict):
        """
        Construct FiLMLayer object.

        Parameters
        ----------
        params : Dict
            Requires: coord_dim, feature_dim.
        """
        super(FiLMLayer, self).__init__()
        coord_dim = params["coord_dim"]
        feat_dim = params["feature_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, feat_dim * 2),
        )

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        """
        Forward pass aplicando modulación FiLM.

        Args:
            x (Tensor): Feature volumétrico [N, C, D, H, W].
            coords (Tensor): Coordenadas normalizadas [N, coord_dim].

        Returns:
            Tensor: x modulado por (1 + gamma) * x + beta.
        """
        film_params = self.mlp(coords)
        gamma, beta = torch.chunk(film_params, 2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return (1 + gamma) * x + beta