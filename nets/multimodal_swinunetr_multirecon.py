"""
This module implements a multi-modal Swin UNETR architecture with feature reconstruction capabilities.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Final, Dict, Any, Optional, List, Tuple
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, Convolution
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from utils.mm_network_utils import (
    window_partition, window_reverse, WindowAttention, SwinTransformerBlock,
    PatchMergingV2, PatchMerging, SwinTransformer, DeepSupervisionHead
)

rearrange, _ = optional_import("einops", name="rearrange")
MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

__all__ = [
    "Multimodal_SwinUNETR",
    "FeatureReconstructionTransformer",
]

class Multimodal_SwinUNETR(nn.Module):
    """
    Multi-modal Swin UNETR with feature reconstruction for missing modalities.

    This network extends the Swin UNETR architecture to handle multiple input modalities (e.g.,
    different MRI sequences). It includes a mechanism to reconstruct features for modalities
    that might be missing in the input data, based on the available modalities.

    The architecture consists of:
    - A separate Swin Transformer backbone for each modality.
    - A shared UNET-like decoder that fuses features from all modalities.
    - Optional feature reconstruction modules at different stages of the Swin Transformer backbones.
    - Separate decoders for each modality for deep supervision during training.
    """

    patch_size: Final[int] = 2

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
        recon_level: str = "hs3",
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes the Multimodal_SwinUNETR model.

        Args:
            img_size: The size of the input image.
            in_channels: The number of input channels for each modality (should be 1).
            out_channels: The number of output channels for the segmentation task.
            depths: The depths of the Swin Transformer stages.
            num_heads: The number of attention heads in the Swin Transformer stages.
            feature_size: The base feature size.
            norm_name: The normalization layer to use.
            drop_rate: The dropout rate.
            attn_drop_rate: The attention dropout rate.
            dropout_path_rate: The stochastic depth rate.
            normalize: Whether to normalize the input.
            use_checkpoint: Whether to use gradient checkpointing.
            spatial_dims: The number of spatial dimensions.
            downsample: The downsampling method to use.
            use_v2: Whether to use Swin Transformer V2.
            recon_level: The level at which to perform feature reconstruction ('hs3', 'hs4', 'hs3_hs4', or 'none').
            device: The device to run the model on.
        """
        super().__init__()

        self._validate_inputs(img_size, drop_rate, attn_drop_rate, dropout_path_rate, feature_size, spatial_dims)

        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        self.window_size = ensure_tuple_rep(7, spatial_dims)
        self.normalize = normalize
        self.feature_size = feature_size
        self.is_training = True
        self.recon_level = recon_level.lower()
        self.device = device
        self.recon_embed_dim = 256
        self.recon_transformer_depth = 4
        self.recon_transformer_heads = 8

        self.hs3_recon_config = self._setup_reconstruction_module(3) if "hs3" in self.recon_level else None
        self.hs4_recon_config = self._setup_reconstruction_module(4) if "hs4" in self.recon_level else None

        self.swinViTs = self._create_swin_transformers(in_channels, depths, num_heads, drop_rate, attn_drop_rate, dropout_path_rate, use_checkpoint, spatial_dims, downsample, use_v2)
        self._create_encoders(spatial_dims, norm_name)
        self._create_channel_reductions(spatial_dims)
        self._create_decoders(spatial_dims, out_channels, norm_name)
        self._create_deep_supervision_heads(out_channels)

    def _validate_inputs(self, img_size, drop_rate, attn_drop_rate, dropout_path_rate, feature_size, spatial_dims):
        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")
        self._check_input_size(ensure_tuple_rep(img_size, spatial_dims))
        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")
        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")
        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")
        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

    def _setup_reconstruction_module(self, stage_idx: int) -> Dict[str, Any]:
        return self._initialize_recon_modules(
            stage_swin_list_idx=stage_idx,
            img_size_tuple=self.img_size,
            patch_sizes_tuple=self.patch_sizes,
            base_feature_size=self.feature_size,
            recon_embed_dim=self.recon_embed_dim,
            recon_transformer_depth=self.recon_transformer_depth,
            recon_transformer_heads=self.recon_transformer_heads,
            device=self.device
        )

    def _create_swin_transformers(self, in_channels, depths, num_heads, drop_rate, attn_drop_rate, dropout_path_rate, use_checkpoint, spatial_dims, downsample, use_v2):
        return nn.ModuleList([
            SwinTransformer(
                in_chans=in_channels, embed_dim=self.feature_size, window_size=self.window_size,
                patch_size=self.patch_sizes, depths=depths, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=dropout_path_rate, norm_layer=nn.LayerNorm, use_checkpoint=use_checkpoint,
                spatial_dims=spatial_dims, downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
                use_v2=use_v2,
            ) for _ in range(4)
        ])

    def _create_encoders(self, spatial_dims, norm_name):
        encoder_configs = [
            (self.feature_size, self.feature_size),
            (self.feature_size, self.feature_size),
            (2 * self.feature_size, 2 * self.feature_size),
            (4 * self.feature_size, 4 * self.feature_size),
            (16 * self.feature_size, 16 * self.feature_size)
        ]
        for i, (in_c, out_c) in enumerate(encoder_configs):
            in_channels = in_c if i > 0 else 1
            setattr(self, f"encoder{i+1}_list", nn.ModuleList([
                UnetrBasicBlock(
                    spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_c,
                    kernel_size=3, stride=1, norm_name=norm_name, res_block=True
                ) for _ in range(4)
            ]))

    def _create_channel_reductions(self, spatial_dims):
        reduction_configs = [
            (4 * self.feature_size, self.feature_size), (4 * self.feature_size, self.feature_size),
            (8 * self.feature_size, 2 * self.feature_size), (16 * self.feature_size, 4 * self.feature_size),
            (32 * self.feature_size, 8 * self.feature_size), (64 * self.feature_size, 16 * self.feature_size)
        ]
        self.channel_reductions = nn.ModuleList([
            Convolution(
                spatial_dims=spatial_dims, in_channels=in_c, out_channels=out_c,
                strides=1, kernel_size=1, dropout=0.2
            ) for in_c, out_c in reduction_configs
        ])

    def _create_decoder_block(self, spatial_dims, out_channels, norm_name):
        return nn.ModuleDict({
            "d5": UnetrUpBlock(spatial_dims, 16 * self.feature_size, 8 * self.feature_size, 3, 2, norm_name, True),
            "d4": UnetrUpBlock(spatial_dims, 8 * self.feature_size, 4 * self.feature_size, 3, 2, norm_name, True),
            "d3": UnetrUpBlock(spatial_dims, 4 * self.feature_size, 2 * self.feature_size, 3, 2, norm_name, True),
            "d2": UnetrUpBlock(spatial_dims, 2 * self.feature_size, self.feature_size, 3, 2, norm_name, True),
            "d1": UnetrUpBlock(spatial_dims, self.feature_size, self.feature_size, 3, 2, norm_name, True),
            "out": UnetOutBlock(spatial_dims, self.feature_size, out_channels),
        })

    def _create_decoders(self, spatial_dims, out_channels, norm_name):
        self.decoder = self._create_decoder_block(spatial_dims, out_channels, norm_name)
        self.modality_decoders = nn.ModuleList([
            self._create_decoder_block(spatial_dims, 3, norm_name) for _ in range(4)
        ])

    def _create_deep_supervision_heads(self, out_channels):
        ds_configs = [
            (16 * self.feature_size, 32), (8 * self.feature_size, 16),
            (4 * self.feature_size, 8), (2 * self.feature_size, 4),
            (self.feature_size, 2)
        ]
        self.deep_supervision_heads = nn.ModuleList([
            DeepSupervisionHead(in_channels=in_c, out_channels=out_channels, scale_factor=sf)
            for in_c, sf in ds_configs
        ])

    def _calculate_stage_dims_channels(self, stage_idx_in_swin_list: int,
                                     img_size: Tuple[int, ...],
                                     patch_sizes_tuple: Tuple[int, ...],
                                     base_feature_size: int) -> Dict[str, Any]:
        s_d, s_h, s_w = (
            img_size[0] // patch_sizes_tuple[0],
            img_size[1] // patch_sizes_tuple[1],
            img_size[2] // patch_sizes_tuple[2],
        )

        if stage_idx_in_swin_list in [3, 4]:
            downscale_factor = 2**stage_idx_in_swin_list
            s_d, s_h, s_w = s_d // downscale_factor, s_h // downscale_factor, s_w // downscale_factor
            channels = base_feature_size * (2**stage_idx_in_swin_list)
        else:
            raise ValueError(f"Unsupported stage_idx_in_swin_list for reconstruction: {stage_idx_in_swin_list}. Only 3 or 4 supported.")

        num_patches = s_d * s_h * s_w
        return {"spatial_dims": (s_d, s_h, s_w), "num_patches": num_patches, "channels": channels}

    def _initialize_recon_modules(self, stage_swin_list_idx: int, img_size_tuple: Tuple[int, ...],
                                  patch_sizes_tuple: Tuple[int, ...], base_feature_size: int,
                                  recon_embed_dim: int, recon_transformer_depth: int,
                                  recon_transformer_heads: int,
                                  device: Optional[torch.device] = None) -> Dict[str, Any]:
        stage_info = self._calculate_stage_dims_channels(stage_swin_list_idx, img_size_tuple, patch_sizes_tuple, base_feature_size)

        projector_to_embed = nn.Conv3d(stage_info["channels"], recon_embed_dim, kernel_size=1).to(device)
        pos_embed = nn.Parameter(torch.zeros(1, stage_info["num_patches"], recon_embed_dim)).to(device)
        reconstructor = FeatureReconstructionTransformer(
            embed_dim=recon_embed_dim, depth=recon_transformer_depth,
            num_heads=recon_transformer_heads, mlp_ratio=4.0,
            num_patches=stage_info["num_patches"]
        ).to(device)
        projector_from_embed = nn.Conv3d(recon_embed_dim, stage_info["channels"], kernel_size=1).to(device)

        return {
            "projector_to_embed": projector_to_embed, "pos_embed": pos_embed,
            "reconstructor": reconstructor, "projector_from_embed": projector_from_embed,
            "num_patches": stage_info["num_patches"], "spatial_dims": stage_info["spatial_dims"],
            "input_channels": stage_info["channels"]
        }

    def _perform_reconstruction_for_stage(self,
                                          raw_features_current_stage: List[torch.Tensor],
                                          is_modality_present_list: List[bool],
                                          num_missing_modalities_val: int,
                                          recon_config: Dict[str, Any]) -> List[torch.Tensor]:
        final_features = list(raw_features_current_stage)
        if num_missing_modalities_val == 0 or recon_config is None:
            return final_features

        available_features = []
        for i, present in enumerate(is_modality_present_list):
            if present:
                feat_map = raw_features_current_stage[i]
                projected = recon_config["projector_to_embed"](feat_map)
                flat_feat = projected.flatten(2).transpose(1, 2) + recon_config["pos_embed"]
                available_features.append(flat_feat)

        if not available_features:
            return final_features

        concatenated_available = torch.cat(available_features, dim=1)
        reconstructed_output = recon_config["reconstructor"](
            concatenated_available, num_missing_modalities_val, recon_config["num_patches"]
        )

        recon_idx = 0
        for i, present in enumerate(is_modality_present_list):
            if not present:
                start = recon_idx * recon_config["num_patches"]
                end = start + recon_config["num_patches"]
                if end <= reconstructed_output.shape[1]:
                    flat_recon = reconstructed_output[:, start:end, :]
                    b, _, e = flat_recon.shape
                    d, h, w = recon_config["spatial_dims"]
                    unflat_recon = flat_recon.transpose(1, 2).reshape(b, e, d, h, w)
                    final_features[i] = recon_config["projector_from_embed"](unflat_recon)
                recon_idx += 1
        return final_features

    def forward(self, x_in: torch.Tensor, modalities_dropped_info: Optional[list] = None):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        is_modality_present = [True] * 4
        num_missing = 0
        if modalities_dropped_info:
            for idx in modalities_dropped_info:
                if 0 <= idx < 4:
                    is_modality_present[idx] = False
            num_missing = len(modalities_dropped_info)

        hidden_states = [vit(x_in[:, i:i+1], normalize=self.normalize)[0] for i, vit in enumerate(self.swinViTs)]

        hs_raw = [[h[j] for h in hidden_states] for j in range(5)]

        current_hs3 = self._perform_reconstruction_for_stage(hs_raw[3], is_modality_present, num_missing, self.hs3_recon_config) if self.hs3_recon_config else hs_raw[3]
        current_hs4 = self._perform_reconstruction_for_stage(hs_raw[4], is_modality_present, num_missing, self.hs4_recon_config) if self.hs4_recon_config else hs_raw[4]

        enc_outs = [
            [self.encoder1_list[i](x_in[:, i:i+1]) for i in range(4)],
            [self.encoder2_list[i](hs_raw[0][i]) for i in range(4)],
            [self.encoder3_list[i](hs_raw[1][i]) for i in range(4)],
            [self.encoder4_list[i](hs_raw[2][i]) for i in range(4)],
            [self.encoder5_list[i](current_hs4[i]) for i in range(4)]
        ]

        fused_skips = [self.channel_reductions[i](torch.cat(enc_outs[i], dim=1)) for i in range(4)]
        fused_skips.append(self.channel_reductions[5](torch.cat(enc_outs[4], dim=1)))
        
        hs_combined = self.channel_reductions[4](torch.cat(current_hs3, dim=1))

        dec3 = self.decoder["d5"](fused_skips[4], hs_combined)
        dec2 = self.decoder["d4"](dec3, fused_skips[3])
        dec1 = self.decoder["d3"](dec2, fused_skips[2])
        dec0 = self.decoder["d2"](dec1, fused_skips[1])
        out = self.decoder["d1"](dec0, fused_skips[0])
        logits = self.decoder["out"](out)

        if self.is_training:
            sep_dec_outputs = []
            for i in range(4):
                dec = self.modality_decoders[i]
                d3 = dec["d5"](enc_outs[4][i], hs_raw[3][i])
                d2 = dec["d4"](d3, enc_outs[3][i])
                d1 = dec["d3"](d2, enc_outs[2][i])
                d0 = dec["d2"](d1, enc_outs[1][i])
                o = dec["d1"](d0, enc_outs[0][i])
                sep_dec_outputs.append(dec["out"](o))

            ds_outputs = [
                self.deep_supervision_heads[4](dec0), self.deep_supervision_heads[3](dec1),
                self.deep_supervision_heads[2](dec2), self.deep_supervision_heads[1](dec3),
                self.deep_supervision_heads[0](fused_skips[4])
            ]
            return logits, ds_outputs, sep_dec_outputs
        
        return logits

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % (self.patch_size ** 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape}) must be divisible by {self.patch_size}**5.")

class FeatureReconstructionTransformer(nn.Module):
    """
    A transformer-based module to reconstruct missing feature representations.
    It takes concatenated features from available modalities and a set of learnable query tokens
    (representing the missing modalities) and outputs the reconstructed features for the missing ones.
    """
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4., num_patches=None):
        super().__init__()
        self.num_patches_per_modality = num_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        if self.num_patches_per_modality is not None:
            self.missing_tokens = nn.Parameter(torch.randn(1, self.num_patches_per_modality, embed_dim))
        else:
            self.missing_tokens = None

    def forward(self, available_features_cat: torch.Tensor, num_missing_modalities: int, target_num_patches_per_modality: int):
        B, S_avail, E = available_features_cat.shape
        if num_missing_modalities == 0:
            return torch.empty(B, 0, E, device=available_features_cat.device)

        if self.missing_tokens is not None and self.missing_tokens.shape[1] == target_num_patches_per_modality:
            queries = self.missing_tokens.expand(B, -1, -1).repeat_interleave(num_missing_modalities, dim=1)
        else:
            queries = torch.zeros(B, num_missing_modalities * target_num_patches_per_modality, E, device=available_features_cat.device)

        transformer_input = torch.cat([available_features_cat, queries], dim=1)
        transformer_output = self.transformer_encoder(transformer_input)
        return transformer_output[:, S_avail:]