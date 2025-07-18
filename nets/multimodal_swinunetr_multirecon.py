from __future__ import annotations
from collections.abc import Sequence
from typing_extensions import Final, Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import (
    UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, Convolution
)
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from utils.mm_network_utils import (
    window_partition, window_reverse, WindowAttention, SwinTransformerBlock,
    PatchMergingV2, PatchMerging, SwinTransformer, DeepSupervisionHead
)

rearrange, _ = optional_import("einops", name="rearrange")
MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]

# =========================
# Main Model
# =========================

class Multimodal_SwinUNETR(nn.Module):
    """
    Swin UNETR for multimodal MRI segmentation.

    Based on: "Hatamizadeh et al., Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"
    <https://arxiv.org/abs/2201.01266>

    Args:
        img_size (Sequence[int] | int): Input image size.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        depths (Sequence[int]): Swin transformer depths.
        num_heads (Sequence[int]): Swin transformer heads.
        feature_size (int): Base feature size.
        norm_name (tuple | str): Normalization type.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        dropout_path_rate (float): Drop path rate.
        normalize (bool): Whether to normalize input.
        use_checkpoint (bool): Use checkpointing.
        spatial_dims (int): Number of spatial dimensions.
        downsample (str): Downsampling mode.
        use_v2 (bool): Use v2 blocks.
        recon_level (str): Feature reconstruction level.
        device: Device to use.
    """

    patch_size: Final[int] = 2

    def _calculate_stage_dims_channels(
        self, stage_idx_in_swin_list: int, img_size: Tuple[int, ...],
        patch_sizes_tuple: Tuple[int, ...], base_feature_size: int
    ) -> Dict[str, Any]:
        """
        Calculate spatial dimensions, number of patches, and channels for a given Swin stage output.

        Args:
            stage_idx_in_swin_list (int): Index in the hidden_states_out list from SwinTransformer.
            img_size (Tuple[int, ...]): Input image size.
            patch_sizes_tuple (Tuple[int, ...]): Patch sizes.
            base_feature_size (int): Base feature size.

        Returns:
            Dict[str, Any]: Dictionary with spatial_dims, num_patches, and channels.
        """
        s_d, s_h, s_w = img_size[0] // patch_sizes_tuple[0], \
                        img_size[1] // patch_sizes_tuple[1], \
                        img_size[2] // patch_sizes_tuple[2]

        if stage_idx_in_swin_list == 0: # Patch embedding output
            channels = base_feature_size
        elif stage_idx_in_swin_list > 0 and stage_idx_in_swin_list <= 4 : # Swin stage outputs
            channels = base_feature_size * (2**(stage_idx_in_swin_list-1))
            if stage_idx_in_swin_list == 1: channels = base_feature_size
            elif stage_idx_in_swin_list == 2: channels = 2 * base_feature_size
            elif stage_idx_in_swin_list == 3: channels = 4 * base_feature_size
            elif stage_idx_in_swin_list == 4: channels = 8 * base_feature_size
            elif stage_idx_in_swin_list == 5: channels = 16 * base_feature_size

            if stage_idx_in_swin_list == 3:
                s_d, s_h, s_w = s_d // (2**3), s_h // (2**3), s_w // (2**3)
                channels = base_feature_size * 8
            elif stage_idx_in_swin_list == 4:
                s_d, s_h, s_w = s_d // (2**4), s_h // (2**4), s_w // (2**4)
                channels = base_feature_size * 16
            else:
                raise ValueError(f"Unsupported stage_idx_in_swin_list for reconstruction: {stage_idx_in_swin_list}. Only 3 or 4 supported.")

        else:
            raise ValueError(f"Invalid stage_idx_in_swin_list: {stage_idx_in_swin_list}")

        num_patches = s_d * s_h * s_w
        return {"spatial_dims": (s_d, s_h, s_w), "num_patches": num_patches, "channels": channels}

    def _initialize_recon_modules(
        self, stage_swin_list_idx: int, img_size_tuple: Tuple[int, ...],
        patch_sizes_tuple: Tuple[int, ...], base_feature_size: int,
        recon_embed_dim: int, recon_transformer_depth: int,
        recon_transformer_heads: int, device=None
    ) -> Dict[str, Any]:
        """
        Initialize modules for feature reconstruction at a given stage.

        Args:
            stage_swin_list_idx (int): Swin stage index.
            img_size_tuple (Tuple[int, ...]): Image size.
            patch_sizes_tuple (Tuple[int, ...]): Patch sizes.
            base_feature_size (int): Base feature size.
            recon_embed_dim (int): Embedding dimension for reconstruction.
            recon_transformer_depth (int): Depth of reconstruction transformer.
            recon_transformer_heads (int): Number of heads in reconstruction transformer.
            device: Device to use.

        Returns:
            Dict[str, Any]: Dictionary of reconstruction modules and parameters.
        """
        stage_info = self._calculate_stage_dims_channels(stage_swin_list_idx, img_size_tuple, patch_sizes_tuple, base_feature_size)
        num_patches = stage_info["num_patches"]
        input_channels = stage_info["channels"]
        spatial_dims_at_stage = stage_info["spatial_dims"]

        projector_to_embed = nn.Conv3d(input_channels, recon_embed_dim, kernel_size=1).to(device)
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, recon_embed_dim)).to(device)
        reconstructor = FeatureReconstructionTransformer(
            embed_dim=recon_embed_dim,
            depth=recon_transformer_depth,
            num_heads=recon_transformer_heads,
            mlp_ratio=4.0,
            num_patches=num_patches
        ).to(device)
        projector_from_embed = nn.Conv3d(recon_embed_dim, input_channels, kernel_size=1).to(device)

        return {
            "projector_to_embed": projector_to_embed,
            "pos_embed": pos_embed,
            "reconstructor": reconstructor,
            "projector_from_embed": projector_from_embed,
            "num_patches": num_patches,
            "spatial_dims": spatial_dims_at_stage,
            "input_channels": input_channels
        }

    def _get_recon_target_num_patches(self, img_size, patch_sizes):
        """
        Calculate the number of patches for the reconstruction target.

        Args:
            img_size: Image size.
            patch_sizes: Patch sizes.
            depths: Depths.

        Returns:
            int: Number of patches.
        """
        s_d, s_h, s_w = img_size[0]//patch_sizes[0], img_size[1]//patch_sizes[1], img_size[2]//patch_sizes[2]
        s_d, s_h, s_w = s_d // (2**3), s_h // (2**3), s_w // (2**3)
        self.recon_feature_map_spatial_dims = (s_d, s_h, s_w)
        return s_d * s_h * s_w

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
        device=None,
    ) -> None:
        """
        Initialize the Multimodal SwinUNETR model.
        """
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.feature_size = feature_size
        self.is_training = True
        self.recon_level = recon_level.lower()

        self.hs3_recon_config: Optional[Dict[str, Any]] = None
        self.hs4_recon_config: Optional[Dict[str, Any]] = None

        self.recon_embed_dim=256
        self.recon_transformer_depth=4
        self.recon_transformer_heads=8

        self.device = device

        if "hs3" in self.recon_level:
            self.hs3_recon_config = self._initialize_recon_modules(
                stage_swin_list_idx=3,
                img_size_tuple=img_size, patch_sizes_tuple=patch_sizes,
                base_feature_size=feature_size, recon_embed_dim=self.recon_embed_dim,
                recon_transformer_depth=self.recon_transformer_depth,
                recon_transformer_heads=self.recon_transformer_heads,
                device=self.device
            )

        if "hs4" in self.recon_level:
            self.hs4_recon_config = self._initialize_recon_modules(
                stage_swin_list_idx=4,
                img_size_tuple=img_size, patch_sizes_tuple=patch_sizes,
                base_feature_size=feature_size, recon_embed_dim=self.recon_embed_dim,
                recon_transformer_depth=self.recon_transformer_depth,
                recon_transformer_heads=self.recon_transformer_heads,
                device=self.device
            )

        self.swinViTs = nn.ModuleList([SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        ) for _ in range(4)])

        self.encoder1_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(4)])

        self.encoder2_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(4)])

        self.encoder3_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(4)])

        self.encoder4_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(4)])

        self.encoder10_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(4)])

        self.channel_reductions = nn.ModuleList([
            Convolution(spatial_dims=3, in_channels=4 * feature_size, out_channels=feature_size, strides=1, kernel_size=1, dropout=0.2),
            Convolution(spatial_dims=3, in_channels=4 * feature_size, out_channels=feature_size, strides=1, kernel_size=1, dropout=0.2),
            Convolution(spatial_dims=3, in_channels=8 * feature_size, out_channels=2 * feature_size, strides=1, kernel_size=1, dropout=0.2),
            Convolution(spatial_dims=3, in_channels=16 * feature_size, out_channels=4 * feature_size, strides=1, kernel_size=1, dropout=0.2),
            Convolution(spatial_dims=3, in_channels=32 * feature_size, out_channels=8 * feature_size, strides=1, kernel_size=1, dropout=0.2),
            Convolution(spatial_dims=3, in_channels=64 * feature_size, out_channels=16 * feature_size, strides=1, kernel_size=1, dropout=0.2)
        ])

        # Decoder part
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        # Separate decoders for each modality
        self.f_decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.f_decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.f_decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.f_decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.f_decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.f_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=3)

        self.t1c_decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1c_decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1c_decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1c_decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1c_decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1c_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=3)

        self.t1_decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1_decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1_decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1_decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1_decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t1_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=3)

        self.t2_decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t2_decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t2_decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t2_decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t2_decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.t2_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=3)

        # Deep supervision heads
        self.ds_head4 = DeepSupervisionHead(in_channels=16 * feature_size, out_channels=out_channels, scale_factor=32)
        self.ds_head3 = DeepSupervisionHead(in_channels=8 * feature_size, out_channels=out_channels, scale_factor=16)
        self.ds_head2 = DeepSupervisionHead(in_channels=4 * feature_size, out_channels=out_channels, scale_factor=8)
        self.ds_head1 = DeepSupervisionHead(in_channels=2 * feature_size, out_channels=out_channels, scale_factor=4)
        self.ds_head0 = DeepSupervisionHead(in_channels=feature_size, out_channels=out_channels, scale_factor=2)

    def _perform_reconstruction_for_stage(
        self,
        raw_features_current_stage: List[torch.Tensor],
        is_modality_present_list: List[bool],
        num_missing_modalities_val: int,
        recon_config: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """
        Perform feature reconstruction for a specific stage.

        Args:
            raw_features_current_stage (List[torch.Tensor]): Raw features for each modality.
            is_modality_present_list (List[bool]): List indicating which modalities are present.
            num_missing_modalities_val (int): Number of missing modalities.
            recon_config (Dict[str, Any]): Reconstruction config.

        Returns:
            List[torch.Tensor]: List of reconstructed features for each modality.
        """
        final_features_for_stage = list(raw_features_current_stage)
        total_modalities = len(raw_features_current_stage)

        if num_missing_modalities_val == 0 or recon_config is None:
            return final_features_for_stage

        available_transformer_inputs_stage = []
        for i in range(total_modalities):
            if is_modality_present_list[i]:
                feat_map = raw_features_current_stage[i]
                projected_map = recon_config["projector_to_embed"](feat_map)
                _B, _E, _Df, _Hf, _Wf = projected_map.shape
                if _Df*_Hf*_Wf != recon_config["num_patches"]:
                    raise ValueError(f"Spatial dim mismatch. Expected {recon_config['num_patches']} patches, got {_Df*_Hf*_Wf}")
                flat_feat = projected_map.flatten(2).transpose(1, 2)
                flat_feat = flat_feat + recon_config["pos_embed"]
                available_transformer_inputs_stage.append(flat_feat)

        reconstructed_flat_features_map = {}

        if available_transformer_inputs_stage:
            concatenated_available_flat = torch.cat(available_transformer_inputs_stage, dim=1)
            raw_reconstructed_output = recon_config["reconstructor"](
                concatenated_available_flat,
                num_missing_modalities_val,
                recon_config["num_patches"]
            )
            current_recon_batch_idx = 0
            for i in range(total_modalities):
                if not is_modality_present_list[i]:
                    start = current_recon_batch_idx * recon_config["num_patches"]
                    end = start + recon_config["num_patches"]
                    if end <= raw_reconstructed_output.shape[1]:
                        reconstructed_flat_features_map[i] = raw_reconstructed_output[:, start:end, :]
                    current_recon_batch_idx += 1

        for i in range(total_modalities):
            if not is_modality_present_list[i] and i in reconstructed_flat_features_map:
                flat_recon_embed = reconstructed_flat_features_map[i]
                _B, _NP, _E_recon = flat_recon_embed.shape
                _Df_recon, _Hf_recon, _Wf_recon = recon_config["spatial_dims"]
                unflat_recon_embed = flat_recon_embed.transpose(1, 2).reshape(
                    _B, _E_recon, _Df_recon, _Hf_recon, _Wf_recon
                )
                projected_back_recon = recon_config["projector_from_embed"](unflat_recon_embed)
                final_features_for_stage[i] = projected_back_recon
        return final_features_for_stage

    def forward(
        self, x_in: torch.Tensor, modalities_dropped_info: list | tuple | str | None = None
    ):
        """
        Forward pass of the model.

        Args:
            x_in (torch.Tensor): Input tensor of shape (B, C, ...).
            modalities_dropped_info (list | tuple | str | None): Info about dropped modalities.

        Returns:
            If self.is_training:
                Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: Main output, deep supervision outputs, separate decoder outputs.
            Else:
                torch.Tensor: Main output.
        """
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        total_modalities = 4
        is_modality_present = [True] * total_modalities
        num_missing_modalities = 0
        actual_modalities_dropped_indices = []

        if modalities_dropped_info is None or modalities_dropped_info == "no_drop" or \
           (isinstance(modalities_dropped_info, (list, tuple)) and len(modalities_dropped_info) == 0):
            num_missing_modalities = 0
        elif isinstance(modalities_dropped_info, (list, tuple)):
            actual_modalities_dropped_indices = list(modalities_dropped_info)
            for dropped_idx in actual_modalities_dropped_indices:
                if 0 <= dropped_idx < total_modalities:
                    is_modality_present[dropped_idx] = False
            num_missing_modalities = len(actual_modalities_dropped_indices)
        else:
            print(f"Warning: Unexpected format for modalities_dropped_info: {modalities_dropped_info}. Assuming all present.")
            num_missing_modalities = 0

        device = x_in.device

        # 1. Get all hidden states from SwinViTs
        hidden_states_out_all_modalities = [
            self.swinViTs[i](x_in[:, i:i+1, :, :], normalize=self.normalize)[0] for i in range(total_modalities)
        ]

        # 2. Extract raw features at different levels
        hs0_raw = [hs[0] for hs in hidden_states_out_all_modalities]
        hs1_raw = [hs[1] for hs in hidden_states_out_all_modalities]
        hs2_raw = [hs[2] for hs in hidden_states_out_all_modalities]
        hs3_features_raw = [hs[3] for hs in hidden_states_out_all_modalities]
        hs4_features_raw_for_enc10 = [hs[4] for hs in hidden_states_out_all_modalities]

        # 3. Perform reconstruction for HS3 level if configured
        current_hs3_features = hs3_features_raw
        if "hs3" in self.recon_level and self.hs3_recon_config is not None:
            current_hs3_features = self._perform_reconstruction_for_stage(
                raw_features_current_stage=hs3_features_raw,
                is_modality_present_list=is_modality_present,
                num_missing_modalities_val=num_missing_modalities,
                recon_config=self.hs3_recon_config,
            )
        hidden_states_combined_input = torch.cat(current_hs3_features, dim=1)
        hidden_states_combined = self.channel_reductions[4](hidden_states_combined_input)

        # 4. Perform reconstruction for HS4 level (inputs to encoder10_list) if configured
        current_hs4_inputs_for_enc10 = hs4_features_raw_for_enc10
        if "hs4" in self.recon_level and self.hs4_recon_config is not None:
            current_hs4_inputs_for_enc10 = self._perform_reconstruction_for_stage(
                raw_features_current_stage=hs4_features_raw_for_enc10,
                is_modality_present_list=is_modality_present,
                num_missing_modalities_val=num_missing_modalities,
                recon_config=self.hs4_recon_config,
            )

        # 5. Process through UnetrBasicBlocks (original encoder path)
        enc0_1 = self.encoder1_list[0](x_in[:,0:1,:,:])
        enc1_1 = self.encoder2_list[0](hs0_raw[0])
        enc2_1 = self.encoder3_list[0](hs1_raw[0])
        enc3_1 = self.encoder4_list[0](hs2_raw[0])
        dec4_1 = self.encoder10_list[0](current_hs4_inputs_for_enc10[0])

        enc0_2 = self.encoder1_list[1](x_in[:,1:2,:,:])
        enc1_2 = self.encoder2_list[1](hs0_raw[1])
        enc2_2 = self.encoder3_list[1](hs1_raw[1])
        enc3_2 = self.encoder4_list[1](hs2_raw[1])
        dec4_2 = self.encoder10_list[1](current_hs4_inputs_for_enc10[1])

        enc0_3 = self.encoder1_list[2](x_in[:,2:3,:,:])
        enc1_3 = self.encoder2_list[2](hs0_raw[2])
        enc2_3 = self.encoder3_list[2](hs1_raw[2])
        enc3_3 = self.encoder4_list[2](hs2_raw[2])
        dec4_3 = self.encoder10_list[2](current_hs4_inputs_for_enc10[2])

        enc0_4 = self.encoder1_list[3](x_in[:,3:4,:,:])
        enc1_4 = self.encoder2_list[3](hs0_raw[3])
        enc2_4 = self.encoder3_list[3](hs1_raw[3])
        enc3_4 = self.encoder4_list[3](hs2_raw[3])
        dec4_4 = self.encoder10_list[3](current_hs4_inputs_for_enc10[3])

        # 6. Channel reductions for skip connections
        enc0 = self.channel_reductions[0](torch.cat([enc0_1, enc0_2, enc0_3, enc0_4], dim=1))
        enc1 = self.channel_reductions[1](torch.cat([enc1_1, enc1_2, enc1_3, enc1_4], dim=1))
        enc2 = self.channel_reductions[2](torch.cat([enc2_1, enc2_2, enc2_3, enc2_4], dim=1))
        enc3 = self.channel_reductions[3](torch.cat([enc3_1, enc3_2, enc3_3, enc3_4], dim=1))
        dec4 = self.channel_reductions[5](torch.cat([dec4_1, dec4_2, dec4_3, dec4_4], dim=1))

        # 7. Decoder
        dec3 = self.decoder5(dec4, hidden_states_combined)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        # 8. Training specific outputs (deep supervision, separate decoders)
        if self.is_training:
            sep_dec_outputs = []
            f_dec3 = self.f_decoder5(dec4_1, hs3_features_raw[0])
            f_dec2 = self.f_decoder4(f_dec3, enc3_1)
            f_dec1 = self.f_decoder3(f_dec2, enc2_1)
            f_dec0 = self.f_decoder2(f_dec1, enc1_1)
            f_out_final = self.f_decoder1(f_dec0, enc0_1)
            sep_dec_outputs.append(self.f_out(f_out_final))

            t1c_dec3 = self.t1c_decoder5(dec4_2, hs3_features_raw[1])
            t1c_dec2 = self.t1c_decoder4(t1c_dec3, enc3_2)
            t1c_dec1 = self.t1c_decoder3(t1c_dec2, enc2_2)
            t1c_dec0 = self.t1c_decoder2(t1c_dec1, enc1_2)
            t1c_out_final = self.t1c_decoder1(t1c_dec0, enc0_2)
            sep_dec_outputs.append(self.t1c_out(t1c_out_final))

            t1_dec3 = self.t1_decoder5(dec4_3, hs3_features_raw[2])
            t1_dec2 = self.t1_decoder4(t1_dec3, enc3_3)
            t1_dec1 = self.t1_decoder3(t1_dec2, enc2_3)
            t1_dec0 = self.t1_decoder2(t1_dec1, enc1_3)
            t1_out_final = self.t1_decoder1(t1_dec0, enc0_3)
            sep_dec_outputs.append(self.t1_out(t1_out_final))

            t2_dec3 = self.t2_decoder5(dec4_4, hs3_features_raw[3])
            t2_dec2 = self.t2_decoder4(t2_dec3, enc3_4)
            t2_dec1 = self.t2_decoder3(t2_dec2, enc2_4)
            t2_dec0 = self.t2_decoder2(t2_dec1, enc1_4)
            t2_out_final = self.t2_decoder1(t2_dec0, enc0_4)
            sep_dec_outputs.append(self.t2_out(t2_out_final))

            ds_outputs = []
            ds_outputs.append(self.ds_head0(dec0))
            ds_outputs.append(self.ds_head1(dec1))
            ds_outputs.append(self.ds_head2(dec2))
            ds_outputs.append(self.ds_head3(dec3))
            ds_outputs.append(self.ds_head4(dec4))

            return logits, ds_outputs, sep_dec_outputs

        else:
            return logits

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        """
        Check if the input spatial shape is valid for the patch size.

        Args:
            spatial_shape: Spatial shape of the input.
        """
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

# =========================
# Feature Reconstruction Transformer
# =========================

class FeatureReconstructionTransformer(nn.Module):
    """
    Transformer for reconstructing missing modality features.

    Args:
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP ratio.
        num_patches (int): Number of patches per modality.
    """
    def __init__(
        self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4., num_patches: Optional[int] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_per_modality = num_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        if self.num_patches_per_modality is not None:
            self.missing_tokens = nn.Parameter(torch.randn(1, self.num_patches_per_modality, embed_dim))
        else:
            self.missing_tokens = None
            print("Warning: num_patches_per_modality for FeatureReconstructionTransformer not specified. Missing tokens fallback to zeros.")

    def forward(
        self, available_features_cat: torch.Tensor, num_missing_modalities: int, target_num_patches_per_modality: int
    ) -> torch.Tensor:
        """
        Forward pass for reconstructing features.

        Args:
            available_features_cat (torch.Tensor): Concatenated available features.
            num_missing_modalities (int): Number of missing modalities.
            target_num_patches_per_modality (int): Number of patches per missing modality.

        Returns:
            torch.Tensor: Reconstructed features.
        """
        B, S_avail, E = available_features_cat.shape

        if num_missing_modalities == 0:
            return torch.empty(B, 0, E, device=available_features_cat.device)

        if self.missing_tokens is not None and self.missing_tokens.shape[1] == target_num_patches_per_modality:
            single_mod_query_tokens = self.missing_tokens.expand(B, -1, -1)
            queries_for_all_missing = single_mod_query_tokens.repeat_interleave(num_missing_modalities, dim=1)
        else:
            print(f"Warning: Using zero tokens as queries for reconstruction. target_num_patches_per_modality={target_num_patches_per_modality}")
            queries_for_all_missing = torch.zeros(B, num_missing_modalities * target_num_patches_per_modality, E, device=available_features_cat.device)

        transformer_input = torch.cat([available_features_cat, queries_for_all_missing], dim=1)
        transformer_output = self.transformer_encoder(transformer_input)
        reconstructed_features = transformer_output[:, S_avail:]
        return reconstructed_features

