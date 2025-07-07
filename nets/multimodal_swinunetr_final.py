from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Final
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, Convolution
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from utils.mm_network_utils import (window_partition,window_reverse,
                                    WindowAttention,SwinTransformerBlock,PatchMergingV2,PatchMerging,
                                    SwinTransformer,MM_Transformer,FeatureFusionModule,DeepSupervisionHead, DecUpsampleBlock)
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

class Multimodal_SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    patch_size: Final[int] = 2


    def _get_recon_target_num_patches(self, img_size, patch_sizes, depths):
        # Calculate spatial dimensions after Swin Stage 3
        # SwinTransformer downsamples by patch_size initially, then by 2 at each of the first 3 merging layers
        s_d, s_h, s_w = img_size[0]//patch_sizes[0], img_size[1]//patch_sizes[1], img_size[2]//patch_sizes[2]
        # After 3 patch merging stages (depths[0], depths[1], depths[2])
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
        
    ) -> None:
        
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

        self.recon_embed_dim=256
        self.recon_transformer_depth=4
        self.recon_transformer_heads=8

        self.target_num_patches_for_recon = self._get_recon_target_num_patches(img_size, patch_sizes, depths)

        recon_input_feature_channels = 8 * self.feature_size
        self.recon_feature_projector_to_embed = nn.Conv3d(
            recon_input_feature_channels, self.recon_embed_dim, kernel_size=1
        )
        self.recon_transformer_pos_embed = nn.Parameter(
            torch.zeros(1, self.target_num_patches_for_recon, self.recon_embed_dim)
        )
        self.feature_reconstructor = FeatureReconstructionTransformer( 
            embed_dim=self.recon_embed_dim,
            depth=self.recon_transformer_depth,
            num_heads=self.recon_transformer_heads,
            mlp_ratio=4.0, 
            num_patches=self.target_num_patches_for_recon 
        )
        
        self.recon_feature_projector_from_embed = nn.Conv3d(
            self.recon_embed_dim, recon_input_feature_channels, kernel_size=1
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
                                        Convolution(spatial_dims=3,
                                            in_channels=4 * feature_size,
                                            out_channels=feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3,
                                            in_channels=4 * feature_size,
                                            out_channels=feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3,
                                            in_channels=8 * feature_size,
                                            out_channels=2 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3,
                                            in_channels=16 * feature_size,
                                            out_channels=4 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3,
                                            in_channels=32 * feature_size,
                                            out_channels=8 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3,
                                            in_channels=64 * feature_size,
                                            out_channels=16 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2)
            ])   

        # decoder part
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

    def forward(self, x_in: torch.Tensor, modalities_dropped_info: list | tuple | str | None = None):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        # d_m_str: e.g., "1011" means modality 1 (0-indexed) is missing.
        is_modality_present = [True] * 4
        num_missing_modalities = 0
        
        # Ensure modalities_dropped_info is processed consistently
        actual_modalities_dropped_indices = []

        if modalities_dropped_info is None or modalities_dropped_info == "no_drop":
            num_missing_modalities = 0
        elif isinstance(modalities_dropped_info, (list, tuple)) and len(modalities_dropped_info) > 0:
            actual_modalities_dropped_indices = list(modalities_dropped_info)
            for dropped_idx in actual_modalities_dropped_indices:
                if 0 <= dropped_idx < 4:
                    is_modality_present[dropped_idx] = False
            num_missing_modalities = len(actual_modalities_dropped_indices)
        elif isinstance(modalities_dropped_info, (list, tuple)) and len(modalities_dropped_info) == 0: # Empty list/tuple means no drop
            num_missing_modalities = 0
        else:
            # This case should ideally not happen if rand_drop_channel_new is used consistently.
            # However, for robustness during inference if an unexpected format is passed:
            print(f"Warning: Unexpected format for modalities_dropped_info: {modalities_dropped_info}. Assuming all modalities present.")
            num_missing_modalities = 0
            
        hidden_states_out_modalities = [
            self.swinViTs[i](x_in[:, i:i+1, :, :], normalize=self.normalize)[0] for i in range(4)
        ]

        features_at_recon_level_raw = [hs[3] for hs in hidden_states_out_modalities]
        final_features_at_recon_level = [None] * 4

        if num_missing_modalities > 0 :
            available_transformer_inputs = []
            
            for i in range(4):

                if is_modality_present[i]:
                    feat_map = features_at_recon_level_raw[i]
                    projected_map = self.recon_feature_projector_to_embed(feat_map) # (B, recon_embed_dim, Df, Hf, Wf)
                    
                    # Validate spatial dimensions
                    _B, _E, _Df, _Hf, _Wf = projected_map.shape
                    if _Df*_Hf*_Wf != self.target_num_patches_for_recon:
                        raise ValueError(f"Spatial dim mismatch for recon transformer. Expected {self.target_num_patches_for_recon} patches, got {_Df*_Hf*_Wf}")

                    flat_feat = projected_map.flatten(2).transpose(1, 2) # (B, NumPatches, recon_embed_dim)
                    flat_feat = flat_feat + self.recon_transformer_pos_embed
                    available_transformer_inputs.append(flat_feat)
                    
                    # Store the original (unprojected) feature for available modalities
                    final_features_at_recon_level[i] = features_at_recon_level_raw[i]

            reconstructed_features_flat_embed_dim = {} # Dict: {missing_idx: flat_reconstructed_feature}

            if available_transformer_inputs: # If there are features to guide reconstruction
                concatenated_available_flat = torch.cat(available_transformer_inputs, dim=1)
                
                raw_reconstructed_output = self.feature_reconstructor(
                    concatenated_available_flat,
                    num_missing_modalities,
                    self.target_num_patches_for_recon
                ) # (B, num_missing * num_patches, recon_embed_dim)

                current_recon_batch_idx = 0
                for i in range(4):
                    if not is_modality_present[i]: # If modality 'i' is missing
                        start = current_recon_batch_idx * self.target_num_patches_for_recon
                        end = start + self.target_num_patches_for_recon
                        reconstructed_features_flat_embed_dim[i] = raw_reconstructed_output[:, start:end, :]
                        current_recon_batch_idx += 1
            
            # Populate final_features_at_recon_level with reconstructed features
            for i in range(4):
                if not is_modality_present[i]: # If modality was missing
                    if i in reconstructed_features_flat_embed_dim:
                        flat_recon_embed = reconstructed_features_flat_embed_dim[i] # (B, NumPatches, recon_embed_dim)
                        _B, _NP, _E_recon = flat_recon_embed.shape
                        _Df_recon, _Hf_recon, _Wf_recon = self.recon_feature_map_spatial_dims
                        
                        unflat_recon_embed = flat_recon_embed.transpose(1, 2).reshape(
                            _B, _E_recon, _Df_recon, _Hf_recon, _Wf_recon
                        )
                        # Project back to original channel dimension (8 * feature_size)
                        projected_back_recon = self.recon_feature_projector_from_embed(unflat_recon_embed)
                        final_features_at_recon_level[i] = projected_back_recon
                    else:
                        # Missing but not reconstructed (e.g., no available features to guide)
                        # Use the original one from mean-filled input
                        final_features_at_recon_level[i] = features_at_recon_level_raw[i]
        else: # All modalities are present, no reconstruction needed
            final_features_at_recon_level = features_at_recon_level_raw

    
        enc0_1 = self.encoder1_list[0](x_in[:,0:1,:,:]) 
        enc1_1 = self.encoder2_list[0](hidden_states_out_modalities[0][0]) 
        enc2_1 = self.encoder3_list[0](hidden_states_out_modalities[0][1]) 
        enc3_1 = self.encoder4_list[0](hidden_states_out_modalities[0][2]) 
        dec4_1 = self.encoder10_list[0](hidden_states_out_modalities[0][4]) # Bottleneck Swin features

        enc0_2 = self.encoder1_list[1](x_in[:,1:2,:,:])
        enc1_2 = self.encoder2_list[1](hidden_states_out_modalities[1][0])
        enc2_2 = self.encoder3_list[1](hidden_states_out_modalities[1][1])
        enc3_2 = self.encoder4_list[1](hidden_states_out_modalities[1][2])
        dec4_2 = self.encoder10_list[1](hidden_states_out_modalities[1][4])

        enc0_3 = self.encoder1_list[2](x_in[:,2:3,:,:])
        enc1_3 = self.encoder2_list[2](hidden_states_out_modalities[2][0])
        enc2_3 = self.encoder3_list[2](hidden_states_out_modalities[2][1])
        enc3_3 = self.encoder4_list[2](hidden_states_out_modalities[2][2])
        dec4_3 = self.encoder10_list[2](hidden_states_out_modalities[2][4])

        enc0_4 = self.encoder1_list[3](x_in[:,3:4,:,:])
        enc1_4 = self.encoder2_list[3](hidden_states_out_modalities[3][0])
        enc2_4 = self.encoder3_list[3](hidden_states_out_modalities[3][1])
        enc3_4 = self.encoder4_list[3](hidden_states_out_modalities[3][2])
        dec4_4 = self.encoder10_list[3](hidden_states_out_modalities[3][4])

        # Channel reductions for skip connections (these use features from potentially mean-filled inputs)
        enc0 = self.channel_reductions[0](torch.cat([enc0_1, enc0_2, enc0_3, enc0_4], dim=1))
        enc1 = self.channel_reductions[1](torch.cat([enc1_1, enc1_2, enc1_3, enc1_4], dim=1))
        enc2 = self.channel_reductions[2](torch.cat([enc2_1, enc2_2, enc2_3, enc2_4], dim=1))
        enc3 = self.channel_reductions[3](torch.cat([enc3_1, enc3_2, enc3_3, enc3_4], dim=1))
        hidden_states_combined_input = torch.cat(final_features_at_recon_level, dim=1) # (B, 4 * 8*fs, Df, Hf, Wf)
        hidden_states_combined = self.channel_reductions[4](hidden_states_combined_input) # Expects 32*fs -> 8*fs

        dec4 = self.channel_reductions[5](torch.cat([dec4_1, dec4_2, dec4_3, dec4_4], dim=1))

        # decoding for all starts here
        dec3 = self.decoder5(dec4, hidden_states_combined) # [1, 192, 8, 8, 8]
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        if self.is_training:
            f_dec3 = self.f_decoder5(dec4_1, features_at_recon_level_raw[0])
            f_dec2 = self.f_decoder4(f_dec3, enc3_1)
            f_dec1 = self.f_decoder3(f_dec2, enc2_1)
            f_dec0 = self.f_decoder2(f_dec1, enc1_1)
            f_out = self.f_decoder1(f_dec0, enc0_1)
            f_logits = self.f_out(f_out)

            t1c_dec3 = self.t1c_decoder5(dec4_2, features_at_recon_level_raw[1])
            t1c_dec2 = self.t1c_decoder4(t1c_dec3, enc3_2)
            t1c_dec1 = self.t1c_decoder3(t1c_dec2, enc2_2)
            t1c_dec0 = self.t1c_decoder2(t1c_dec1, enc1_2)
            t1c_out = self.t1c_decoder1(t1c_dec0, enc0_2)
            t1c_logits = self.t1c_out(t1c_out)

            t1_dec3 = self.t1_decoder5(dec4_3, features_at_recon_level_raw[2])
            t1_dec2 = self.t1_decoder4(t1_dec3, enc3_3)
            t1_dec1 = self.t1_decoder3(t1_dec2, enc2_3)
            t1_dec0 = self.t1_decoder2(t1_dec1, enc1_3)
            t1_out = self.t1_decoder1(t1_dec0, enc0_3)
            t1_logits = self.t1_out(t1_out)

            t2_dec3 = self.t2_decoder5(dec4_4, features_at_recon_level_raw[3])
            t2_dec2 = self.t2_decoder4(t2_dec3, enc3_4)
            t2_dec1 = self.t2_decoder3(t2_dec2, enc2_4)
            t2_dec0 = self.t2_decoder2(t2_dec1, enc1_4)
            t2_out = self.t2_decoder1(t2_dec0, enc0_4)
            t2_logits = self.t2_out(t2_out)

            ds_output4 = self.ds_head4(dec4)
            ds_output3 = self.ds_head3(dec3)
            ds_output2 = self.ds_head2(dec2)
            ds_output1 = self.ds_head1(dec1)
            ds_output0 = self.ds_head0(dec0)
    
            return logits, [ds_output0, ds_output1, ds_output2, ds_output3, ds_output4], [f_logits, t1c_logits, t1_logits, t2_logits]
        
        else:
            return logits

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )



# --- Feature Reconstruction Transformer ---
class FeatureReconstructionTransformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4., num_patches=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, # Standard dropout
            activation='relu',
            batch_first=True # Expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        if self.num_patches is not None:
            self.missing_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        else:
            # If num_patches is not known at init, this will need to be handled dynamically or a fixed max chosen
            print("Warning: num_patches for FeatureReconstructionTransformer not specified. Missing tokens might not work as expected.")
            self.missing_tokens = None


    def forward(self, available_features_cat: torch.Tensor, num_missing_modalities: int, target_num_patches: int):
    
        B, _, E = available_features_cat.shape

        if num_missing_modalities == 0:
            return torch.empty(B, 0, E, device=available_features_cat.device) # No features to reconstruct

        if self.missing_tokens is not None and self.missing_tokens.shape[1] == target_num_patches:
            query_tokens = self.missing_tokens.expand(B, -1, -1) # (B, target_num_patches, E)
            queries_for_all_missing = query_tokens.repeat_interleave(num_missing_modalities, dim=1) # (B, num_missing_modalities * target_num_patches, E)
        else:
            # Fallback: create zero tokens if not properly initialized or if num_patches mismatch.
            # This part needs more info from the paper for a precise MAE-like mechanism.
            print(f"Warning: Using zero tokens as queries for reconstruction. target_num_patches={target_num_patches}")
            queries_for_all_missing = torch.zeros(B, num_missing_modalities * target_num_patches, E, device=available_features_cat.device)
        transformer_input = torch.cat([available_features_cat, queries_for_all_missing], dim=1)
        transformer_output = self.transformer_encoder(transformer_input)

        reconstructed_features = transformer_output[:, available_features_cat.shape[1]:]

        return reconstructed_features


