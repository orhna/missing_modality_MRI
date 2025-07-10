from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Final, Dict, Any, Optional, List, Tuple # Added some types
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, Convolution
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from utils.mm_network_utils import (window_partition,window_reverse,
                                    WindowAttention,SwinTransformerBlock,PatchMergingV2,PatchMerging,
                                    SwinTransformer,DeepSupervisionHead)
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

class FeatureMemory(nn.Module):
    """
    A memory module that stores and retrieves feature prototypes for different modalities.
    It uses a queue-based memory bank for each modality. When a modality is missing,
    it uses an available modality's features to query the memory (based on cosine similarity)
    and retrieve a corresponding feature prototype for the missing modality.
    """
    def __init__(self, num_modalities, feature_dim, memory_size):
        super().__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        self.memory_size = memory_size

        for i in range(num_modalities):
            self.register_buffer(f"memory_{i}", torch.randn(memory_size, feature_dim))
            self.register_buffer(f"ptr_{i}", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_memory(self, prototypes: torch.Tensor, mod_idx: int):
        """ Update memory bank for a given modality with new prototypes. """
        batch_size = prototypes.shape[0]
        ptr = int(getattr(self, f"ptr_{mod_idx}")[0])
        memory_bank = getattr(self, f"memory_{mod_idx}")

        indices = torch.arange(ptr, ptr + batch_size).to(prototypes.device) % self.memory_size
        memory_bank[indices] = F.normalize(prototypes, dim=1)
        
        ptr = (ptr + batch_size) % self.memory_size
        getattr(self, f"ptr_{mod_idx}")[0] = ptr

    def _get_prototype_from_tokens(self, tokens: torch.Tensor):
        """ Get a single prototype vector from a batch of tokens by averaging. """
        return tokens.mean(dim=1)

    @torch.no_grad()
    def retrieve(self, query_tokens: torch.Tensor, query_mod_idx: int, missing_mod_idx: int):
        """ Retrieve a feature prototype for a missing modality. """
        query_prototype = self._get_prototype_from_tokens(query_tokens.detach())
        query_prototype = F.normalize(query_prototype, dim=1)
        
        memory_to_query = getattr(self, f"memory_{query_mod_idx}")
        memory_to_value = getattr(self, f"memory_{missing_mod_idx}")

        attn = torch.einsum('bd,md->bm', query_prototype, memory_to_query)
        attn = F.softmax(attn, dim=1)

        retrieved_prototype = torch.einsum('bm,md->bd', attn, memory_to_value)
        return retrieved_prototype


class Multimodal_SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    patch_size: Final[int] = 2

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        num_modalities: int = 2, # MODIFIED: Added num_modalities parameter, default to 2
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
        use_memory: bool = False, # MODIFIED: Add use_memory flag
        memory_size: int = 1024, # MODIFIED: Add memory_size
        memory_feature_dim: int = 128, # MODIFIED: Add memory_feature_dim
        device = None,
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
        self.num_modalities = num_modalities # MODIFIED: Store num_modalities
        self.use_memory = use_memory

        self.hs3_memory_params: Optional[Dict[str, Any]] = None
        self.hs4_memory_params: Optional[Dict[str, Any]] = None

        self.recon_embed_dim=256
        self.recon_transformer_depth=4
        self.recon_transformer_heads=8

        self.device = device

        if self.use_memory:
            self.hs3_memory_params = self._initialize_memory_modules(
                stage_idx=3, img_size_tuple=img_size, patch_sizes_tuple=patch_sizes,
                base_feature_size=feature_size, memory_feature_dim=memory_feature_dim,
                memory_size=memory_size, num_modalities=num_modalities
            )
            self.hs4_memory_params = self._initialize_memory_modules(
                stage_idx=4, img_size_tuple=img_size, patch_sizes_tuple=patch_sizes,
                base_feature_size=feature_size, memory_feature_dim=memory_feature_dim,
                memory_size=memory_size, num_modalities=num_modalities
            )
        self.swinViTs = nn.ModuleList([SwinTransformer(
            in_chans=in_channels, # This is 1 as each SwinViT processes one modality
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
        ) for _ in range(self.num_modalities)]) # MODIFIED: Use self.num_modalities
        
        self.encoder1_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels, # This is 1 as it's the input to the first UnetrBasicBlock for each modality
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(self.num_modalities)]) # MODIFIED: Use self.num_modalities
        
        self.encoder2_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(self.num_modalities)]) # MODIFIED: Use self.num_modalities
        
        self.encoder3_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(self.num_modalities)]) # MODIFIED: Use self.num_modalities
        
        self.encoder4_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(self.num_modalities)]) # MODIFIED: Use self.num_modalities

        self.encoder10_list = nn.ModuleList([UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True) for _ in range(self.num_modalities)]) # MODIFIED: Use self.num_modalities

        self.channel_reductions = nn.ModuleList([
                                        Convolution(spatial_dims=3, # enc0
                                            in_channels=self.num_modalities * feature_size, # MODIFIED
                                            out_channels=feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3, # enc1
                                            in_channels=self.num_modalities * feature_size, # MODIFIED
                                            out_channels=feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3, # enc2
                                            in_channels=self.num_modalities * 2 * feature_size, # MODIFIED
                                            out_channels=2 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3, # enc3
                                            in_channels=self.num_modalities * 4 * feature_size, # MODIFIED
                                            out_channels=4 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3, # hidden_states_combined (from hs3)
                                            in_channels=self.num_modalities * 8 * feature_size, # MODIFIED
                                            out_channels=8 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2),
                                        Convolution(spatial_dims=3, # dec4 (bottleneck output combined)
                                            in_channels=self.num_modalities * 16 * feature_size, # MODIFIED
                                            out_channels=16 * feature_size,
                                            strides=1,
                                            kernel_size=1,
                                            dropout=0.2)
            ])   
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
        self.t1c_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=3) # Assuming output is 3 channels for segmentation task
        
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
        self.t1_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=3) # Assuming output is 3 channels
        
        # Deep supervision heads
        self.ds_head4 = DeepSupervisionHead(in_channels=16 * feature_size, out_channels=out_channels, scale_factor=32)
        self.ds_head3 = DeepSupervisionHead(in_channels=8 * feature_size, out_channels=out_channels, scale_factor=16)
        self.ds_head2 = DeepSupervisionHead(in_channels=4 * feature_size, out_channels=out_channels, scale_factor=8)
        self.ds_head1 = DeepSupervisionHead(in_channels=2 * feature_size, out_channels=out_channels, scale_factor=4)
        self.ds_head0 = DeepSupervisionHead(in_channels=feature_size, out_channels=out_channels, scale_factor=2)
    
    def _get_stage_spatial_dims(self, swin_list_idx: int, img_size: Tuple[int, ...], patch_sizes: Tuple[int, ...]):
        """Calculates spatial dimensions for a given Swin stage output."""
        s_d, s_h, s_w = tuple(i // p for i, p in zip(img_size, patch_sizes))
        if swin_list_idx > 0: # Downsampling happens after each swin layer
            # The output of swin layer i (hs[i]) is downsampled i times by a factor of 2 from patch-embedded resolution.
            downscale_factor = 2 ** swin_list_idx
            s_d, s_h, s_w = s_d // downscale_factor, s_h // downscale_factor, s_w // downscale_factor
        return (s_d, s_h, s_w)
    
    def _initialize_memory_modules(self, stage_idx: int, img_size_tuple: Tuple[int, ...], 
                                  patch_sizes_tuple: Tuple[int, ...], base_feature_size: int,
                                  memory_feature_dim: int, memory_size: int, num_modalities: int) -> Dict[str, Any]:
        """Initializes memory and projection modules for a given stage."""
        input_channels = base_feature_size * (2 ** stage_idx)
        spatial_dims = self._get_stage_spatial_dims(stage_idx, img_size_tuple, patch_sizes_tuple)
        
        return {
            "memory_module": FeatureMemory(num_modalities, memory_feature_dim, memory_size).to(self.device),
            "project_to_embed": nn.Conv3d(input_channels, memory_feature_dim, kernel_size=1).to(self.device),
            "project_from_embed": nn.Conv3d(memory_feature_dim, input_channels, kernel_size=1).to(self.device),
            "spatial_dims": spatial_dims,
            "num_patches": np.prod(spatial_dims)
        }

    def _apply_memory_reconstruction(self, 
                                     raw_features: List[torch.Tensor], 
                                     is_modality_present: List[bool], 
                                     num_missing: int,
                                     memory_params: Dict[str, Any]) -> List[torch.Tensor]:
        """Performs memory update and retrieval for a given stage."""

        # 1. Project all raw features to common embedding dim and get tokens
        tokens_raw = []
        for i in range(self.num_modalities):
            projected_map = memory_params["project_to_embed"](raw_features[i])
            tokens = projected_map.flatten(2).transpose(1, 2)
            tokens_raw.append(tokens)

        # 2. Update memory if training
        if self.is_training:
            for i in range(self.num_modalities):
                if is_modality_present[i]:
                    prototype = memory_params["memory_module"]._get_prototype_from_tokens(tokens_raw[i].detach())
                    memory_params["memory_module"]._update_memory(prototype, i)

        final_tokens = list(tokens_raw)
        # 3. Retrieve for missing modalities
        if num_missing > 0 and self.num_modalities == 2:
            present_idx = is_modality_present.index(True)
            missing_idx = is_modality_present.index(False)
            
            retrieved_prototype = memory_params["memory_module"].retrieve(tokens_raw[present_idx], present_idx, missing_idx)
            reconstructed = retrieved_prototype.unsqueeze(1).expand(-1, int(memory_params["num_patches"]), -1)
            final_tokens[missing_idx] = reconstructed

        # 4. Project tokens back to feature maps
        final_features = []
        _B, _E = final_tokens[0].shape[0], final_tokens[0].shape[2]
        _D, _H, _W = memory_params["spatial_dims"]
        for i in range(self.num_modalities):
            map_embedded = final_tokens[i].transpose(1, 2).reshape(_B, _E, _D, _H, _W)
            map_final = memory_params["project_from_embed"](map_embedded)
            final_features.append(map_final)
            
        return final_features
    
    def forward(self, x_in: torch.Tensor, modalities_dropped_info: list | tuple | str | None = None):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        # MODIFIED: Use self.num_modalities
        total_modalities = self.num_modalities 
        is_modality_present = [True] * total_modalities
        num_missing_modalities = 0
        
        actual_modalities_dropped_indices = []

        if modalities_dropped_info is None or modalities_dropped_info == "no_drop" or \
           (isinstance(modalities_dropped_info, (list, tuple)) and len(modalities_dropped_info) == 0) :
            num_missing_modalities = 0
        elif isinstance(modalities_dropped_info, (list, tuple)): # Check if all passed indices are valid for current num_modalities
            valid_dropped_indices = [idx for idx in modalities_dropped_info if 0 <= idx < total_modalities]
            if len(valid_dropped_indices) != len(modalities_dropped_info):
                 print(f"Warning: Some dropped indices {modalities_dropped_info} are out of range for {total_modalities} modalities. Using valid ones: {valid_dropped_indices}")
            actual_modalities_dropped_indices = list(valid_dropped_indices)
            for dropped_idx in actual_modalities_dropped_indices:
                is_modality_present[dropped_idx] = False
            num_missing_modalities = len(actual_modalities_dropped_indices)
        elif isinstance(modalities_dropped_info, str) and self.num_modalities == len(modalities_dropped_info): # Added support for string based drop info
            num_missing_modalities = 0
            for i, char_present in enumerate(modalities_dropped_info):
                if char_present == '0': # '0' means missing
                    is_modality_present[i] = False
                    actual_modalities_dropped_indices.append(i)
                    num_missing_modalities +=1
                elif char_present != '1': # '1' means present
                    print(f"Warning: Invalid character '{char_present}' in modalities_dropped_info string '{modalities_dropped_info}'. Assuming present.")
        else:
            print(f"Warning: Unexpected format or length mismatch for modalities_dropped_info: {modalities_dropped_info} with {self.num_modalities} modalities. Assuming all present.")
            num_missing_modalities = 0

        hidden_states_out_all_modalities = [
            self.swinViTs[i](x_in[:, i:i+1, :, :], normalize=self.normalize)[0] for i in range(self.num_modalities)
        ]

        hs0_raw = [hs[0] for hs in hidden_states_out_all_modalities] 
        hs1_raw = [hs[1] for hs in hidden_states_out_all_modalities] 
        hs2_raw = [hs[2] for hs in hidden_states_out_all_modalities] 
        hs3_raw = [hs[3] for hs in hidden_states_out_all_modalities] 
        hs4_raw_for_enc10 = [hs[4] for hs in hidden_states_out_all_modalities]

        current_hs3_features = hs3_raw
        if self.use_memory and num_missing_modalities > 0 and self.hs3_memory_params:
            current_hs3_features = self._apply_memory_reconstruction(
                hs3_raw, is_modality_present, num_missing_modalities, self.hs3_memory_params
            )

        hidden_states_combined_input = torch.cat(current_hs3_features, dim=1)
        hidden_states_combined = self.channel_reductions[4](hidden_states_combined_input)

        current_hs4_inputs_for_enc10 = hs4_raw_for_enc10
        if self.use_memory and num_missing_modalities > 0 and self.hs4_memory_params:
            current_hs4_inputs_for_enc10 = self._apply_memory_reconstruction(
                hs4_raw_for_enc10, is_modality_present, num_missing_modalities, self.hs4_memory_params
            )

        enc0_outputs = [self.encoder1_list[i](x_in[:, i:i+1, :, :]) for i in range(self.num_modalities)]
        enc1_outputs = [self.encoder2_list[i](hs0_raw[i]) for i in range(self.num_modalities)]
        enc2_outputs = [self.encoder3_list[i](hs1_raw[i]) for i in range(self.num_modalities)]
        enc3_outputs = [self.encoder4_list[i](hs2_raw[i]) for i in range(self.num_modalities)]
        dec4_modality_specific_outputs = [self.encoder10_list[i](current_hs4_inputs_for_enc10[i]) for i in range(self.num_modalities)]

        enc0 = self.channel_reductions[0](torch.cat(enc0_outputs, dim=1))
        enc1 = self.channel_reductions[1](torch.cat(enc1_outputs, dim=1))
        enc2 = self.channel_reductions[2](torch.cat(enc2_outputs, dim=1))
        enc3 = self.channel_reductions[3](torch.cat(enc3_outputs, dim=1))
        dec4 = self.channel_reductions[5](torch.cat(dec4_modality_specific_outputs, dim=1))

        dec3 = self.decoder5(dec4, hidden_states_combined) 
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        if self.is_training:
            sep_dec_outputs = []            
            t1c_dec3 = self.t1c_decoder5(dec4_modality_specific_outputs[0], current_hs3_features[0]) 
            t1c_dec2 = self.t1c_decoder4(t1c_dec3, enc3_outputs[0]) 
            t1c_dec1 = self.t1c_decoder3(t1c_dec2, enc2_outputs[0])
            t1c_dec0 = self.t1c_decoder2(t1c_dec1, enc1_outputs[0])
            t1c_out_final = self.t1c_decoder1(t1c_dec0, enc0_outputs[0])
            sep_dec_outputs.append(self.t1c_out(t1c_out_final))

            t1_dec3 = self.t1_decoder5(dec4_modality_specific_outputs[1], current_hs3_features[1])
            t1_dec2 = self.t1_decoder4(t1_dec3, enc3_outputs[1])
            t1_dec1 = self.t1_decoder3(t1_dec2, enc2_outputs[1])
            t1_dec0 = self.t1_decoder2(t1_dec1, enc1_outputs[1])
            t1_out_final = self.t1_decoder1(t1_dec0, enc0_outputs[1])
            sep_dec_outputs.append(self.t1_out(t1_out_final))

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
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )