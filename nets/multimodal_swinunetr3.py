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


class Multimodal_SwinUNETR3(nn.Module):
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
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        cross_attention:bool = False,
        deep_supervision:bool = False,
        t1c_spec:bool = False,
        sep_dec:bool = False,
        tp_conv:bool = False,
        dec_upsample:bool = False,
        ldm_eval:bool = False,
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
        self.cross_attention = cross_attention
        self.deep_supervision = deep_supervision
        self.is_training = True
        self.t1c_spec = t1c_spec
        self.sep_dec = sep_dec
        self.tp_conv = tp_conv
        self.dec_upsample = dec_upsample
        self.ldm_eval = ldm_eval
        self.d_m = 0
        self.diff_on = "combined" # {"combined","separate"}


        if self.dec_upsample and self.tp_conv:
            raise ValueError("dec_upsample and tp_conv is not suppose to go together")

        if self.sep_dec and self.t1c_spec:
            raise ValueError("separate decoders and t1c only decoder is not suppose to go together")
        # swinvit for each modality
        self.swinViT_1 = SwinTransformer(
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
        )

        self.swinViT_3 = SwinTransformer(
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
        )

        self.swinViT_4 = SwinTransformer(
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
        )

        ######################################################## 
        self.encoder1_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder1_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3_3= UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder1_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3_4= UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
       
        if self.cross_attention:
            self.channel_reduction_1 = CrossAttentionFusion3D(in_channels=feature_size, out_channels=feature_size)
            self.channel_reduction_2 = CrossAttentionFusion3D(in_channels=feature_size, out_channels=feature_size)
            self.channel_reduction_3 = CrossAttentionFusion3D(in_channels=2 * feature_size, out_channels=2 * feature_size)
            self.channel_reduction_4 = CrossAttentionFusion3D(in_channels=4 * feature_size, out_channels=4 * feature_size)
            self.channel_reduction_5 = CrossAttentionFusion3D(in_channels=8 * feature_size, out_channels=8 * feature_size)
            self.channel_reduction_6 = CrossAttentionFusion3D(in_channels=16 * feature_size, out_channels=16 * feature_size)
        else:
            self.channel_reduction_1 = Convolution(spatial_dims=3,
                                                in_channels=3 * feature_size,
                                                out_channels=feature_size,
                                                strides=1,
                                                kernel_size=1,
                                                dropout=0.2)
            self.channel_reduction_2 = Convolution(spatial_dims=3,
                                                in_channels=3 * feature_size,
                                                out_channels=feature_size,
                                                strides=1,
                                                kernel_size=1,
                                                dropout=0.2)
            self.channel_reduction_3 = Convolution(spatial_dims=3,
                                                in_channels=6 * feature_size,
                                                out_channels=2 * feature_size,
                                                strides=1,
                                                kernel_size=1,
                                                dropout=0.2)
            self.channel_reduction_4 = Convolution(spatial_dims=3,
                                                in_channels=12 * feature_size,
                                                out_channels=4 * feature_size,
                                                strides=1,
                                                kernel_size=1,
                                                dropout=0.2)
            self.channel_reduction_5 = Convolution(spatial_dims=3,
                                                in_channels=24 * feature_size,
                                                out_channels=8 * feature_size,
                                                strides=1,
                                                kernel_size=1,
                                                dropout=0.2)
            self.channel_reduction_6 = Convolution(spatial_dims=3,
                                                in_channels=48 * feature_size,
                                                out_channels=16 * feature_size,
                                                strides=1,
                                                kernel_size=1,
                                                dropout=0.2)
            
        
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

        
        if self.sep_dec:
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
           
        if self.deep_supervision:
            # Deep supervision heads
            self.ds_head4 = DeepSupervisionHead(in_channels=16 * feature_size, out_channels=out_channels, scale_factor=32)
            self.ds_head3 = DeepSupervisionHead(in_channels=8 * feature_size, out_channels=out_channels, scale_factor=16)
            self.ds_head2 = DeepSupervisionHead(in_channels=4 * feature_size, out_channels=out_channels, scale_factor=8)
            self.ds_head1 = DeepSupervisionHead(in_channels=2 * feature_size, out_channels=out_channels, scale_factor=4)
            self.ds_head0 = DeepSupervisionHead(in_channels=feature_size, out_channels=out_channels, scale_factor=2)

        if self.dec_upsample:
            self.dec_upblock1 = DecUpsampleBlock(192, 96, kernel_size=4, stride=2, padding=1, output_padding=0)
            self.dec_upblock2 = DecUpsampleBlock(96, 48, kernel_size=4, stride=2, padding=1, output_padding=0)
            self.dec_upblock3 = DecUpsampleBlock(48, 24, kernel_size=4, stride=2, padding=1, output_padding=0)
            self.dec_upblock4 = DecUpsampleBlock(24, 12, kernel_size=4, stride=2, padding=1, output_padding=0)
            self.dec_upblock5 = DecUpsampleBlock(12, 12, kernel_size=4, stride=2, padding=1, output_padding=0)



    def forward(self, x_in, bottleneck=None):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        hidden_states_out_1 = self.swinViT_1(x_in[:,0:1,:,:], self.normalize)
        hidden_states_out_3 = self.swinViT_3(x_in[:,2:3,:,:], self.normalize)
        hidden_states_out_4 = self.swinViT_4(x_in[:,3:4,:,:], self.normalize)
        
        """
        hidden_states_out_1[0].shape #[1, 12, 64, 64, 64]
        hidden_states_out_1[1].shape #[1, 24, 32, 32, 32]
        hidden_states_out_1[2].shape #[1, 48, 16, 16, 16]
        hidden_states_out_1[3].shape #[1, 96, 8, 8, 8]
        hidden_states_out_1[4].shape #[1, 192, 4, 4, 4]
        """
        # overwriting the skip connections of FLAIR during ldm_eval
        if self.tp_conv and self.ldm_eval and bottleneck and self.diff_on=="separate": 
            drop_list = bottleneck[1]
            bottleneck_features = bottleneck[0]
            # for flair
            if 0 in drop_list:
                enc0_1 = self.encoder1_1(x_in[:,0:1,:,:])
                dec4_1 = bottleneck_features[:,0:self.feature_size*16] #[1, 192, 4, 4, 4]
                hidden_states_out_1[3] = self.m1_activation(self.m1_tp_norm1(self.m1_tp_layer1(dec4_1)))  # [1, 96, 8, 8, 8]
                enc3_1 = self.encoder4_1(self.m1_activation(self.m1_tp_norm2(self.m1_tp_layer2(hidden_states_out_1[3]))))  # [1, 48, 16, 16, 16]
                enc2_1 = self.encoder3_1(self.m1_activation(self.m1_tp_norm3(self.m1_tp_layer3(enc3_1)))) # [1, 24, 32, 32, 32]
                enc1_1 = self.encoder2_1(self.m1_activation(self.m1_tp_norm4(self.m1_tp_layer4(enc2_1))))
            else:
                enc0_1 = self.encoder1_1(x_in[:,0:1,:,:]) 
                enc1_1 = self.encoder2_1(hidden_states_out_1[0]) 
                enc2_1 = self.encoder3_1(hidden_states_out_1[1]) 
                enc3_1 = self.encoder4_1(hidden_states_out_1[2]) 
                dec4_1 = self.encoder10_1(hidden_states_out_1[4])
            # for t1
            if 2 in drop_list:
                enc0_3 = self.encoder1_3(x_in[:,2:3,:,:])
                dec4_3 = bottleneck_features[:,2*self.feature_size*16:3*self.feature_size*16] #[1, 192, 4, 4, 4]
                hidden_states_out_3[3] = self.m3_activation(self.m3_tp_norm1(self.m3_tp_layer1(dec4_3)))  # [1, 96, 8, 8, 8]
                enc3_3 = self.encoder4_3(self.m3_activation(self.m3_tp_norm2(self.m3_tp_layer2(hidden_states_out_3[3]))))  # [1, 48, 16, 16, 16]
                enc2_3 = self.encoder3_3(self.m3_activation(self.m3_tp_norm3(self.m3_tp_layer3(enc3_3)))) # [1, 24, 32, 32, 32]
                enc1_3 = self.encoder2_3(self.m3_activation(self.m3_tp_norm4(self.m3_tp_layer4(enc2_3))))
            else:
                enc0_3 = self.encoder1_3(x_in[:,2:3,:,:])
                enc1_3 = self.encoder2_3(hidden_states_out_3[0])
                enc2_3 = self.encoder3_3(hidden_states_out_3[1])
                enc3_3 = self.encoder4_3(hidden_states_out_3[2])
                dec4_3 = self.encoder10_3(hidden_states_out_3[4])
            # for t2
            if 3 in drop_list:
                enc0_4 = self.encoder1_4(x_in[:,3:4,:,:])
                dec4_4 = bottleneck_features[:,3*self.feature_size*16:] #[1, 192, 4, 4, 4]
                hidden_states_out_4[3] = self.m4_activation(self.m4_tp_norm1(self.m4_tp_layer1(dec4_4)))  # [1, 96, 8, 8, 8]
                enc3_4 = self.encoder4_4(self.m4_activation(self.m4_tp_norm2(self.m4_tp_layer2(hidden_states_out_4[3]))))  # [1, 48, 16, 16, 16]
                enc2_4 = self.encoder3_4(self.m4_activation(self.m4_tp_norm3(self.m4_tp_layer3(enc3_4)))) # [1, 24, 32, 32, 32]
                enc1_4 = self.encoder2_4(self.m4_activation(self.m4_tp_norm4(self.m4_tp_layer4(enc2_4))))
            else:
                enc0_4 = self.encoder1_4(x_in[:,3:4,:,:])
                enc1_4 = self.encoder2_4(hidden_states_out_4[0])
                enc2_4 = self.encoder3_4(hidden_states_out_4[1])
                enc3_4 = self.encoder4_4(hidden_states_out_4[2])
                dec4_4 = self.encoder10_4(hidden_states_out_4[4])
        
        else:
            enc0_1 = self.encoder1_1(x_in[:,0:1,:,:]) 
            enc1_1 = self.encoder2_1(hidden_states_out_1[0]) 
            enc2_1 = self.encoder3_1(hidden_states_out_1[1]) 
            enc3_1 = self.encoder4_1(hidden_states_out_1[2]) 
            dec4_1 = self.encoder10_1(hidden_states_out_1[4])

            """
            print(enc0_1.shape) #[1, 12, 128, 128, 128]
            print(enc1_1.shape) #[1, 12, 64, 64, 64]
            print(enc2_1.shape) #[1, 24, 32, 32, 32]
            print(enc3_1.shape) #[1, 48, 16, 16, 16]
            """
            
            enc0_3 = self.encoder1_3(x_in[:,2:3,:,:])
            enc1_3 = self.encoder2_3(hidden_states_out_3[0])
            enc2_3 = self.encoder3_3(hidden_states_out_3[1])
            enc3_3 = self.encoder4_3(hidden_states_out_3[2])
            dec4_3 = self.encoder10_3(hidden_states_out_3[4])

            enc0_4 = self.encoder1_4(x_in[:,3:4,:,:])
            enc1_4 = self.encoder2_4(hidden_states_out_4[0])
            enc2_4 = self.encoder3_4(hidden_states_out_4[1])
            enc3_4 = self.encoder4_4(hidden_states_out_4[2])
            dec4_4 = self.encoder10_4(hidden_states_out_4[4])

        if self.cross_attention:
            enc0 = self.channel_reduction_1(torch.stack([enc0_1, enc0_3, enc0_4], dim=1))
            enc1 = self.channel_reduction_2(torch.stack([enc1_1, enc1_3, enc1_4], dim=1))
            enc2 = self.channel_reduction_3(torch.stack([enc2_1, enc2_3, enc2_4], dim=1))
            enc3 = self.channel_reduction_4(torch.stack([enc3_1, enc3_3, enc3_4], dim=1))
            hidden_states_combined = self.channel_reduction_5(torch.stack([hidden_states_out_1[3],
                                                                    hidden_states_out_3[3],
                                                                    hidden_states_out_4[3]], dim=1))
            if bottleneck == None:
                dec4 = self.channel_reduction_6(torch.stack([dec4_1, dec4_3, dec4_4], dim=1))
            elif bottleneck and self.diff_on == "separate":
                drop_list = bottleneck[1]

                if isinstance(bottleneck[0], list):
                    # bottleneck layer, level4
                    to_cat_level4 = [dec4_1, dec4_3, dec4_4]
                    for i in drop_list:
                        start_i= i *self.feature_size*16
                        end_i= start_i+ self.feature_size*16
                        #print("star_i:",start_i,"--end_i:",end_i)
                        to_cat_level4[i]=bottleneck[0][0][:,start_i:end_i]
                    dec4 = self.channel_reduction_6(torch.stack(to_cat_level4, dim=1))
                    # upper layer, level3
                    to_cat_level3 = [hidden_states_out_1[3], hidden_states_out_3[3], hidden_states_out_4[3]]
                    for i in drop_list:
                        start_i= i *self.feature_size*8
                        end_i= start_i+ self.feature_size*8
                        #print("star_i:",start_i,"--end_i:",end_i)
                        to_cat_level4[i]=bottleneck[0][1][:,start_i:end_i]
                    hidden_states_combined = self.channel_reduction_5(torch.stack(to_cat_level3, dim=1))

                else:
                    to_cat_level4 = [dec4_1, dec4_3, dec4_4]
                    for i in drop_list:
                        start_i= i *self.feature_size*16
                        end_i= start_i+ self.feature_size*16
                        #print("star_i:",start_i,"--end_i:",end_i)
                        to_cat_level4[i]=bottleneck[0][:,start_i:end_i]
                    
                    dec4 = self.channel_reduction_6(torch.stack(to_cat_level4, dim=1))
                    
            elif bottleneck and self.diff_on == "combined":    
                if isinstance(bottleneck[0], list):
                    dec4 = bottleneck[0][0]
                    hidden_states_combined = bottleneck[0][1]
                else:
                    dec4 = bottleneck[0]
        else:
            enc0 = self.channel_reduction_1(torch.cat([enc0_1, enc0_3, enc0_4], dim=1))
            enc1 = self.channel_reduction_2(torch.cat([enc1_1, enc1_3, enc1_4], dim=1))
            enc2 = self.channel_reduction_3(torch.cat([enc2_1, enc2_3, enc2_4], dim=1))
            enc3 = self.channel_reduction_4(torch.cat([enc3_1, enc3_3, enc3_4], dim=1))
            hidden_states_combined = self.channel_reduction_5(torch.cat([hidden_states_out_1[3],
                                                                        hidden_states_out_3[3],
                                                                        hidden_states_out_4[3]], dim=1))
                        
            if bottleneck == None:
                dec4 = self.channel_reduction_6(torch.cat([dec4_1, dec4_3, dec4_4], dim=1))
            
            elif bottleneck and self.diff_on == "separate":
                drop_list = bottleneck[1]

                if isinstance(bottleneck[0], list):
                    # bottleneck layer, level4
                    to_cat_level4 = [dec4_1, dec4_3, dec4_4]
                    for i in drop_list:
                        start_i= i *self.feature_size*16
                        end_i= start_i+ self.feature_size*16
                        #print("star_i:",start_i,"--end_i:",end_i)
                        to_cat_level4[i]=bottleneck[0][0][:,start_i:end_i]
                    dec4 = self.channel_reduction_6(torch.cat(to_cat_level4, dim=1))
                    # upper layer, level3
                    to_cat_level3 = [hidden_states_out_1[3], hidden_states_out_3[3], hidden_states_out_4[3]]
                    for i in drop_list:
                        start_i= i *self.feature_size*8
                        end_i= start_i+ self.feature_size*8
                        #print("star_i:",start_i,"--end_i:",end_i)
                        to_cat_level4[i]=bottleneck[0][1][:,start_i:end_i]
                    hidden_states_combined = self.channel_reduction_5(torch.cat(to_cat_level3, dim=1))

                else:
                    to_cat_level4 = [dec4_1, dec4_3, dec4_4]
                    for i in drop_list:
                        start_i= i *self.feature_size*16
                        end_i= start_i+ self.feature_size*16
                        #print("star_i:",start_i,"--end_i:",end_i)
                        to_cat_level4[i]=bottleneck[0][:,start_i:end_i]
                    
                    dec4 = self.channel_reduction_6(torch.cat(to_cat_level4, dim=1))
                    
            elif bottleneck and self.diff_on == "combined":    
                if isinstance(bottleneck[0], list):
                    dec4 = bottleneck[0][0]
                    hidden_states_combined = bottleneck[0][1]
                else:
                    dec4 = bottleneck[0]

        if self.dec_upsample and self.ldm_eval and bottleneck and self.diff_on=="combined":
            hidden_states_combined = self.dec_upblock1(dec4)
            enc3 = self.dec_upblock2(hidden_states_combined)
            enc2 = self.dec_upblock3(enc3)
            enc1 = self.dec_upblock4(enc2)
            enc0 = self.dec_upblock5(enc1)

        if self.sep_dec and self.is_training:
            f_dec3 = self.f_decoder5(dec4_1, hidden_states_out_1[3])
            f_dec2 = self.f_decoder4(f_dec3, enc3_1)
            f_dec1 = self.f_decoder3(f_dec2, enc2_1)
            f_dec0 = self.f_decoder2(f_dec1, enc1_1)
            f_out = self.f_decoder1(f_dec0, enc0_1)
            f_logits = self.f_out(f_out)

            t1_dec3 = self.t1_decoder5(dec4_3, hidden_states_out_3[3])
            t1_dec2 = self.t1_decoder4(t1_dec3, enc3_3)
            t1_dec1 = self.t1_decoder3(t1_dec2, enc2_3)
            t1_dec0 = self.t1_decoder2(t1_dec1, enc1_3)
            t1_out = self.t1_decoder1(t1_dec0, enc0_3)
            t1_logits = self.t1_out(t1_out)

            t2_dec3 = self.t2_decoder5(dec4_4, hidden_states_out_4[3])
            t2_dec2 = self.t2_decoder4(t2_dec3, enc3_4)
            t2_dec1 = self.t2_decoder3(t2_dec2, enc2_4)
            t2_dec0 = self.t2_decoder2(t2_dec1, enc1_4)
            t2_out = self.t2_decoder1(t2_dec0, enc0_4)
            t2_logits = self.t2_out(t2_out)
        

        # decoding for all starts here
        dec3 = self.decoder5(dec4, hidden_states_combined) # [1, 192, 8, 8, 8]
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        
        """
        print(dec4.shape)
        print(hidden_states_combined.shape)
        print(dec3.shape)
        print(dec2.shape)
        print(dec1.shape)
        print(dec0.shape)
        """


        """
        print(dec4.shape) [1, 192, 4, 4, 4] 
        print(hidden_states_combined.shape) [1, 96, 8, 8, 8]
        print(enc3.shape) [1, 48, 16, 16, 16]
        print(enc2.shape) [1, 24, 32, 32, 32]
        print(enc1.shape) [1, 12, 64, 64, 64]
        print(enc0.shape) [1, 12, 128, 128, 128]
        """
        if self.dec_upsample and self.is_training and self.d_m == "no_drop":
            reconstructed_h_s_combined = self.dec_upblock1(dec4)
            reconstructed_enc3 = self.dec_upblock2(reconstructed_h_s_combined)
            reconstructed_enc2 = self.dec_upblock3(reconstructed_enc3)
            reconstructed_enc1 = self.dec_upblock4(reconstructed_enc2)
            reconstructed_enc0 = self.dec_upblock5(reconstructed_enc1)

            mse_rec_hsc = F.mse_loss(reconstructed_h_s_combined, hidden_states_combined)
            mse_rec_enc3 = F.mse_loss(reconstructed_enc3, enc3)
            mse_rec_enc2 = F.mse_loss(reconstructed_enc2, enc2)
            mse_rec_enc1 = F.mse_loss(reconstructed_enc1, enc1)
            mse_rec_enc0 = F.mse_loss(reconstructed_enc0, enc0)
            dec_upsample_mse_list = [mse_rec_hsc,mse_rec_enc3,mse_rec_enc2,mse_rec_enc1,mse_rec_enc0]

        if self.tp_conv and self.is_training and self.d_m == "no_drop":

            mse_list = []

            # for flair
            m1_tp1 = self.m1_activation(self.m1_tp_norm1(self.m1_tp_layer1(dec4_1)))  # [1, 96, 8, 8, 8]
            m1_tp2 = self.m1_activation(self.m1_tp_norm2(self.m1_tp_layer2(m1_tp1)))  # [1, 48, 16, 16, 16]
            m1_tp3 = self.m1_activation(self.m1_tp_norm3(self.m1_tp_layer3(m1_tp2)))  # [1, 24, 32, 32, 32]
            m1_tp4 = self.m1_activation(self.m1_tp_norm4(self.m1_tp_layer4(m1_tp3)))  # [1, 12, 64, 64, 64]
            #print(m1_tp1.shape) # [1, 96, 8, 8, 8]
            #print(m1_tp2.shape) # [1, 48, 16, 16, 16]
            #print(m1_tp3.shape) # [1, 24, 32, 32, 32]
            #print(m1_tp4.shape) # [1, 12, 64, 64, 64]

            mse_m1_tp4 = F.mse_loss(m1_tp4,hidden_states_out_1[0])
            mse_m1_tp3 = F.mse_loss(m1_tp3,hidden_states_out_1[1])
            mse_m1_tp2 = F.mse_loss(m1_tp2,hidden_states_out_1[2])
            mse_m1_tp1 = F.mse_loss(m1_tp1,hidden_states_out_1[3])
            mse_list.append([mse_m1_tp4,mse_m1_tp3,mse_m1_tp2,mse_m1_tp1])

            # for t1
            m3_tp1 = self.m3_activation(self.m3_tp_norm1(self.m3_tp_layer1(dec4_3)))  # [1, 96, 8, 8, 8]
            m3_tp2 = self.m3_activation(self.m3_tp_norm2(self.m3_tp_layer2(m3_tp1)))  # [1, 48, 16, 16, 16]
            m3_tp3 = self.m3_activation(self.m3_tp_norm3(self.m3_tp_layer3(m3_tp2)))  # [1, 24, 32, 32, 32]
            m3_tp4 = self.m3_activation(self.m3_tp_norm4(self.m3_tp_layer4(m3_tp3)))  # [1, 12, 64, 64, 64]

            mse_m3_tp4 = F.mse_loss(m3_tp4,hidden_states_out_3[0])
            mse_m3_tp3 = F.mse_loss(m3_tp3,hidden_states_out_3[1])
            mse_m3_tp2 = F.mse_loss(m3_tp2,hidden_states_out_3[2])
            mse_m3_tp1 = F.mse_loss(m3_tp1,hidden_states_out_3[3])
            mse_list.append([mse_m3_tp4,mse_m3_tp3,mse_m3_tp2,mse_m3_tp1])

            # for t2
            m4_tp1 = self.m4_activation(self.m4_tp_norm1(self.m4_tp_layer1(dec4_4)))  # [1, 96, 8, 8, 8]
            m4_tp2 = self.m4_activation(self.m4_tp_norm2(self.m4_tp_layer2(m4_tp1)))  # [1, 48, 16, 16, 16]
            m4_tp3 = self.m4_activation(self.m4_tp_norm3(self.m4_tp_layer3(m4_tp2)))  # [1, 24, 32, 32, 32]
            m4_tp4 = self.m4_activation(self.m4_tp_norm4(self.m4_tp_layer4(m4_tp3)))  # [1, 12, 64, 64, 64]

            mse_m4_tp4 = F.mse_loss(m4_tp4,hidden_states_out_4[0])
            mse_m4_tp3 = F.mse_loss(m4_tp3,hidden_states_out_4[1])
            mse_m4_tp2 = F.mse_loss(m4_tp2,hidden_states_out_4[2])
            mse_m4_tp1 = F.mse_loss(m4_tp1,hidden_states_out_4[3])
            mse_list.append([mse_m4_tp4,mse_m4_tp3,mse_m4_tp2,mse_m4_tp1])



        # dec4 deepest [768,4,4,4]
        
        if self.deep_supervision and self.is_training:
            ds_output4 = self.ds_head4(dec4)
            ds_output3 = self.ds_head3(dec3)
            ds_output2 = self.ds_head2(dec2)
            ds_output1 = self.ds_head1(dec1)
            ds_output0 = self.ds_head0(dec0)

            if self.sep_dec:
                if self.tp_conv and self.d_m == "no_drop":
                    return logits, [ds_output0, ds_output1, ds_output2, ds_output3, ds_output4], [f_logits, t1_logits, t2_logits], mse_list
                elif self.dec_upsample and self.d_m == "no_drop":
                    return logits, [ds_output0, ds_output1, ds_output2, ds_output3, ds_output4], [f_logits, t1_logits, t2_logits], dec_upsample_mse_list
                else:
                    return logits, [ds_output0, ds_output1, ds_output2, ds_output3, ds_output4], [f_logits, t1_logits, t2_logits]
            else:
                return logits, [ds_output0, ds_output1, ds_output2, ds_output3, ds_output4]

        if self.sep_dec and self.is_training:
            return logits, [f_logits, t1_logits, t2_logits]

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


"""

# pos encoding
        if self.pos_encoding:
            transformer_basic_dims = 512
            depth = 1
            mlp_dim = 4096
            num_modals = 4
            i_dim = self.feature_size * 8
            self.m1_pos = nn.Parameter(torch.zeros(1, i_dim, transformer_basic_dims))
            self.m2_pos = nn.Parameter(torch.zeros(1, i_dim, transformer_basic_dims))
            self.m3_pos = nn.Parameter(torch.zeros(1, i_dim, transformer_basic_dims))
            self.m4_pos = nn.Parameter(torch.zeros(1, i_dim, transformer_basic_dims))

            self.multimodal_transformer = MM_Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=8, mlp_dim=mlp_dim, n_levels=num_modals)
        
        ##################


 # pos encoding
        if self.pos_encoding:
            hidden_states_combined = torch.cat([hidden_states_out_1[3],
                                                hidden_states_out_2[3],
                                                hidden_states_out_3[3],
                                                hidden_states_out_4[3]], dim=1).view(1,self.feature_size*32,512)
            multimodal_pos = torch.cat((self.m1_pos, self.m2_pos, self.m3_pos, self.m4_pos), dim=1)
            multimodal_hidden_states_combined = self.multimodal_transformer(hidden_states_combined, multimodal_pos).view(1,self.feature_size*32,8,8,8)
            multimodal_hidden_states_combined = self.channel_reduction_5(multimodal_hidden_states_combined)
            dec3 = self.decoder5(dec4, multimodal_hidden_states_combined)
        else:
            hidden_states_combined = self.channel_reduction_5(torch.cat([hidden_states_out_1[3],
                                                                     hidden_states_out_2[3],
                                                                     hidden_states_out_3[3],
                                                                     hidden_states_out_4[3]], dim=1))
            dec3 = self.decoder5(dec4, hidden_states_combined) # [1, 192, 8, 8, 8]
            

"""

class CrossAttentionFusion3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_modalities=3, heads=4, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.query_proj = nn.Linear(in_channels, out_channels)
        self.key_proj = nn.Linear(in_channels, out_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)

        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.out_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, features):  # [B, num_modalities, C, D, H, W]
        B, M, C, D, H, W = features.shape
        x = features.permute(0, 3, 4, 5, 1, 2)  # [B, D, H, W, M, C]
        x = x.reshape(-1, M, C)  # [(B*D*H*W), M, C]

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        attn_output, _ = self.attn(q, k, v)  # [(B*D*H*W), M, C]
        attn_output = attn_output.mean(dim=1)  # [(B*D*H*W), C]
        attn_output = attn_output.reshape(B, D, H, W, self.out_channels).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]

        return self.out_conv(attn_output)

