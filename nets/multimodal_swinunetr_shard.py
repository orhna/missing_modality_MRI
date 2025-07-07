from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Final
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, Convolution
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from utils.mm_network_utils import (window_partition,window_reverse,
                                    WindowAttention,SwinTransformerBlock,PatchMergingV2,PatchMerging,
                                    SwinTransformer,MM_Transformer,FeatureFusionModule,DeepSupervisionHead)

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


class Multimodal_SwinUNETR_shard(nn.Module):
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
        device_list: list,
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
        self.device_list = device_list
        self.cross_attention = cross_attention
        self.deep_supervision = deep_supervision
        self.is_training = True
        self.t1c_spec = t1c_spec

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
        ).to(self.device_list[0])

        self.swinViT_2 = SwinTransformer(
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
        ).to(self.device_list[0])

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
        ).to(self.device_list[1])

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
        ).to(self.device_list[1])

        ######################################################## 
        self.encoder1_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder2_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder3_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder4_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder10_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])


        self.encoder1_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder2_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder3_2= UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder4_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.encoder10_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])
        
        
        self.encoder1_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder2_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder3_3= UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder4_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder10_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])


        self.encoder1_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder2_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder3_4= UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder4_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])

        self.encoder10_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])
       

        self.channel_reduction_1 = Convolution(spatial_dims=3,
                                               in_channels=4 * feature_size,
                                               out_channels=feature_size,
                                               strides=1,
                                               kernel_size=1,
                                               dropout=0.2).to(self.device_list[0])
        self.channel_reduction_2 = Convolution(spatial_dims=3,
                                               in_channels=4 * feature_size,
                                               out_channels=feature_size,
                                               strides=1,
                                               kernel_size=1,
                                               dropout=0.2).to(self.device_list[0])
        self.channel_reduction_3 = Convolution(spatial_dims=3,
                                               in_channels=8 * feature_size,
                                               out_channels=2 * feature_size,
                                               strides=1,
                                               kernel_size=1,
                                               dropout=0.2).to(self.device_list[0])
        self.channel_reduction_4 = Convolution(spatial_dims=3,
                                               in_channels=16 * feature_size,
                                               out_channels=4 * feature_size,
                                               strides=1,
                                               kernel_size=1,
                                               dropout=0.2).to(self.device_list[0])
        self.channel_reduction_5 = Convolution(spatial_dims=3,
                                               in_channels=32 * feature_size,
                                               out_channels=8 * feature_size,
                                               strides=1,
                                               kernel_size=1,
                                               dropout=0.2).to(self.device_list[0])
        self.channel_reduction_6 = Convolution(spatial_dims=3,
                                               in_channels=64 * feature_size,
                                               out_channels=16 * feature_size,
                                               strides=1,
                                               kernel_size=1,
                                               dropout=0.2).to(self.device_list[0])
        
        
        # cross attention
        if self.cross_attention:
            self.fusion_module = FeatureFusionModule(in_channels=self.feature_size * 16, num_modalities=4).to(self.device_list[1])

        # decoder part
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[0])

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels).to(self.device_list[0])

        if self.t1c_spec:
            self.t1c_decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ).to(self.device_list[1])
            self.t1c_decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            ).to(self.device_list[1])
            self.t1c_decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            ).to(self.device_list[1])
            self.t1c_decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            ).to(self.device_list[1])
            self.t1c_decoder1 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            ).to(self.device_list[1])
            self.t1c_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=1).to(self.device_list[1])

        if self.deep_supervision:
            # Deep supervision heads
            self.ds_head4 = DeepSupervisionHead(in_channels=16 * feature_size, out_channels=out_channels, scale_factor=32).to(self.device_list[1])
            self.ds_head3 = DeepSupervisionHead(in_channels=8 * feature_size, out_channels=out_channels, scale_factor=16).to(self.device_list[1])
            self.ds_head2 = DeepSupervisionHead(in_channels=4 * feature_size, out_channels=out_channels, scale_factor=8).to(self.device_list[1])
            self.ds_head1 = DeepSupervisionHead(in_channels=2 * feature_size, out_channels=out_channels, scale_factor=4).to(self.device_list[1])
            self.ds_head0 = DeepSupervisionHead(in_channels=feature_size, out_channels=out_channels, scale_factor=2).to(self.device_list[1])


    def forward(self, x_in, bottleneck=None):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        hidden_states_out_1 = self.swinViT_1(x_in[:,0:1,:,:].to(self.device_list[0]), self.normalize)
        hidden_states_out_2 = self.swinViT_2(x_in[:,1:2,:,:].to(self.device_list[0]), self.normalize)
        hidden_states_out_3 = self.swinViT_3(x_in[:,2:3,:,:].to(self.device_list[1]), self.normalize)
        hidden_states_out_4 = self.swinViT_4(x_in[:,3:4,:,:].to(self.device_list[1]), self.normalize)
        
        enc0_1 = self.encoder1_1(x_in[:,0:1,:,:])
        enc1_1 = self.encoder2_1(hidden_states_out_1[0])
        enc2_1 = self.encoder3_1(hidden_states_out_1[1])
        enc3_1 = self.encoder4_1(hidden_states_out_1[2])
        dec4_1 = self.encoder10_1(hidden_states_out_1[4])

        enc0_2 = self.encoder1_2(x_in[:,1:2,:,:])
        enc1_2 = self.encoder2_2(hidden_states_out_2[0])
        enc2_2 = self.encoder3_2(hidden_states_out_2[1])
        enc3_2 = self.encoder4_2(hidden_states_out_2[2])
        dec4_2 = self.encoder10_2(hidden_states_out_2[4])
        
        enc0_3 = self.encoder1_3(x_in[:,2:3,:,:].to(self.device_list[1]))
        enc1_3 = self.encoder2_3(hidden_states_out_3[0])
        enc2_3 = self.encoder3_3(hidden_states_out_3[1])
        enc3_3 = self.encoder4_3(hidden_states_out_3[2])
        dec4_3 = self.encoder10_3(hidden_states_out_3[4])

        enc0_4 = self.encoder1_4(x_in[:,3:4,:,:].to(self.device_list[1]))
        enc1_4 = self.encoder2_4(hidden_states_out_4[0])
        enc2_4 = self.encoder3_4(hidden_states_out_4[1])
        enc3_4 = self.encoder4_4(hidden_states_out_4[2])
        dec4_4 = self.encoder10_4(hidden_states_out_4[4])

        enc0 = self.channel_reduction_1(torch.cat([enc0_1, enc0_2, enc0_3.to(self.device_list[0]), enc0_4.to(self.device_list[0])], dim=1))
        enc1 = self.channel_reduction_2(torch.cat([enc1_1, enc1_2, enc1_3.to(self.device_list[0]), enc1_4.to(self.device_list[0])], dim=1))
        enc2 = self.channel_reduction_3(torch.cat([enc2_1, enc2_2, enc2_3.to(self.device_list[0]), enc2_4.to(self.device_list[0])], dim=1))
        enc3 = self.channel_reduction_4(torch.cat([enc3_1, enc3_2, enc3_3.to(self.device_list[0]), enc3_4.to(self.device_list[0])], dim=1))
        hidden_states_combined = self.channel_reduction_5(torch.cat([hidden_states_out_1[3],
                                                                     hidden_states_out_2[3],
                                                                     hidden_states_out_3[3].to(self.device_list[0]),
                                                                     hidden_states_out_4[3].to(self.device_list[0])], dim=1))
        
        if self.cross_attention:
            if bottleneck == None:
                dec4 = self.fusion_module([dec4_1.to(self.device_list[1]), dec4_2.to(self.device_list[1]), dec4_3, dec4_4]).to(self.device_list[0])
            else:
                dec4 = self.channel_reduction_6(bottleneck)
        else:
            if bottleneck == None:
                dec4 = self.channel_reduction_6(torch.cat([dec4_1, dec4_2, dec4_3.to(self.device_list[0]), dec4_4.to(self.device_list[0])], dim=1))
            else:
                dec4 = self.channel_reduction_6(bottleneck)
        
        # t1c special
        if self.t1c_spec and self.is_training:
            t1c_dec3 = self.t1c_decoder5(dec4_2.to(self.device_list[1]), hidden_states_out_2[3].to(self.device_list[1]))
            t1c_dec2 = self.t1c_decoder4(t1c_dec3.to(self.device_list[1]), enc3_2.to(self.device_list[1]))
            t1c_dec1 = self.t1c_decoder3(t1c_dec2.to(self.device_list[1]), enc2_2.to(self.device_list[1]))
            t1c_dec0 = self.t1c_decoder2(t1c_dec1.to(self.device_list[1]), enc1_2.to(self.device_list[1]))
            t1c_out = self.t1c_decoder1(t1c_dec0.to(self.device_list[1]), enc0_2.to(self.device_list[1]))
            t1c_logits = self.t1c_out(t1c_out.to(self.device_list[1])).to(self.device_list[0])


        dec3 = self.decoder5(dec4, hidden_states_combined) # [1, 192, 8, 8, 8]
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        
        # dec4 deepest [768,4,4,4]
        if self.deep_supervision and self.is_training:
            ds_output4 = self.ds_head4(dec4.to(self.device_list[1])).to(self.device_list[0])
            ds_output3 = self.ds_head3(dec3.to(self.device_list[1])).to(self.device_list[0])
            ds_output2 = self.ds_head2(dec2.to(self.device_list[1])).to(self.device_list[0])
            ds_output1 = self.ds_head1(dec1.to(self.device_list[1])).to(self.device_list[0])
            ds_output0 = self.ds_head0(dec0.to(self.device_list[1])).to(self.device_list[0])

            if self.t1c_spec:
                return logits, [ds_output0, ds_output1, ds_output2, ds_output3, ds_output4], t1c_logits
            else:
                return logits, [ds_output0, ds_output1, ds_output2, ds_output3, ds_output4]
        
        if self.t1c_spec and self.is_training:
            return logits, t1c_logits
        
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
