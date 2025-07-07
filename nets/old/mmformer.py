import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
import random
import numpy as np
import os
import glob
from monai.transforms import Transform
from monai.transforms import (Lambda,
                               Compose, EnsureChannelFirst,
                                 RandSpatialCrop, RandRotate90                             
                                )
from monai.data import ImageDataset,DataLoader
from monai.losses import FocalLoss
from monai.transforms import Compose, Lambda
from monai.networks.blocks import  Convolution

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[:,mask, ...] = x[:,mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        ########### IntraFormer
        self.flair_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.flair_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1ce_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims*num_modals, basic_dims*16*num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        # channel reduction for training without interformer
        self.channel_reduction_1 = Convolution(spatial_dims=3,
                                               in_channels=4 * transformer_basic_dims,
                                               out_channels=transformer_basic_dims,
                                               strides=1,
                                               kernel_size=1,
                                               dropout=0.2)
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask=None):
        #extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        assert not torch.isnan(flair_x1).any(), "flair_x1 contains NaNs!"
        assert not torch.isnan(t1ce_x1).any(), "t1ce_x1 contains NaNs!"
        assert not torch.isnan(t1_x1).any(), "t1_x1 contains NaNs!"
        assert not torch.isnan(t2_x1).any(), "t2_x1 contains NaNs!"
        ########### IntraFormer
        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        assert not torch.isnan(flair_token_x5).any(), "flair_token_x5 contains NaNs!"
        assert not torch.isnan(t1ce_token_x5).any(), "t1ce_token_x5 contains NaNs!"
        assert not torch.isnan(t1_token_x5).any(), "t1_token_x5 contains NaNs!"
        assert not torch.isnan(t2_token_x5).any(), "t2_token_x5 contains NaNs!"

        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)
        assert not torch.isnan(flair_intra_token_x5).any(), "flair_intra_token_x5 contains NaNs!"
        assert not torch.isnan(t1ce_intra_token_x5).any(), "t1ce_intra_token_x5 contains NaNs!"
        assert not torch.isnan(t1_intra_token_x5).any(), "t1_intra_token_x5 contains NaNs!"
        assert not torch.isnan(t2_intra_token_x5).any(), "t2_intra_token_x5 contains NaNs!"

        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        assert not torch.isnan(flair_intra_x5).any(), "flair_intra_x5 contains NaNs!"
        assert not torch.isnan(t1ce_intra_x5).any(), "t1ce_intra_x5 contains NaNs!"
        assert not torch.isnan(t1_intra_x5).any(), "t1_intra_x5 contains NaNs!"
        assert not torch.isnan(t2_intra_x5).any(), "t2_intra_x5 contains NaNs!"

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
            assert not torch.isnan(flair_pred).any(), "flair_pred contains NaNs!"
            assert not torch.isnan(t1ce_pred).any(), "t1ce_pred contains NaNs!"
            assert not torch.isnan(t1_pred).any(), "t1_pred contains NaNs!"
            assert not torch.isnan(t2_pred).any(), "t2_pred contains NaNs!"

        if not self.is_training:
            mask = torch.tensor([True,True,True,True])

        ########### IntraFormer
        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask) #Bx4xCxHWZ
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)
        
        """    
        ########### InterFormer
        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1) 
        multimodal_token_x5 = torch.cat((flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x5_inter = multimodal_inter_x5
        
        
        #fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer
        """

        # to convert x5_intra to shape of x5_inter
        # x5_intra: [1, 2048, 8, 8, 8]
        # x5_inter :[1, 512, 8, 8, 8]
        x5_intra_reduced = self.channel_reduction_1(x5_intra)

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_intra_reduced)

        if self.is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
        
        return fuse_pred



def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)


class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            now_lr = round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8) 
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr



def mmformer_mask(dataset_modalities: list, batch_img_data: torch.Tensor, mode: str = "zero"):
    batch_size = batch_img_data.shape[0]
    dropped_info = []  # Store dropped modalities information
    
    for i in range(batch_size):
        number_of_dropped_modalities = np.random.randint(0, len(dataset_modalities))
        modalities_dropped = random.sample(list(np.arange(len(dataset_modalities))), number_of_dropped_modalities)
        modalities_dropped.sort()
        
        if number_of_dropped_modalities > 0:
            remaining_modalities = list(set(range(len(dataset_modalities))) - set(modalities_dropped))
            
            if mode == "zero":
                batch_img_data[i, modalities_dropped, :, :, :] = 0.
            elif mode == "mean":
                mean_value = batch_img_data[i, remaining_modalities, :, :, :].mean(dim=0, keepdim=True)
                batch_img_data[i, modalities_dropped, :, :, :] = mean_value
            elif mode == "noise":
                mean_value = batch_img_data[i, remaining_modalities, :, :, :].mean(dim=0, keepdim=True)
                std_value = batch_img_data[i, remaining_modalities, :, :, :].std(dim=0, keepdim=True)
                noise = torch.normal(mean=mean_value, std=std_value)
                batch_img_data[i, modalities_dropped, :, :, :] = noise
            else:
                raise ValueError("Invalid mode. Choose from 'zero', 'mean', or 'noise'.")

        # Create boolean tensor where dropped modalities are False
        bool_mask = torch.ones(len(dataset_modalities), dtype=torch.bool)  # Start with all True
        bool_mask[modalities_dropped] = False  # Set dropped modalities to False
        
        dropped_info.append(bool_mask)

    return batch_img_data, dropped_info[0]

class NormalizeNonZeroIntensity(Transform):
    """
    Normalize each modality (channel) separately using mean and std computed
    only from nonzero voxels, as done in MMFormer preprocessing.
    """
    def __call__(self, vol: torch.Tensor):
        vol = vol.clone()  # Ensure we don't modify the input tensor in-place
        mask = vol.sum(dim=0) > 0  # Create a mask of nonzero voxels

        for k in range(vol.shape[0]):  # Iterate over each modality
            x = vol[k, ...]
            y = x[mask]  # Select only nonzero values
            if y.numel() > 0:  # Ensure there are nonzero values
                mean = y.mean()
                std = y.std()
                vol[k, ...] = (x - mean) / (std + 1e-8)  # Normalize

        return vol
    
def get_mmformer_loaders(images,
                segs,
                train_file, val_file,
                workers,
                train_batch_size: int,
                cropped_input_size:list,
                channel_indices):

    train_images, train_segs, val_images, val_segs = separate_paths(images, segs, train_file, val_file)

    def select_channels(x):
        if x.ndim == 4:
            return x[..., channel_indices]
        else:
            return x

    train_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            #NormalizeIntensity(nonzero=True,channel_wise=True),
            NormalizeNonZeroIntensity(),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])
    train_labeltrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])

    val_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(),
            #NormalizeIntensity(nonzero=True,channel_wise=True)
            NormalizeNonZeroIntensity()
        ])
    val_segtrans = Compose([
            EnsureChannelFirst(),
        ])
    
    # create a training data loader
    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_labeltrans)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, drop_last=True, shuffle=True, num_workers=workers, pin_memory=0)
    
    # create a validation data loader
    val_ds = ImageDataset(val_images, val_segs, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    
    return train_loader, val_loader


def get_mmformer_test_loader(images_folder, labels_folder, txt_file, channel_indices):

    def select_channels(x):
        if x.ndim == 4:
            return x[..., channel_indices]
        else:
            return x

    #test_imtrans = Compose([Lambda(select_channels),EnsureChannelFirst(), NormalizeIntensity(nonzero=True,channel_wise=True)])
    test_imtrans = Compose([Lambda(select_channels),EnsureChannelFirst(),NormalizeNonZeroIntensity()])
    #test_imtrans = Compose([Lambda(select_channels),EnsureChannelFirst()])

    test_segtrans = Compose([EnsureChannelFirst()])

    with open(txt_file, 'r') as f:
        sample_names = {line.strip() for line in f}  # Use a set for faster lookup
    
    image_dict = {}
    label_dict = {}

    for file_name in os.listdir(images_folder):
        if file_name.endswith('.nii.gz'):
            base_name = file_name[:-7] 
            if base_name in sample_names:
                image_dict[base_name] = os.path.join(images_folder, file_name)

    for file_name in os.listdir(labels_folder):
        if file_name.endswith('.nii.gz') and '_label' in file_name:
            base_name = file_name[:-13] 
            if base_name in sample_names:
                label_dict[base_name] = os.path.join(labels_folder, file_name)

    common_base_names = sorted(image_dict.keys() & label_dict.keys())
    image_list = [image_dict[name] for name in common_base_names]
    label_list = [label_dict[name] for name in common_base_names]
    
    test_ds = ImageDataset(image_list, label_list, transform=test_imtrans, seg_transform=test_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=0)
    
    return test_loader


def separate_paths(image_list, label_list, train_file, val_file):
    
    def read_file(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    # Read train and validation sample names
    train_samples = read_file(train_file)
    val_samples = read_file(val_file)

    # Initialize lists for training and validation subsets
    train_images, train_labels = [], []
    val_images, val_labels = [], []

    # Separate paths based on sample names
    for image, label in zip(image_list, label_list):
        sample_name = os.path.basename(image).replace('.nii.gz', '')
        if sample_name in train_samples:
            train_images.append(image)
            train_labels.append(label)
        elif sample_name in val_samples:
            val_images.append(image)
            val_labels.append(label)

    return train_images, train_labels, val_images, val_labels


def dice_loss(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def softmax_weighted_loss(output, target, num_cls=4):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

def dice_loss_changed(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    dice = 0.0
    for i in range(num_cls):
        output_i = torch.clamp(output[:, i, :, :, :], min=eps, max=1.0)  # Clamp output
        target_i = target[:, i, :, :, :]
        num = torch.sum(output_i * target_i)
        l = torch.sum(output_i)
        r = torch.sum(target_i)
        dice += 2.0 * num / (l + r + eps)  # Add epsilon to denominator
    return 1.0 - (dice / num_cls)  # Normalize by number of classes

def softmax_weighted_loss_changed(output, target, num_cls=4, eps=1e-7):
    target = target.float()
    B, _, H, W, Z = output.size()
    cross_loss = 0.0
    for i in range(num_cls):
        output_i = torch.clamp(output[:, i, :, :, :], min=eps, max=1.0)  # Clamp output
        target_i = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(target_i, (1, 2, 3)) / (torch.sum(target, (1, 2, 3, 4)) + eps))  # Add epsilon
        weighted = torch.reshape(weighted, (-1, 1, 1, 1)).repeat(1, H, W, Z)
        cross_loss += -1.0 * weighted * target_i * torch.log(output_i)  # Use clamped output
    return torch.mean(cross_loss)

def softmax_weighted_loss_monai(output, target, num_cls=4, eps=1e-7):

    # Compute class weights dynamically
    weights = []
    for i in range(num_cls):
        target_i = (target == i).float()  # Binary mask for class i
        weight_i = 1.0 - (torch.sum(target_i) / torch.sum(target >= 0))  # Weight for class i
        weights.append(weight_i)
    weights = torch.stack(weights).to(output.device)  # Shape: (num_cls,)

    # Define FocalLoss with custom weights
    loss_fn = FocalLoss(
        weight=weights,  # Apply computed weights
        to_onehot_y=False,  # Convert target to one-hot encoding
        gamma=0.0,  # Set gamma=0 to disable focusing (pure cross-entropy)
        reduction="mean",  # Mean reduction (same as your original function)
    )

    # Compute loss
    loss = loss_fn(output, target)
    return loss


def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print('#' * 100)
        ecart = (128 - (xmax - xmin)) // 2
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax

def crop(vol):
    """
    Crop a 5D tensor (B, C, H, W, D) to the smallest bounding box that contains all nonzero elements,
    while ensuring a minimum size of 128 per dimension.
    """
    assert len(vol.shape) == 5  # Ensure input has shape (B, C, H, W, D)
    B, C, H, W, D = vol.shape
    cropped_regions = []
    
    for b in range(B):
        vol_b = vol[b]  # Shape (C, H, W, D)
        vol_b = torch.amax(vol_b, dim=0)  # Max over the channel dimension, shape (H, W, D)
        
        nonzero_indices = torch.nonzero(vol_b, as_tuple=True)
        
        if len(nonzero_indices[0]) == 0:
            # If the volume is completely empty, return full dimensions
            cropped_regions.append((0, H, 0, W, 0, D))
            continue
        
        x_min, x_max = sup_128(torch.min(nonzero_indices[0]).item(), torch.max(nonzero_indices[0]).item())
        y_min, y_max = sup_128(torch.min(nonzero_indices[1]).item(), torch.max(nonzero_indices[1]).item())
        z_min, z_max = sup_128(torch.min(nonzero_indices[2]).item(), torch.max(nonzero_indices[2]).item())
        
        cropped_regions.append((x_min, x_max, y_min, y_max, z_min, z_max))
    
    return cropped_regions

def normalize(vol):
    """
    Normalize a 5D tensor (B, C, H, W, D) independently for each batch element and channel.
    """
    assert len(vol.shape) == 5  # Ensure input has shape (B, C, H, W, D)
    B, C, H, W, D = vol.shape
    
    for b in range(B):
        for k in range(C):
            x = vol[b, k, ...]
            mask = x > 0  # Compute mask for non-zero elements
            if mask.sum() > 0:  # Avoid division by zero
                y = x[mask]
                x = (x - y.mean()) / (y.std() + 1e-8)  # Add small epsilon to avoid division by zero
                vol[b, k, ...] = x
    
    return vol