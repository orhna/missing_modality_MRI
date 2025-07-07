import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, Convolution
from monai.networks.layers import Norm

class CustomUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomUNet, self).__init__()
        
        self.in_channels = int(in_channels/4)
        self.out_channels = out_channels

        self.norm=Norm.INSTANCE

        self.cross_attn = CrossAttentionFusion()
        self.initial_feature_size = 32

        # Encoder layers
        self.encoder1_m1 = ResidualUnit(spatial_dims=3, in_channels=1, out_channels=32, strides=2, kernel_size=3, norm=self.norm)
        self.encoder2_m1 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3, norm=self.norm)
        self.encoder3_m1 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3, norm=self.norm)
        self.encoder4_m1 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=256, strides=2, kernel_size=3, norm=self.norm)
        self.bottleneck_m1 = ResidualUnit(spatial_dims=3, in_channels=256, out_channels=512, strides=1, kernel_size=3, norm=self.norm)

        self.encoder1_m2 = ResidualUnit(spatial_dims=3, in_channels=1, out_channels=32, strides=2, kernel_size=3, norm=self.norm)
        self.encoder2_m2 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3, norm=self.norm)
        self.encoder3_m2 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3, norm=self.norm)
        self.encoder4_m2 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=256, strides=2, kernel_size=3, norm=self.norm)
        self.bottleneck_m2 = ResidualUnit(spatial_dims=3, in_channels=256, out_channels=512, strides=1, kernel_size=3, norm=self.norm)

        self.encoder1_m3 = ResidualUnit(spatial_dims=3, in_channels=1, out_channels=32, strides=2, kernel_size=3, norm=self.norm)
        self.encoder2_m3 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3, norm=self.norm)
        self.encoder3_m3 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3, norm=self.norm)
        self.encoder4_m3 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=256, strides=2, kernel_size=3, norm=self.norm)
        self.bottleneck_m3 = ResidualUnit(spatial_dims=3, in_channels=256, out_channels=512, strides=1, kernel_size=3, norm=self.norm)

        self.encoder1_m4 = ResidualUnit(spatial_dims=3, in_channels=1, out_channels=32, strides=2, kernel_size=3, norm=self.norm)
        self.encoder2_m4 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3, norm=self.norm)
        self.encoder3_m4 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3, norm=self.norm)
        self.encoder4_m4 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=256, strides=2, kernel_size=3, norm=self.norm)
        self.bottleneck_m4 = ResidualUnit(spatial_dims=3, in_channels=256, out_channels=512, strides=1, kernel_size=3, norm=self.norm)

        # dim red
        self.c_red_1 = Convolution(spatial_dims=3,
                                    in_channels= 4 * self.initial_feature_size,
                                    out_channels= self.initial_feature_size,
                                    strides=1,
                                    kernel_size=1, norm=self.norm,
                                    dropout=0.2)
        self.c_red_2 = Convolution(spatial_dims=3,
                                    in_channels= 8 * self.initial_feature_size,
                                    out_channels= 2 * self.initial_feature_size,
                                    strides=1,
                                    kernel_size=1, norm=self.norm,
                                    dropout=0.2)
        self.c_red_3 = Convolution(spatial_dims=3,
                                    in_channels= 16 * self.initial_feature_size,
                                    out_channels= 4 * self.initial_feature_size,
                                    strides=1,
                                    kernel_size=1, norm=self.norm,
                                    dropout=0.2)                                       
        self.c_red_4 = Convolution(spatial_dims=3,
                                    in_channels= 32 * self.initial_feature_size,
                                    out_channels= 8 * self.initial_feature_size,
                                    strides=1,
                                    kernel_size=1, norm=self.norm,
                                    dropout=0.2)
        self.c_red_5 = Convolution(spatial_dims=3,
                                    in_channels= 64 * self.initial_feature_size,
                                    out_channels= 16 * self.initial_feature_size,
                                    strides=1,
                                    kernel_size=1, norm=self.norm,
                                    dropout=0.2)
        # Decoder layers
        self.upconv4 = Convolution(spatial_dims=3, in_channels=768, out_channels=128, strides=2, kernel_size=3, norm=self.norm, is_transposed=True)
        self.decoder4 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=128, strides=1, kernel_size=3, norm=self.norm)
        
        self.upconv3 = Convolution(spatial_dims=3, in_channels=256, out_channels=64, strides=2, kernel_size=3, norm=self.norm, is_transposed=True)
        self.decoder3 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=64, strides=1, kernel_size=3, norm=self.norm)
        
        self.upconv2 = Convolution(spatial_dims=3, in_channels=128, out_channels=32, strides=2, kernel_size=3, norm=self.norm, is_transposed=True)
        self.decoder2 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=32, strides=1, kernel_size=3, norm=self.norm)
        
        self.upconv1 = Convolution(spatial_dims=3, in_channels=64, out_channels=3, strides=2, kernel_size=3, norm=self.norm,is_transposed=True)
        self.decoder1 = ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, strides=1, kernel_size=3, norm=self.norm)

    def forward(self, x):
        
        # splitting 4 channels to 1 for each modality
        x_mods = torch.chunk(x, 4, dim=1)

        # modality 1 encoding 
        m1_e1 = self.encoder1_m1(x_mods[0])
        m1_e2 = self.encoder2_m1(m1_e1)
        m1_e3 = self.encoder3_m1(m1_e2)
        m1_e4 = self.encoder4_m1(m1_e3)
        m1_bottleneck = self.bottleneck_m1(m1_e4)

        # modality 2 encoding 
        m2_e1 = self.encoder1_m2(x_mods[1])
        m2_e2 = self.encoder2_m2(m2_e1)
        m2_e3 = self.encoder3_m2(m2_e2)
        m2_e4 = self.encoder4_m2(m2_e3)
        m2_bottleneck = self.bottleneck_m1(m2_e4)

        # modality 3 encoding 
        m3_e1 = self.encoder1_m3(x_mods[2])
        m3_e2 = self.encoder2_m3(m3_e1)
        m3_e3 = self.encoder3_m3(m3_e2)
        m3_e4 = self.encoder4_m3(m3_e3)
        m3_bottleneck = self.bottleneck_m3(m3_e4)

        # modality 4 encoding 
        m4_e1 = self.encoder1_m4(x_mods[3])
        m4_e2 = self.encoder2_m4(m4_e1)
        m4_e3 = self.encoder3_m4(m4_e2)
        m4_e4 = self.encoder4_m4(m4_e3)
        m4_bottleneck = self.bottleneck_m4(m4_e4)

        # applying cross attention to deepest level
        #fused_bottleneck_features = self.cross_attn(m1_bottleneck,m2_bottleneck,m3_bottleneck,m4_bottleneck) #[1, 512, 8, 8, 8]

        # using 1x1 convolutions on other levels to reduce dimensionality
        enc1 = self.c_red_1(torch.cat([m1_e1, m2_e1, m3_e1, m4_e1], dim=1))
        enc2 = self.c_red_2(torch.cat([m1_e2, m2_e2, m3_e2, m4_e2], dim=1))
        enc3 = self.c_red_3(torch.cat([m1_e3, m2_e3, m3_e3, m4_e3], dim=1))
        enc4 = self.c_red_4(torch.cat([m1_e4, m2_e4, m3_e4, m4_e4], dim=1))
        fused_bottleneck_features = self.c_red_5(torch.cat([m1_bottleneck,m2_bottleneck,m3_bottleneck,m4_bottleneck], dim=1))

        # Decoding path with skip connections
        d4 = self.upconv4(torch.cat([fused_bottleneck_features, enc4], dim=1))
        d4 = self.decoder4(d4)

        d3 = self.upconv3(torch.cat([d4, enc3], dim=1))
        d3 = self.decoder3(d3)

        d2 = self.upconv2(torch.cat([d3, enc2], dim=1))
        d2 = self.decoder2(d2)

        d1 = self.upconv1(torch.cat([d2, enc1], dim=1))
        d1 = self.decoder1(d1)
        
        return d1

  
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, enc1, enc2, enc3, enc4):
        # Reshape from (B, C, D, H, W) -> (B, Seq, C)
        B, C, D, H, W = enc1.shape
        seq_len = D * H * W
        enc1 = enc1.view(B, C, seq_len).permute(0, 2, 1)  # Query
        enc2 = enc2.view(B, C, seq_len).permute(0, 2, 1)  # Key
        enc3 = enc3.view(B, C, seq_len).permute(0, 2, 1)  # Key
        enc4 = enc4.view(B, C, seq_len).permute(0, 2, 1)  # Key

        # Stack keys & values
        keys = torch.cat([enc2, enc3, enc4], dim=1)  # (B, 3*Seq, C)
        values = keys.clone()

        # Apply cross-attention (Query=enc1, Key=keys, Value=values)
        attn_output, _ = self.attn(enc1, keys, values)  # (B, Seq, C)

        # Add residual connection and normalize
        fused = self.norm(enc1 + attn_output)

        # Reshape back to (B, C, D, H, W)
        fused = fused.permute(0, 2, 1).view(B, C, D, H, W)

        return fused

"""
class MultiEncoderResUNet(torch.nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=4,  # T1, T1ce, T2, FLAIR
        out_channels=3,  # WT, TC, ET
        init_filters=32,
        depth=4,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.depth = depth

        # Build 4 separate encoders (1 per modality)
        self.encoders = torch.nn.ModuleList([
            self._build_encoder(in_channels=1, init_filters=init_filters) 
            for _ in range(in_channels)
        ])

        # Build shared decoder
        self.decoder = self._build_decoder(init_filters * (2 ** (depth - 1)) * in_channels)

        # Final convolution
        self.final_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=init_filters,
            out_channels=out_channels,
            kernel_size=1,
            act=None,
            norm=None,
            adn_ordering="",
        )

    def _build_encoder(self, in_channels, init_filters):
        encoder = torch.nn.ModuleList()
        filters = init_filters
        for _ in range(self.depth):
            # Downsample block (stride=2)
            block = torch.nn.Sequential(
                ResidualUnit(
                    spatial_dims=self.spatial_dims,
                    in_channels=in_channels,
                    out_channels=filters,
                    subunits=2,
                    adn_ordering="NDA",
                    act=Act.LEAKYRELU,
                    norm=Norm.INSTANCE,
                    dropout=0.1,
                    kernel_size=3,
                    strides=1,
                ),
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=filters,
                    out_channels=filters,
                    strides=2,  # Downsample
                    kernel_size=3,
                    act=None,
                    norm=None,
                ),
            )
            encoder.append(block)
            in_channels = filters
            filters *= 2  # Double filters at each level
        return encoder
    
    def _build_decoder(self, bottleneck_channels):
        decoder = torch.nn.ModuleList()
        filters = bottleneck_channels // 2  # Start from fused bottleneck
        for _ in range(self.depth - 1):
            # Upsample block
            up = UpSample(
                spatial_dims=self.spatial_dims,
                in_channels=filters,
                out_channels=filters // 2,
                scale_factor=2,
                mode="deconv", # nontrainable
                align_corners=False,
            )
            res_unit = ResidualUnit(
                spatial_dims=self.spatial_dims,
                in_channels=filters,
                out_channels=filters // 2,
                subunits=2,
                adn_ordering="NDA",
                act=Act.LEAKYRELU,
                norm=Norm.INSTANCE,
                kernel_size=3,
            )
            decoder.append(torch.nn.ModuleList([up, res_unit]))
            filters = filters // 2
        return decoder
    
    
    def forward(self, x):
        # Split input into 4 modalities [B,1,H,W,D] each
        modalities = torch.split(x, 1, dim=1)
        
        # Encoder passes (save features at each level)
        encoder_features = []
        #for enc in self.encoders:
        for i in range(len(self.encoders)):
            enc = self.encoders[i]
            features = []
            x_enc = modalities[i]
            for i, block in enumerate(enc):
                x_enc = block(x_enc)
                features.append(x_enc)  # Save pre-downsample features
            encoder_features.append(features)

        # Concatenate at bottleneck (last layer of each encoder)
        fused = torch.cat([feats[-1] for feats in encoder_features], dim=1)

        # Decoder pass
        x_dec = fused
        for i, (up, res_unit) in enumerate(self.decoder):
            x_dec = up(x_dec)
            # Add skip connection from corresponding encoder level
            skip = torch.cat([feats[-(i+2)] for feats in encoder_features], dim=1)
            x_dec = torch.cat([x_dec, skip], dim=1)
            x_dec = res_unit(x_dec)

        return self.final_conv(x_dec)
        
"""