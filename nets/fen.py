import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import ResidualUnit

class TransformerSelfAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape to (B, N, C) for attention
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, D, H, W)  # Reshape back
        return attn_out

class FeatureEmbeddingNetwork(nn.Module):
    def __init__(self, in_channels=192, out_channels=192, num_heads=8):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 192, kernel_size=3, padding=1)
        self.res_block = ResidualUnit(192, 192, kernel_size=3, strides=1, norm="instance")
        self.transformer_attn = TransformerSelfAttention(embed_dim=192, num_heads=num_heads)
        self.conv2 = nn.Conv3d(192, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block(x)
        attn_out = self.transformer_attn(x)  # Transformer-based Self-Attention
        x = x + attn_out  # Add refined attention output
        x = self.conv2(x)
        x = self.norm(x)
        return self.relu(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)


class CosineContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive):
        # Normalize feature vectors
        anchor = F.normalize(anchor, p=2, dim=1)  # L2 normalization
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = torch.sum(anchor * positive, dim=1)  # Cosine similarity
        loss = 1 - cosine_sim.mean()  # Minimize distance (maximize similarity)
        
        return loss