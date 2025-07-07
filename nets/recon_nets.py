import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureReconstructor(nn.Module):
    """
    Network that reconstructs complete modality features from incomplete/missing modality features.
    
    Input: Feature vector with missing modalities (192,4,4,4)
    Output: Reconstructed complete feature vector (192,4,4,4)
    """
    def __init__(self, feature_dim=192):
        super(FeatureReconstructor, self).__init__()
        
        # Feature extraction block
        self.encoder = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_dim*2, feature_dim*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim*2),
            nn.LeakyReLU(0.2)
        )
        
        # Self-attention mechanism
        self.query_conv = nn.Conv3d(feature_dim*2, feature_dim//2, kernel_size=1)
        self.key_conv = nn.Conv3d(feature_dim*2, feature_dim//2, kernel_size=1)
        self.value_conv = nn.Conv3d(feature_dim*2, feature_dim*2, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Feature refinement block
        self.decoder = nn.Sequential(
            nn.Conv3d(feature_dim*2, feature_dim*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_dim*2, feature_dim, kernel_size=3, padding=1)
        )
        
        # Residual connection
        self.residual = nn.Conv3d(feature_dim, feature_dim, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: Input feature tensor (B, feature_dim, D, H, W)
        Returns:
            Reconstructed feature tensor (B, feature_dim, D, H, W)
        """
        # Initial feature extraction
        features = self.encoder(x)
        
        # Self-attention mechanism
        batch_size, C, D, H, W = features.size()
        
        # Compute query, key, value projections
        proj_query = self.query_conv(features).view(batch_size, -1, D*H*W).permute(0, 2, 1)  # B x (D*H*W) x C'
        proj_key = self.key_conv(features).view(batch_size, -1, D*H*W)  # B x C' x (D*H*W)
        
        # Compute attention map
        energy = torch.bmm(proj_query, proj_key)  # B x (D*H*W) x (D*H*W)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        proj_value = self.value_conv(features).view(batch_size, -1, D*H*W)  # B x C x (D*H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)
        
        # Apply weighted attention
        attended_features = self.gamma * out + features
        
        # Decode features
        decoded = self.decoder(attended_features)
        
        # Add residual connection
        residual = self.residual(x)
        output = decoded + residual
        
        return output


class MRIFeatureReconstructor(nn.Module):
    """
    Main reconstruction network for brain MRI features.
    """
    def __init__(self, feature_dim=192):
        super(MRIFeatureReconstructor, self).__init__()
        
        # Main reconstruction network
        self.reconstructor = FeatureReconstructor(feature_dim)
        
        # Final refinement
        self.refinement = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor containing features with missing modalities (B, 192, 4, 4, 4)
        Returns:
            Reconstructed feature tensor (B, 192, 4, 4, 4)
        """
        # Apply the main reconstructor
        reconstructed = self.reconstructor(x)
        
        # Final refinement
        output = self.refinement(reconstructed)
        
        return output