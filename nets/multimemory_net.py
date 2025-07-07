import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.multimodal_swinunetr import Multimodal_SwinUNETR
from monai.transforms import NormalizeIntensity

class ModalityMemory(nn.Module):
    def __init__(self, device, memory_size=100, modality_dim=192*4*4*4,):  # 192*4*4*4
        super(ModalityMemory, self).__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, modality_dim), requires_grad=True).to(device) 

    def retrieve(self, key):
        """
        Args:
            key: Tensor of shape (B, 3072)
        Returns:
            Tensor of shape (B, 3072)
        """
        key = key.reshape(key.shape[0],-1)
        key = F.normalize(key, dim=1)
        memory_norm = F.normalize(self.memory, dim=1)
        #print("key.shape",key.shape)
        #print("memory_norm.shape",memory_norm.shape)

        sim = torch.matmul(key, memory_norm.t())  # (B, M)
        attn = F.softmax(sim, dim=1)
        retrieved = torch.matmul(attn, self.memory)  # (B, 3072)
        return retrieved


class multimemoryNet(torch.nn.Module):

    def __init__(self,
                config,
                device):
        super(multimemoryNet, self).__init__()

        self.config =config
        self.device = device
        self.generation = config.generation_mode
        self.bs = config.train_batch_size
        self.fs = config.feature_size
        self.diff_on = config.diff_on
        self.channels_to_drop = 0
    
        if self.fs == 12:
            self.mean_features = torch.load("/mnt/disk1/hjlee/orhun/repo/thesis/mean_features_mm12_sd_ds_separate_768_4_4_4.pt", map_location=self.device)
        elif self.fs == 24:
            self.mean_features = torch.load("/mnt/disk1/hjlee/orhun/repo/thesis/mean_features_1536_4_4_4.pt", map_location=self.device)
    
        self.multiplier = 1 if self.diff_on == "combined" else 4

        # feature extraction related
        self.swinunetr = Multimodal_SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=1, 
            out_channels=config.output_channel,
            feature_size=self.fs,
            deep_supervision=config.deep_supervision,
            sep_dec=config.sep_dec,
            tp_conv=config.tp_conv,
            dec_upsample=config.dec_upsample).to(self.device)

        # memory related

        self.memory_size = 50 #config.memory_size
        self.channels_per_modality = 192
        self.modality_size = self.channels_per_modality * 4 * 4 * 4  # each modality contributes 192x4x4x4
        self.num_modalities = 4
        
        
        #self.multimemoryKeyEncoder = Transformer(embedding_dim=512, depth=1, heads=8, mlp_dim=4096, n_levels=4).to(self.device)

        self.multimemoryKeyEncoder = Transformer3D(
            in_channels=768,   # input channel dimension
            out_channels=192,  # desired output channel dimension
            depth=4,           # number of transformer blocks
            heads=8,           # number of attention heads
            mlp_dim=512,       # hidden dim in the feedforward network
            dropout_rate=0.1   # dropout rate
        ).to(self.device)

        self.memories = nn.ModuleList([
            ModalityMemory(self.device, 100, self.modality_size, )
            for _ in range(self.num_modalities)
        ])
        
        
        self.swinunetr.is_training=False
        self.swinunetr.diff_on = self.diff_on

    def forward(self, x, available_modalities):
        
        """
        x: Tensor (B, 768, 4, 4)
        available_modalities: list of indices (e.g. [0, 1] means FLAIR and T1C available)
        """
        B, C, H, W, D = x.shape
        x_split = torch.chunk(x, self.num_modalities, dim=1)  # list of (B, 192, 4, 4, 4)

        # Generate key using transformer
        key = self.multimemoryKeyEncoder(x)
        output_chunks = []

        for i in range(self.num_modalities):
            if i in available_modalities:
                output_chunks.append(x_split[i])  # use existing
            else:
                retrieved = self.memories[i].retrieve(key)
                retrieved = retrieved.view(B, self.channels_per_modality, H, W, D)
                output_chunks.append(retrieved)

        out = torch.cat(output_chunks, dim=1)  # (B, 768, 4, 4, 4)
        return out
        
    
    def load_swinunetr_weights(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=self.device)["model_state_dict"]
        #checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.swinunetr.load_state_dict(checkpoint)
        print("model weights loaded successfully!")
        for param in self.swinunetr.parameters():
            param.requires_grad = False
        print("model weights are frozen")

def extract_complete_modality_features(model, complete_modality_image, diff_on):
        with torch.no_grad():
            c_hidden_states_out_m1 = model.swinViT_1(complete_modality_image[:,0:1,:,:], normalize=True)[4]
            c_hidden_states_out_m2 = model.swinViT_2(complete_modality_image[:,1:2,:,:], normalize=True)[4]
            c_hidden_states_out_m3 = model.swinViT_3(complete_modality_image[:,2:3,:,:], normalize=True)[4]
            c_hidden_states_out_m4 = model.swinViT_4(complete_modality_image[:,3:4,:,:], normalize=True)[4]
            c_dec4_m1 = model.encoder10_1(c_hidden_states_out_m1)
            c_dec4_m2 = model.encoder10_2(c_hidden_states_out_m2)
            c_dec4_m3 = model.encoder10_3(c_hidden_states_out_m3)
            c_dec4_m4 = model.encoder10_4(c_hidden_states_out_m4)

            if diff_on == "combined":
                complete_modality_features = model.channel_reduction_6(torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1))
            elif diff_on == "separate":
                complete_modality_features  = torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1) # ([B, 4 * fs*16, 4, 4, 4])
    
        return complete_modality_features


def drop_modality_image_channel(_input, method, idx_to_drop, remaining_modalities):

    missing_modality_image = _input.clone()

    if method == "whole_mean":
        mean_value = torch.mean(_input)
        missing_modality_image[:,idx_to_drop,...] = mean_value
    elif method == "modality_mean":
        mean_value = torch.mean(_input[:, remaining_modalities, :, :, :],dim=1,keepdim=True)
        missing_modality_image[:,idx_to_drop,...] = mean_value
    elif method == "zero":
        missing_modality_image[:,idx_to_drop,...] = 0

    return missing_modality_image

def drop_modality_feature_channel(_input, method, idx_to_drop, fs, pc_mean_features):

    for idx in idx_to_drop:
        start_i= idx *fs*16
        end_i= start_i+ fs*16
        
        if method == "whole_mean":
            mean_value = torch.mean(_input)
            _input[:,start_i:end_i] = mean_value
        elif method == "zero":
            _input[:,start_i:end_i] = 0
        elif method == "gnoise":
            _input[:, start_i:end_i, :, :, :] = torch.randn_like(_input[:, start_i:end_i, :, :, :])
        elif method == "pcmean_features":
            _input[:, start_i:end_i] = pc_mean_features[:,start_i:end_i,...]

    return _input

#  dynamic wrapper for sliding window inference 
class multimemoryNetWrapper:
    def __init__(self, model):
        self.rec_model = model
        self.channels_to_drop = 0
        self.remaining_channels = 0
        self.normalize = NormalizeIntensity(nonzero=True, channel_wise=True)

    def __call__(self, x):

        missing_modality_image = x

        with torch.no_grad():

            m_hidden_states_out_m1 = self.rec_model.swinunetr.swinViT_1(missing_modality_image[:,0:1,:,:], normalize=True)[4]
            m_hidden_states_out_m2 = self.rec_model.swinunetr.swinViT_2(missing_modality_image[:,1:2,:,:], normalize=True)[4]
            m_hidden_states_out_m3 = self.rec_model.swinunetr.swinViT_3(missing_modality_image[:,2:3,:,:], normalize=True)[4]
            m_hidden_states_out_m4 = self.rec_model.swinunetr.swinViT_4(missing_modality_image[:,3:4,:,:], normalize=True)[4]
            m_dec4_m1 = self.rec_model.swinunetr.encoder10_1(m_hidden_states_out_m1)
            m_dec4_m2 = self.rec_model.swinunetr.encoder10_2(m_hidden_states_out_m2)
            m_dec4_m3 = self.rec_model.swinunetr.encoder10_3(m_hidden_states_out_m3)
            m_dec4_m4 = self.rec_model.swinunetr.encoder10_4(m_hidden_states_out_m4)
            
            missing_modality_features = torch.cat((m_dec4_m1, m_dec4_m2, m_dec4_m3, m_dec4_m4), dim=1) # ([B, 4 * 768, 8, 8])
            
            if self.rec_model.diff_on == "separate":
                missing_modality_features = drop_modality_feature_channel(missing_modality_features,
                                                                        self.rec_model.config.generation_mode,
                                                                        self.channels_to_drop,
                                                                        self.rec_model.fs,
                                                                        self.rec_model.mean_features)
            elif self.rec_model.diff_on == "combined":
                missing_modality_features = self.rec_model.swinunetr.channel_reduction_6(missing_modality_features)

            #missing_modality_features = self.normalize(missing_modality_features)
            recon_complete_features = self.rec_model( missing_modality_features, self.remaining_channels)
            c_f_m_generated_logits = self.rec_model.swinunetr(missing_modality_image, bottleneck=[recon_complete_features,self.channels_to_drop])
            missing_img_logits = self.rec_model.swinunetr(missing_modality_image)

        return [missing_img_logits, c_f_m_generated_logits]


# mmformer interformer

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


    def forward(self, x):
        for j in range(self.depth):
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x



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


class Transformer3D(nn.Module):
    def __init__(self, in_channels=768, out_channels=192, depth=4, heads=8, mlp_dim=512, dropout_rate=0.1):
        super().__init__()
        self.spatial_size = (4, 4, 4)
        self.seq_len = self.spatial_size[0] * self.spatial_size[1] * self.spatial_size[2]

        # Project to lower dimension if needed
        self.embedding_proj = nn.Linear(in_channels, out_channels)

        # Transformer backbone
        self.transformer = Transformer(
            embedding_dim=out_channels,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        B, C, D, H, W = x.shape  # B, 768, 4, 4, 4
        x = x.view(B, C, -1).permute(0, 2, 1)  # B, 64, 768
        x = self.embedding_proj(x)  # B, 64, 192
        x = self.transformer(x)  # B, 64, 192
        x = x.permute(0, 2, 1).view(B, -1, *self.spatial_size)  # B, 192, 4, 4, 4
        return x