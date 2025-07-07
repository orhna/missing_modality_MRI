import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from .denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from .denoising_diffusion_pytorch.denoising_diffusion_pytorch_3D import Unet3D, GaussianDiffusion3D
from nets.multimodal_swinunetr import Multimodal_SwinUNETR

class mm_finalldm(torch.nn.Module):

    def __init__(self,
                config,
                device = None):
        super(mm_finalldm, self).__init__()

        self.config =config
        self.device = device
        self.generation = config.generation_mode
        self.bs = config.train_batch_size
        self.fs = config.feature_size
        self.diff_domain = config.diff_domain
        
        if self.fs == 12:
            self.mean_features = torch.load("/mnt/disk1/hjlee/orhun/repo/thesis/mean_features_mm12_sd_ds_768_4_4_4.pt", map_location=self.device)
        elif self.fs == 24:
            self.mean_features = torch.load("/mnt/disk1/hjlee/orhun/repo/thesis/mean_features_1536_4_4_4.pt", map_location=self.device)
    
        # feature extraction related
        self.swinunetr = Multimodal_SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=1, 
            out_channels=config.output_channel,
            feature_size=self.fs,
            deep_supervision=config.deep_supervision,
            sep_dec=config.sep_dec).to(self.device)

        if self.diff_domain =="2D":
            # diffusion related
            self.image_size = 16
            self.unet = Unet(
                    dim=config.ldm_dim,
                    dim_mults=config.ldm_dim_mults,
                    channels=self.fs *16
                    ).to(self.device)
            
            self.diffusion = GaussianDiffusion(
                model=self.unet,
                image_size=self.image_size,
                timesteps=config.timesteps,
                objective = 'pred_x0', #{'pred_noise', 'pred_x0', 'pred_v'}
                beta_schedule = 'cosine' # {"linear","cosine","sigmoid"}
                ).to(self.device)
            
        elif self.diff_domain =="3D":
            # diffusion related
            self.image_size = 4 

            self.unet = Unet3D(
                    dim=config.ldm_dim, # initial layer dim
                    dim_mults=config.ldm_dim_mults,
                    channels=self.fs *16 *4, # initial input channels
                    ).to(self.device)
            self.diffusion = GaussianDiffusion3D(
                model=self.unet,
                image_size=self.image_size,
                timesteps=config.timesteps,
                loss_type="l2",
                objective="pred_x0", # pred_x0, pred_noise
                beta_schedule="linear" # {"linear","cosine"}
                ).to(self.device)

        if config.recon:
            self.downsample = torch.nn.Conv3d(4*self.fs*16,2*self.fs*16,1).to(self.device)
            self.upsample = torch.nn.Conv3d(2*self.fs*16,4*self.fs*16,1).to(self.device)

        self.swinunetr.is_training=False

    def forward(self, complete_modality_image):

        complete_modality_features = extract_complete_modality_features(self.swinunetr,
                                                                        complete_modality_image)
        
        B,C,H,W,D= complete_modality_features.shape
        
        if self.diff_domain =="2D":
            complete_modality_features = complete_modality_features.reshape(B,C//4,H*4,W*4)
            print("features are reshaped for 2D:",complete_modality_features.shape )

        diff_loss = self.diffusion(complete_modality_features)

        return diff_loss

    def load_swinunetr_weights(self, checkpoint_path):

        # from my saved pts
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))["model_state_dict"]

        self.swinunetr.load_state_dict(checkpoint)
        print("model weights loaded successfully!")
        for param in self.swinunetr.parameters():
            param.requires_grad = False
        print("model weights are frozen")

def extract_complete_modality_features(model,complete_modality_image):
        with torch.no_grad():
            c_hidden_states_out_m1 = model.swinViT_1(complete_modality_image[:,0:1,:,:], normalize=True)[4]
            c_hidden_states_out_m2 = model.swinViT_2(complete_modality_image[:,1:2,:,:], normalize=True)[4]
            c_hidden_states_out_m3 = model.swinViT_3(complete_modality_image[:,2:3,:,:], normalize=True)[4]
            c_hidden_states_out_m4 = model.swinViT_4(complete_modality_image[:,3:4,:,:], normalize=True)[4]
            c_dec4_m1 = model.encoder10_1(c_hidden_states_out_m1)
            c_dec4_m2 = model.encoder10_2(c_hidden_states_out_m2)
            c_dec4_m3 = model.encoder10_3(c_hidden_states_out_m3)
            c_dec4_m4 = model.encoder10_4(c_hidden_states_out_m4)

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

    missing_modality_image = missing_modality_image

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
