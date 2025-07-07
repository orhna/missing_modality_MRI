import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from .denoising_diffusion_pytorch.denoising_diffusion_pytorch_3D import Unet3D, GaussianDiffusion3D
from nets.multimodal_swinunetr_shard import Multimodal_SwinUNETR_shard
from nets.multimodal_swinunetr import Multimodal_SwinUNETR
import random

class mm_3Dldm(torch.nn.Module):

    def __init__(self,
                config,
                device_list = None):
        super(mm_3Dldm, self).__init__()

        self.config =config
        self.device_list = device_list
        self.shard=True if len(self.device_list)>1 else False
        self.generation = config.generation_mode
        self.bs = config.train_batch_size
        self.fs = config.feature_size

        self.sim_missing = False
        self.kl=False
        self.aux = config.aux

        self.image_size = 4
    
        if self.fs == 12:
            self.mean_features = torch.load("/mnt/disk1/hjlee/orhun/repo/thesis/mean_features_mm12_sd_ds_768_4_4_4.pt")
        elif self.fs == 24:
            self.mean_features = torch.load("/mnt/disk1/hjlee/orhun/repo/thesis/mean_features_1536_4_4_4.pt")
    
        # feature extraction related
        if not self.shard:
            self.swinunetr = Multimodal_SwinUNETR(
                img_size=(128, 128, 128),
                in_channels=1, 
                out_channels=config.output_channel,
                feature_size=self.fs,
                cross_attention=config.cross_attention,
                deep_supervision=config.deep_supervision,
                t1c_spec=config.t1c_spec,
                sep_dec=config.sep_dec,
                tp_conv=config.tp_conv).to(device_list[0])

            # diffusion related
            self.unet = Unet3D(
                    dim=config.ldm_dim, # initial layer dim
                    dim_mults=config.ldm_dim_mults,
                    channels=self.fs *16 *4, # initial input channels
                    self_condition = False
                    ).to(device_list[0])

            self.diffusion = GaussianDiffusion3D(
                model=self.unet,
                image_size=self.image_size,
                timesteps=config.timesteps,
                loss_type="l2",
                objective="pred_x0", # pred_x0, pred_noise
                beta_schedule="cosine").to(device_list[0])
            
            if config.recon:
                self.downsample = torch.nn.Conv3d(4*self.fs*16,2*self.fs*16,1).to(self.device_list[0])
                self.upsample = torch.nn.Conv3d(2*self.fs*16,4*self.fs*16,1).to(self.device_list[0])
            self.device_list.append(self.device_list[0])

        else:
            self.swinunetr = Multimodal_SwinUNETR_shard(
                img_size=(128, 128, 128),
                in_channels=1, 
                out_channels=config.output_channel,
                device_list=self.device_list,
                feature_size=self.fs,
                cross_attention=config.cross_attention,
                deep_supervision=config.deep_supervision,
                t1c_spec=config.t1c_spec,
                sep_dec=config.sep_dec)
            
            # diffusion related
            self.unet = Unet3D(
                    dim=config.ldm_dim,
                    dim_mults=config.ldm_dim_mults,
                    channels=self.fs *16 *4,
                    self_condition = False
                    ).to(self.device_list[1])
            self.diffusion = GaussianDiffusion3D(
                model=self.unet,
                image_size=self.image_size,
                timesteps=config.timesteps,
                loss_type="l1",
                objective="pred_noise"
                ).to(self.device_list[1])
            
            self.downsample = torch.nn.Conv3d(4*self.fs*16,2*self.fs*16,1).to(self.device_list[1])
            self.upsample = torch.nn.Conv3d(4*self.fs*16,8*self.fs*16,1).to(self.device_list[1])
        
        self.swinunetr.is_training=False

    def forward(self, complete_modality_image):

        complete_modality_features = extract_complete_modality_features(self.swinunetr,
                                                                        self.device_list,
                                                                        complete_modality_image)
        """
        missing_modality_image = drop_modality_image_channel(complete_modality_image,
                                                            self.config.drop_mode,
                                                            [0],
                                                            [1,2,3],
                                                            self.device_list)
            
        missing_modality_features = extract_complete_modality_features(self.swinunetr,
                                                                        self.device_list,
                                                                        missing_modality_image)
        """
        
        if self.sim_missing and random.random() < 0.5:
            complete_modality_features = drop_modality_feature_channel(complete_modality_features,
                                                                      self.generation,
                                                                      [0],
                                                                      self.fs,
                                                                      self.mean_features)

        #c_m_features = torch.cat([missing_modality_features,complete_modality_features], dim=1)

        diff_loss = self.diffusion(complete_modality_features)

        missing_modality_features= drop_modality_feature_channel(complete_modality_features,
                                                                    self.config.generation_mode,
                                                                    [0],
                                                                    self.fs,
                                                                    self.mean_features)
        
        generated_c_mod_features= self.diffusion.diff_sample(missing_modality_features)
        
        gc_m_mse_loss = F.mse_loss(generated_c_mod_features, complete_modality_features)


        if self.kl:
            kl_loss = kl_divergence_loss(complete_modality_features, generated_c_mod_features)

        if self.aux:
            missing_modality_image = drop_modality_image_channel(complete_modality_image,
                                                        self.config.drop_mode,
                                                        [0],
                                                        [1,2,3],
                                                        self.device_list)
            c_f_m_generated_logits = self.swinunetr(missing_modality_image, bottleneck=[generated_c_mod_features,[0]])

        
        return diff_loss , gc_m_mse_loss, c_f_m_generated_logits#, kl_loss

    def load_swinunetr_weights(self, checkpoint_path):

        # from my saved pts
        if self.shard:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device_list[0]))["model_state_dict"]
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device_list[0]))["model_state_dict"]

        self.swinunetr.load_state_dict(checkpoint)
        print("model weights loaded successfully!")
        for param in self.swinunetr.parameters():
            param.requires_grad = False
        print("model weights are frozen")

def extract_complete_modality_features(model,device_list,complete_modality_image):
        with torch.no_grad():
            c_hidden_states_out_m1 = model.swinViT_1(complete_modality_image[:,0:1,:,:].to(device_list[0]), normalize=True)[4]
            c_hidden_states_out_m2 = model.swinViT_2(complete_modality_image[:,1:2,:,:].to(device_list[0]), normalize=True)[4]
            c_hidden_states_out_m3 = model.swinViT_3(complete_modality_image[:,2:3,:,:].to(device_list[1]), normalize=True)[4]
            c_hidden_states_out_m4 = model.swinViT_4(complete_modality_image[:,3:4,:,:].to(device_list[1]), normalize=True)[4]
            c_dec4_m1 = model.encoder10_1(c_hidden_states_out_m1)
            c_dec4_m2 = model.encoder10_2(c_hidden_states_out_m2)
            c_dec4_m3 = model.encoder10_3(c_hidden_states_out_m3).to(device_list[0])
            c_dec4_m4 = model.encoder10_4(c_hidden_states_out_m4).to(device_list[0])

            complete_modality_features  = torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1) # ([B, 4 * fs*16, 4, 4, 4])
            
        return complete_modality_features

def drop_modality_image_channel(_input, method, idx_to_drop, remaining_modalities, device_list):

    missing_modality_image = _input.clone()

    if method == "whole_mean":
        mean_value = torch.mean(_input)
        missing_modality_image[:,idx_to_drop,...] = mean_value
    elif method == "modality_mean":
        mean_value = torch.mean(_input[:, remaining_modalities, :, :, :],dim=1,keepdim=True)
        missing_modality_image[:,idx_to_drop,...] = mean_value
    elif method == "zero":
        missing_modality_image[:,idx_to_drop,...] = 0

    missing_modality_image = missing_modality_image.to(device_list[0])

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

def kl_divergence_loss(real_features, generated_features):
    
    real_mean = real_features.mean(dim=(2, 3, 4))  # Compute mean across spatial dims
    real_var = real_features.std(dim=(2, 3, 4))  # Compute variance

    gen_mean = generated_features.mean(dim=(2, 3, 4))
    gen_var = generated_features.std(dim=(2, 3, 4))


    log_std1 = torch.log(gen_var + 1e-10)  
    log_std2 = torch.log(real_var + 1e-10)
    kl_loss = (log_std2 - log_std1 + (gen_var**2 + (gen_mean - real_mean)**2) / (2 * real_var**2) - 0.5).mean()

    return kl_loss

#  dynamic wrapper for sliding window inference 
class mm_3DLDMWrapper:
    def __init__(self, model):
        self.ldm_model = model
        self.channels_to_drop = 0
        
    def __call__(self, x):

        with torch.no_grad():

            missing_modality_image, complete_modality_image = torch.chunk(x, 2, dim=1)

            m_hidden_states_out_m1 = self.ldm_model.swinunetr.swinViT_1(missing_modality_image[:,0:1,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
            m_hidden_states_out_m2 = self.ldm_model.swinunetr.swinViT_2(missing_modality_image[:,1:2,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
            m_hidden_states_out_m3 = self.ldm_model.swinunetr.swinViT_3(missing_modality_image[:,2:3,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
            m_hidden_states_out_m4 = self.ldm_model.swinunetr.swinViT_4(missing_modality_image[:,3:4,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
            m_dec4_m1 = self.ldm_model.swinunetr.encoder10_1(m_hidden_states_out_m1).to(self.ldm_model.device_list[1])
            m_dec4_m2 = self.ldm_model.swinunetr.encoder10_2(m_hidden_states_out_m2).to(self.ldm_model.device_list[1])
            m_dec4_m3 = self.ldm_model.swinunetr.encoder10_3(m_hidden_states_out_m3)
            m_dec4_m4 = self.ldm_model.swinunetr.encoder10_4(m_hidden_states_out_m4)
            
            missing_modality_features = torch.cat((m_dec4_m1, m_dec4_m2, m_dec4_m3, m_dec4_m4), dim=1) # ([B, 4 * 768, 8, 8])
            
            missing_modality_features = drop_modality_feature_channel(missing_modality_features,
                                                                      self.ldm_model.config.generation_mode,
                                                                      self.channels_to_drop,
                                                                      self.ldm_model.fs,
                                                                      self.ldm_model.mean_features)
            """
            # extracting complete modality features
            c_hidden_states_out_m1 = self.ldm_model.swinunetr.swinViT_1(complete_modality_image[:,0:1,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
            c_hidden_states_out_m2 = self.ldm_model.swinunetr.swinViT_2(complete_modality_image[:,1:2,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
            c_hidden_states_out_m3 = self.ldm_model.swinunetr.swinViT_3(complete_modality_image[:,2:3,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
            c_hidden_states_out_m4 = self.ldm_model.swinunetr.swinViT_4(complete_modality_image[:,3:4,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
            c_dec4_m1 = self.ldm_model.swinunetr.encoder10_1(c_hidden_states_out_m1).to(self.ldm_model.device_list[1])
            c_dec4_m2 = self.ldm_model.swinunetr.encoder10_2(c_hidden_states_out_m2).to(self.ldm_model.device_list[1])
            c_dec4_m3 = self.ldm_model.swinunetr.encoder10_3(c_hidden_states_out_m3)
            c_dec4_m4 = self.ldm_model.swinunetr.encoder10_4(c_hidden_states_out_m4)
            
            complete_modality_features = torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1) # ([B, 4 * 768, 8, 8])
            complete_modality_features = complete_modality_features.to(self.ldm_model.device_list[1])
            
            c_generated_features = self.ldm_model.diffusion.diff_sample(complete_modality_features).to(self.ldm_model.device_list[0])
            m_w_c_generated_logits = self.ldm_model.swinunetr(missing_modality_image, bottleneck=c_generated_features)
            """ 

            if self.ldm_model.config.recon:
                missing_modality_features = self.ldm_model.downsample(missing_modality_features) 
            
            missing_modality_features = missing_modality_features.to(self.ldm_model.device_list[1])
            
            #temp_c_modality_features = self.ldm_model.mean_features
            #temp_c_modality_features = temp_c_modality_features.to(self.ldm_model.device_list[1])

            #tc_m_features= torch.cat([missing_modality_features,temp_c_modality_features], dim=1)

            generated_features = self.ldm_model.diffusion.diff_sample(missing_modality_features).to(self.ldm_model.device_list[0])
            
            if self.ldm_model.config.recon:
                generated_features = self.ldm_model.upsample(generated_features)

            c_f_m_generated_logits = self.ldm_model.swinunetr(missing_modality_image, bottleneck=[generated_features,self.channels_to_drop])
            missing_img_logits = self.ldm_model.swinunetr(missing_modality_image)

        return [missing_img_logits, c_f_m_generated_logits]