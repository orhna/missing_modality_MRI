# legacy code
# used in experiments that didn't yield decent results

import torch
import torch.nn.functional as F
from .denoising_diffusion_pytorch.denoising_diffusion_pytorch_3D import Unet3D, GaussianDiffusion3D
from nets.multimodal_swinunetr import Multimodal_SwinUNETR

PRECOMPUTED_MEAN_FEATURES_12 = ""
PRECOMPUTED_MEAN_FEATURES_24 = ""

class mm_3Dldm(torch.nn.Module):

    def __init__(self,
                config,
                device):
        super(mm_3Dldm, self).__init__()

        self.config =config
        self.device = device
        self.generation = config.generation_mode
        self.bs = config.train_batch_size
        self.fs = config.feature_size
        self.diff_on = config.diff_on

        self.image_size = 4
    
        if self.fs == 12:
            self.mean_features = torch.load(PRECOMPUTED_MEAN_FEATURES_12, map_location=self.device)
        elif self.fs == 24:
            self.mean_features = torch.load(PRECOMPUTED_MEAN_FEATURES_24, map_location=self.device)
    
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

        # diffusion related
        self.unet = Unet3D(
                dim=config.ldm_dim, # initial layer dim
                dim_mults=config.ldm_dim_mults,
                channels=self.fs *16 *4 if self.diff_on == "separate" else self.fs *16, # initial input channels
                ).to(self.device)
        self.diffusion = GaussianDiffusion3D(
            model=self.unet,
            image_size=self.image_size,
            timesteps=config.timesteps,
            objective="pred_x0", # pred_x0, pred_noise
            beta_schedule="cosine").to(self.device)

        self.swinunetr.is_training=False
        self.swinunetr.diff_on = self.diff_on

    def forward(self, complete_modality_image):

        complete_modality_features = extract_complete_modality_features(self.swinunetr,
                                                                        complete_modality_image,
                                                                        self.diff_on)
        diff_loss = self.diffusion(complete_modality_features)

        return diff_loss

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
            c_hidden_states_out_m1 = model.swinViTs[0](complete_modality_image[:,0:1,:,:], normalize=True)[4]
            c_hidden_states_out_m2 = model.swinViTs[1](complete_modality_image[:,1:2,:,:], normalize=True)[4]
            c_hidden_states_out_m3 = model.swinViTs[2](complete_modality_image[:,2:3,:,:], normalize=True)[4]
            c_hidden_states_out_m4 = model.swinViTs[3](complete_modality_image[:,3:4,:,:], normalize=True)[4]
            c_dec4_m1 = model.encoder10_list[0](c_hidden_states_out_m1)
            c_dec4_m2 = model.encoder10_list[1](c_hidden_states_out_m2)
            c_dec4_m3 = model.encoder10_list[2](c_hidden_states_out_m3)
            c_dec4_m4 = model.encoder10_list[3](c_hidden_states_out_m4)

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
class mm_3DLDMWrapper:
    def __init__(self, model):
        self.ldm_model = model
        self.channels_to_drop = 0
        
    def __call__(self, x):

        missing_modality_image = x

        with torch.no_grad():

            m_hidden_states_out_m1 = self.ldm_model.swinunetr.swinViTs[0](missing_modality_image[:,0:1,:,:], normalize=True)[4]
            m_hidden_states_out_m2 = self.ldm_model.swinunetr.swinViTs[1](missing_modality_image[:,1:2,:,:], normalize=True)[4]
            m_hidden_states_out_m3 = self.ldm_model.swinunetr.swinViTs[2](missing_modality_image[:,2:3,:,:], normalize=True)[4]
            m_hidden_states_out_m4 = self.ldm_model.swinunetr.swinViTs[3](missing_modality_image[:,3:4,:,:], normalize=True)[4]
            m_dec4_m1 = self.ldm_model.swinunetr.encoder10_list[0](m_hidden_states_out_m1)
            m_dec4_m2 = self.ldm_model.swinunetr.encoder10_list[1](m_hidden_states_out_m2)
            m_dec4_m3 = self.ldm_model.swinunetr.encoder10_list[2](m_hidden_states_out_m3)
            m_dec4_m4 = self.ldm_model.swinunetr.encoder10_list[3](m_hidden_states_out_m4)
            
            missing_modality_features = torch.cat((m_dec4_m1, m_dec4_m2, m_dec4_m3, m_dec4_m4), dim=1) # ([B, 4 * 768, 8, 8])
            

            if self.ldm_model.diff_on == "separate":
                missing_modality_features = drop_modality_feature_channel(missing_modality_features,
                                                                        self.ldm_model.config.generation_mode,
                                                                        self.channels_to_drop,
                                                                        self.ldm_model.fs,
                                                                        self.ldm_model.mean_features)
            elif self.ldm_model.diff_on == "combined":
                missing_modality_features = self.ldm_model.swinunetr.channel_reductions[5](missing_modality_features)

            generated_features = self.ldm_model.diffusion.diff_sample(missing_modality_features)
            
            c_f_m_generated_logits = self.ldm_model.swinunetr(missing_modality_image, bottleneck=[generated_features,self.channels_to_drop])
            missing_img_logits = self.ldm_model.swinunetr(missing_modality_image)

        return [missing_img_logits, c_f_m_generated_logits]