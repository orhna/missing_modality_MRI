import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from monai.networks.nets import SwinUNETR
from nets.swinunetrldm import SwinUNETRWithLDM

class ldm(torch.nn.Module):

    def __init__(self, dim=64, channels=768, image_size=8, timesteps=1000, bs=None):
        super(ldm, self).__init__()

        self.bs = bs
        self.training = True
        self.swinunetr = SwinUNETRWithLDM(
            img_size=(128, 128, 128),
            in_channels=4, 
            out_channels=3,
            feature_size=48,
            use_checkpoint=True)
        
        self.unet = Unet(
                dim=dim,
                dim_mults=(1, 2),
                channels=channels
                )
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=image_size,
            timesteps=timesteps,
            )
        
        self.downsample = torch.nn.Conv2d(1536,768,1)
        self.upsample = torch.nn.Conv2d(768,1536,1)
        
    def forward(self, missing_modality_image, complete_modality_image):
        
        with torch.no_grad():

            missing_modality_features_h = self.swinunetr.swinViT(missing_modality_image, normalize=True)[4] #([B, 768, 8, 8])
            missing_modality_features = self.swinunetr.encoder10(missing_modality_features_h).reshape(self.bs,768,8,8) #([B, 768, 8, 8])

            complete_modality_features_h = self.swinunetr.swinViT(complete_modality_image, normalize=True)[4] #([B, 768, 8, 8])
            complete_modality_features = self.swinunetr.encoder10(complete_modality_features_h).reshape(self.bs,768,8,8) #([B, 768, 8, 8])

        concat_features = torch.cat((missing_modality_features, complete_modality_features), dim=1) # ([B, 1536, 8, 8]) 
        concat_features = self.downsample(concat_features)
        diff_loss = self.diffusion(concat_features)

        random_complete_modality_features = torch.randn_like(complete_modality_features)
        concat_features_random = torch.cat((missing_modality_features, random_complete_modality_features), dim=1) # ([B, 1536, 8, 8]) 
        with torch.no_grad():
            concat_features_random = self.downsample(concat_features_random)

        generated_features = self.diffusion.diff_sample(concat_features_random)
        generated_features = self.upsample(generated_features)
        generated_missing_modality_features, generated_complete_modality_features = torch.chunk(generated_features, chunks=2, dim=1)

        #mse_complete_features = F.mse_loss(generated_complete_modality_features, complete_modality_features)
        mse_c_f_m_g_features = F.mse_loss(generated_missing_modality_features, complete_modality_features)

        return diff_loss, mse_c_f_m_g_features #mse_random_features, mse_complete_features,

    def load_swinunetr_weights(self, checkpoint_path):

        #checkpoint = torch.load(checkpoint_path)  
        #self.swinunetr.load_state_dict(checkpoint)

        # from original pt
        #model_dict = torch.load(checkpoint_path)["state_dict"]
        #self.swinunetr.load_state_dict(model_dict)
        
        # from my saved pts
        checkpoint = torch.load(checkpoint_path)["model_state_dict"]
        self.swinunetr.load_state_dict(checkpoint)
        print("SwinUNETR weights loaded successfully!")
        for param in self.swinunetr.parameters():
            param.requires_grad = False
        print("SwinUNETR weights are frozen")


#  dynamic wrapper for sliding window inference for 
class LDMWrapper:
    def __init__(self, model):
        self.ldm_model = model


    def __call__(self, x):

        missing_modality_image, complete_modality_image = torch.split(x, 4, dim=1)

        missing_modality_features = self.ldm_model.swinunetr.swinViT(missing_modality_image, normalize=True)[4].reshape(-1,768,8,8) #([B, 768, 8, 8])
        #complete_modality_features = self.ldm_model.swinunetr.swinViT(complete_modality_image, normalize=True)[4].reshape(-1,768,8,8) #([B, 768, 8, 8])
        
        random_complete_modality_features = torch.randn_like(missing_modality_features)
        concat_features = torch.cat((missing_modality_features, random_complete_modality_features), dim=1) # ([B, 1536, 8, 8]) 
        
        concat_features = self.ldm_model.downsample(concat_features)
        generated_features = self.ldm_model.diffusion.diff_sample(concat_features)
        generated_features = self.ldm_model.upsample(generated_features)
        generated_missing_modality_features, generated_complete_modality_features = torch.chunk(generated_features, chunks=2, dim=1)
        
        complete_img_logits = self.ldm_model.swinunetr(complete_modality_image)
        complete_img_generated_logits = self.ldm_model.swinunetr(complete_modality_image, bottleneck=generated_complete_modality_features.reshape(-1,768,4,4,4))

        c_f_r_generated_logits = self.ldm_model.swinunetr(missing_modality_image, bottleneck=generated_complete_modality_features.reshape(-1,768,4,4,4))

        missing_img_logits = self.ldm_model.swinunetr(missing_modality_image)
        missing_img_generated_logits = self.ldm_model.swinunetr(missing_modality_image, bottleneck=generated_missing_modality_features.reshape(-1,768,4,4,4))

        return [missing_img_logits, missing_img_generated_logits, complete_img_logits, complete_img_generated_logits, c_f_r_generated_logits]