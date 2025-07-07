import torch
import torch.nn as nn
import torch.nn.functional as F
from .denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from nets.multimodal_swinunetr_shard import Multimodal_SwinUNETR_shard
from nets.multimodal_swinunetr import Multimodal_SwinUNETR

class mm_ldm(torch.nn.Module):

    def __init__(self,
                config,
                device_list = None):
        super(mm_ldm, self).__init__()

        self.device_list = device_list
        self.generation = config.generation_mode
        self.bs = config.train_batch_size
        self.fs = config.feature_size
        self.image_size = 8
        
        self.shard=True if len(self.device_list)>1 else False
        
        # feature extraction related
        if not self.shard:
            self.swinunetr = Multimodal_SwinUNETR(
                img_size=(128, 128, 128),
                in_channels=1, 
                out_channels=config.output_channel,
                feature_size=self.fs,
                cross_attention=config.cross_attention,
                deep_supervision=config.deep_supervision,
                t1c_spec=config.t1c_spec)

            # diffusion related
            self.unet = Unet(
                    dim=config.ldm_dim,
                    dim_mults=config.ldm_dim_mults,
                    channels=self.fs *16 *4
                    )
            self.diffusion = GaussianDiffusion(
                model=self.unet,
                image_size=self.image_size,
                timesteps=config.timesteps,
                )
            
            self.downsample = torch.nn.Conv2d(8*self.fs*16,4*self.fs*16,1)
            self.upsample = torch.nn.Conv2d(4*self.fs*16,8*self.fs*16,1)
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
                t1c_spec=config.t1c_spec)
            
            # diffusion related
            self.unet = Unet(
                    dim=config.ldm_dim,
                    dim_mults=config.ldm_dim_mults,
                    channels=self.fs *16 *4
                    ).to(self.device_list[1])
            self.diffusion = GaussianDiffusion(
                model=self.unet,
                image_size=self.image_size,
                timesteps=config.timesteps,
                ).to(self.device_list[1])
            
            self.downsample = torch.nn.Conv2d(8*self.fs*16,4*self.fs*16,1).to(self.device_list[1])
            self.upsample = torch.nn.Conv2d(4*self.fs*16,8*self.fs*16,1).to(self.device_list[1])

    
    def forward(self, missing_modality_image, complete_modality_image):

        missing_modality_features, complete_modality_features = extract_features(self.swinunetr,
                                                                                 self.device_list,
                                                                                 self.bs,
                                                                                 self.fs,
                                                                                 missing_modality_image,
                                                                                 complete_modality_image)

        concat_features = torch.cat((missing_modality_features, complete_modality_features), dim=1).to(self.device_list[1]) # ([B, 8 * 768, 8, 8]) 
        concat_features = self.downsample(concat_features)
        diff_loss = self.diffusion(concat_features)

        random_complete_modality_features = generate_random_features(self.generation,
                                                                     complete_modality_features,
                                                                     self.device_list)
        
        concat_features_random = torch.cat((missing_modality_features, random_complete_modality_features), dim=1).to(self.device_list[1]) # ([B, 1536, 8, 8]) 
        
        with torch.no_grad():
            concat_features_random = self.downsample(concat_features_random)

        generated_features = self.diffusion.diff_sample(concat_features_random)
        generated_features = self.upsample(generated_features)
        generated_missing_modality_features, generated_complete_modality_features = torch.chunk(generated_features, chunks=2, dim=1)

        complete_modality_features = complete_modality_features.to(self.device_list[1])
        modality1_mse = F.mse_loss(generated_complete_modality_features[:,0,...], complete_modality_features[:,0,...])
        modality2_mse = F.mse_loss(generated_complete_modality_features[:,1,...], complete_modality_features[:,1,...])
        modality3_mse = F.mse_loss(generated_complete_modality_features[:,2,...], complete_modality_features[:,2,...])
        modality4_mse = F.mse_loss(generated_complete_modality_features[:,3,...], complete_modality_features[:,3,...])

        return diff_loss, modality1_mse, modality2_mse, modality3_mse, modality4_mse

    

    def load_swinunetr_weights(self, checkpoint_path):

        #checkpoint = torch.load(checkpoint_path)  
        #self.swinunetr.load_state_dict(checkpoint)

        # from original pt
        #model_dict = torch.load(checkpoint_path)["state_dict"]
        #self.swinunetr.load_state_dict(model_dict)
        
        # from my saved pts
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.swinunetr.device_list[0]))["model_state_dict"]
        self.swinunetr.load_state_dict(checkpoint)
        print("model weights loaded successfully!")
        for param in self.swinunetr.parameters():
            param.requires_grad = False
        print("model weights are frozen")

def extract_features(model,device_list,bs,fs,missing_modality_image,complete_modality_image):
        with torch.no_grad():
            m_hidden_states_out_m1 = model.swinViT_1(missing_modality_image[:,0:1,:,:].to(device_list[0]), normalize=True)[4]
            m_hidden_states_out_m2 = model.swinViT_2(missing_modality_image[:,1:2,:,:].to(device_list[0]), normalize=True)[4]
            m_hidden_states_out_m3 = model.swinViT_3(missing_modality_image[:,2:3,:,:].to(device_list[1]), normalize=True)[4]
            m_hidden_states_out_m4 = model.swinViT_4(missing_modality_image[:,3:4,:,:].to(device_list[1]), normalize=True)[4]
            m_dec4_m1 = model.encoder10_1(m_hidden_states_out_m1).reshape(bs,fs*16,8,8)
            m_dec4_m2 = model.encoder10_2(m_hidden_states_out_m2).reshape(bs,fs*16,8,8)
            m_dec4_m3 = model.encoder10_3(m_hidden_states_out_m3).reshape(bs,fs*16,8,8).to(device_list[0])
            m_dec4_m4 = model.encoder10_4(m_hidden_states_out_m4).reshape(bs,fs*16,8,8).to(device_list[0])
            
            c_hidden_states_out_m1 = model.swinViT_1(complete_modality_image[:,0:1,:,:].to(device_list[0]), normalize=True)[4]
            c_hidden_states_out_m2 = model.swinViT_2(complete_modality_image[:,1:2,:,:].to(device_list[0]), normalize=True)[4]
            c_hidden_states_out_m3 = model.swinViT_3(complete_modality_image[:,2:3,:,:].to(device_list[1]), normalize=True)[4]
            c_hidden_states_out_m4 = model.swinViT_4(complete_modality_image[:,3:4,:,:].to(device_list[1]), normalize=True)[4]
            c_dec4_m1 = model.encoder10_1(c_hidden_states_out_m1).reshape(bs,fs*16,8,8)
            c_dec4_m2 = model.encoder10_2(c_hidden_states_out_m2).reshape(bs,fs*16,8,8)
            c_dec4_m3 = model.encoder10_3(c_hidden_states_out_m3).reshape(bs,fs*16,8,8).to(device_list[0])
            c_dec4_m4 = model.encoder10_4(c_hidden_states_out_m4).reshape(bs,fs*16,8,8).to(device_list[0])

            missing_modality_features = torch.cat((m_dec4_m1, m_dec4_m2, m_dec4_m3, m_dec4_m4), dim=1) # ([B, 4 * fs*16, 8, 8])
            complete_modality_features  = torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1) # ([B, 4 * fs*16, 8, 8])

        return missing_modality_features,complete_modality_features

def generate_random_features(generation,complete_modality_features,device_list):
    if generation == "gaussian":
        random_complete_modality_features = torch.randn_like(complete_modality_features)
    elif generation == "normal":
        random_complete_modality_features = torch.normal(complete_modality_features.mean(), complete_modality_features.std(), size=complete_modality_features.shape)
    elif generation == "mean":
        print("not implemented")
    random_complete_modality_features = random_complete_modality_features.to(device_list[0])
    
    return random_complete_modality_features

#  dynamic wrapper for sliding window inference 
class mm_LDMWrapper:
    def __init__(self, model):
        self.ldm_model = model

    def __call__(self, x):

        missing_modality_image, complete_modality_image = torch.split(x, 4, dim=1)

        m_hidden_states_out_m1 = self.ldm_model.swinunetr.swinViT_1(missing_modality_image[:,0:1,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
        m_hidden_states_out_m2 = self.ldm_model.swinunetr.swinViT_2(missing_modality_image[:,1:2,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
        m_hidden_states_out_m3 = self.ldm_model.swinunetr.swinViT_3(missing_modality_image[:,2:3,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
        m_hidden_states_out_m4 = self.ldm_model.swinunetr.swinViT_4(missing_modality_image[:,3:4,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
        m_dec4_m1 = self.ldm_model.swinunetr.encoder10_1(m_hidden_states_out_m1).reshape(self.ldm_model.bs,768,8,8).to(self.ldm_model.device_list[1])
        m_dec4_m2 = self.ldm_model.swinunetr.encoder10_2(m_hidden_states_out_m2).reshape(self.ldm_model.bs,768,8,8).to(self.ldm_model.device_list[1])
        m_dec4_m3 = self.ldm_model.swinunetr.encoder10_3(m_hidden_states_out_m3).reshape(self.ldm_model.bs,768,8,8)
        m_dec4_m4 = self.ldm_model.swinunetr.encoder10_4(m_hidden_states_out_m4).reshape(self.ldm_model.bs,768,8,8)
        
        missing_modality_features = torch.cat((m_dec4_m1, m_dec4_m2, m_dec4_m3, m_dec4_m4), dim=1) # ([B, 4 * 768, 8, 8])

        if self.ldm_model.generation == "gaussian":
            random_complete_modality_features = torch.randn_like(missing_modality_features)
        elif self.ldm_model.generation == "normal":
            random_complete_modality_features = torch.normal(missing_modality_features.mean(), missing_modality_features.std(), size=missing_modality_features.shape)
        elif self.ldm_model.generation == "mean":
            print("not implemented")
        random_complete_modality_features = random_complete_modality_features.to(self.ldm_model.device_list[1])
        
        concat_features = torch.cat((missing_modality_features, random_complete_modality_features), dim=1) # ([B, 8 * 768, 8, 8]) 
        
        concat_features = self.ldm_model.downsample(concat_features)
        generated_features = self.ldm_model.diffusion.diff_sample(concat_features)
        generated_features = self.ldm_model.upsample(generated_features)
        generated_missing_modality_features, generated_complete_modality_features = torch.chunk(generated_features, chunks=2, dim=1) # ([B, 4 * 768, 8, 8])
        
        generated_complete_modality_features = generated_complete_modality_features.to(self.ldm_model.device_list[0])
        generated_missing_modality_features = generated_missing_modality_features.to(self.ldm_model.device_list[0])

        complete_img_logits = self.ldm_model.swinunetr(complete_modality_image)
        complete_img_generated_logits = self.ldm_model.swinunetr(complete_modality_image, bottleneck=generated_complete_modality_features.reshape(-1,4*768,4,4,4))

        c_f_m_generated_logits = self.ldm_model.swinunetr(missing_modality_image, bottleneck=generated_complete_modality_features.reshape(-1,4*768,4,4,4))

        missing_img_logits = self.ldm_model.swinunetr(missing_modality_image)
        missing_img_generated_logits = self.ldm_model.swinunetr(missing_modality_image, bottleneck=generated_missing_modality_features.reshape(-1,4*768,4,4,4))

        return [missing_img_logits, missing_img_generated_logits, complete_img_logits, complete_img_generated_logits, c_f_m_generated_logits]