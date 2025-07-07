import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from .denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from nets.multimodal_swinunetr_shard import Multimodal_SwinUNETR_shard
from nets.multimodal_swinunetr import Multimodal_SwinUNETR

class mm_ldmv2(torch.nn.Module):

    def __init__(self,
                config,
                device_list = None):
        super(mm_ldmv2, self).__init__()

        self.device_list = device_list
        self.generation = config.generation_mode
        self.bs = config.train_batch_size
        self.fs = config.feature_size
        self.image_size = 16
        
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
                t1c_spec=config.t1c_spec).to(device_list[0])

            # diffusion related
            self.unet = Unet(
                    dim=config.ldm_dim,
                    dim_mults=config.ldm_dim_mults,
                    channels=self.fs *16
                    ).to(device_list[0])

            self.diffusion = GaussianDiffusion(
                model=self.unet,
                image_size=self.image_size,
                timesteps=config.timesteps,
                ).to(device_list[0])

            
            #self.downsample = torch.nn.Conv2d(8*self.fs*16,4*self.fs*16,1)
            #self.upsample = torch.nn.Conv2d(4*self.fs*16,8*self.fs*16,1)
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
            
            #self.downsample = torch.nn.Conv2d(4*self.fs*16,4*self.fs*16,1).to(self.device_list[1])
            #self.upsample = torch.nn.Conv2d(4*self.fs*16,8*self.fs*16,1).to(self.device_list[1])
        self.swinunetr.is_training=False
    
    def forward(self, complete_modality_image):

        complete_modality_features = extract_complete_modality_features(self.swinunetr,
                                                                        self.device_list,
                                                                        self.bs,
                                                                        self.fs,
                                                                        self.image_size,
                                                                        complete_modality_image)
        # complete_modality_features [4*384,8,8] or [384, 16, 16]
        #complete_modality_features = self.downsample(complete_modality_features)
        
        diff_loss = self.diffusion(complete_modality_features)

        return diff_loss
    
        """
        B,C,H,W = complete_modality_features.shape
        stacked_complete_modality_features = complete_modality_features.view(B,4,C//4,H,W)
        
        random_modality_features = generate_random_features(self.generation,
                                                            stacked_complete_modality_features[0][0],
                                                            self.device_list)
        #with torch.no_grad():
        #    concat_features_random = self.downsample(concat_features_random)
        
        
        _list_mods = [0,1,2,3]
        loss_dict={}
        for r in range(1, 2):  # Only consider 1, 2, or 3 channels
            for channels in itertools.combinations(range(4), r):  # Generate all valid subsets
                stacked_missing_modality_features = stacked_complete_modality_features.clone()
                for ch in channels:
                    stacked_missing_modality_features[:, ch] = random_modality_features  # Set selected channels
            
                generated_complete_modality_features = self.diffusion.diff_sample(stacked_missing_modality_features.reshape(B,C,H,W))

                _loss = F.mse_loss(generated_complete_modality_features,complete_modality_features)
                print(_loss.item())
                #print("channels to drop:",channels)
                key = str([item for item in _list_mods if item not in list(channels)])
                #print("channels to keep:",key)

                loss_dict[key] = _loss
        
        stacked_missing_modality_features = stacked_complete_modality_features.clone()
        stacked_missing_modality_features[:, 0] = random_modality_features 
            
        generated_complete_modality_features = self.diffusion.diff_sample(stacked_missing_modality_features.reshape(B,C,H,W))
        mse_loss = F.mse_loss(generated_complete_modality_features,complete_modality_features)
        """
         #, mse_loss

    

    def load_swinunetr_weights(self, checkpoint_path):

        #checkpoint = torch.load(checkpoint_path)  
        #self.swinunetr.load_state_dict(checkpoint)

        # from original pt
        #model_dict = torch.load(checkpoint_path)["state_dict"]
        #self.swinunetr.load_state_dict(model_dict)
        
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

def extract_complete_modality_features(model,device_list,bs,fs,image_size,complete_modality_image):
        with torch.no_grad():
            c_hidden_states_out_m1 = model.swinViT_1(complete_modality_image[:,0:1,:,:].to(device_list[0]), normalize=True)[4]
            c_hidden_states_out_m2 = model.swinViT_2(complete_modality_image[:,1:2,:,:].to(device_list[0]), normalize=True)[4]
            c_hidden_states_out_m3 = model.swinViT_3(complete_modality_image[:,2:3,:,:].to(device_list[1]), normalize=True)[4]
            c_hidden_states_out_m4 = model.swinViT_4(complete_modality_image[:,3:4,:,:].to(device_list[1]), normalize=True)[4]
            c_dec4_m1 = model.encoder10_1(c_hidden_states_out_m1).reshape(bs,fs*16,8,8)
            c_dec4_m2 = model.encoder10_2(c_hidden_states_out_m2).reshape(bs,fs*16,8,8)
            c_dec4_m3 = model.encoder10_3(c_hidden_states_out_m3).reshape(bs,fs*16,8,8).to(device_list[0])
            c_dec4_m4 = model.encoder10_4(c_hidden_states_out_m4).reshape(bs,fs*16,8,8).to(device_list[0])

            complete_modality_features  = torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1) # ([B, 4 * fs*16, 8, 8])
            #B,C,H,W = complete_modality_features.shape
            complete_modality_features = complete_modality_features.reshape(bs,fs*16,image_size,image_size)

        return complete_modality_features

def generate_random_features(generation,complete_modality_features,device_list):
    if generation == "gaussian":
        random_complete_modality_features = torch.randn_like(complete_modality_features)
    elif generation == "normal":
        random_complete_modality_features = torch.normal(complete_modality_features.mean(), complete_modality_features.std(), size=complete_modality_features.shape)
    elif generation == "mean":
        print("not implemented")
    random_complete_modality_features = random_complete_modality_features.to(device_list[0])
    
    return random_complete_modality_features

def random_modality_channel(drop,complete_modality_image,device_list):
    if drop == "noise":
        random_modality_channel = torch.randn_like(complete_modality_image.shape)
    elif drop == "normal":
        random_modality_channel = torch.normal(complete_modality_image.mean(), complete_modality_image.std(), size=complete_modality_image.shape)
    elif drop == "zero":
        random_modality_channel = torch.zeros(complete_modality_image.shape)
    elif drop == "mean":
        print("not implemented")
    random_modality_channel = random_modality_channel.to(device_list[0])
    
    return random_modality_channel


#  dynamic wrapper for sliding window inference 
class mm_LDMWrapperv2:
    def __init__(self, model):
        self.ldm_model = model
        self.channels = 0
        self.mean_features = torch.load("/mnt/disk1/hjlee/orhun/repo/thesis/mean_features_1536_4_4_4.pt")
    def __call__(self, x):

        missing_modality_image = x
        m_hidden_states_out_m1 = self.ldm_model.swinunetr.swinViT_1(missing_modality_image[:,0:1,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
        m_hidden_states_out_m2 = self.ldm_model.swinunetr.swinViT_2(missing_modality_image[:,1:2,:,:].to(self.ldm_model.device_list[0]), normalize=True)[4]
        m_hidden_states_out_m3 = self.ldm_model.swinunetr.swinViT_3(missing_modality_image[:,2:3,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
        m_hidden_states_out_m4 = self.ldm_model.swinunetr.swinViT_4(missing_modality_image[:,3:4,:,:].to(self.ldm_model.device_list[1]), normalize=True)[4]
        #m_dec4_m1 = self.ldm_model.swinunetr.encoder10_1(m_hidden_states_out_m1).reshape(self.ldm_model.bs,16*self.ldm_model.fs,8,8).to(self.ldm_model.device_list[1])
        #m_dec4_m2 = self.ldm_model.swinunetr.encoder10_2(m_hidden_states_out_m2).reshape(self.ldm_model.bs,16*self.ldm_model.fs,8,8).to(self.ldm_model.device_list[1])
        #m_dec4_m3 = self.ldm_model.swinunetr.encoder10_3(m_hidden_states_out_m3).reshape(self.ldm_model.bs,16*self.ldm_model.fs,8,8)
        #m_dec4_m4 = self.ldm_model.swinunetr.encoder10_4(m_hidden_states_out_m4).reshape(self.ldm_model.bs,16*self.ldm_model.fs,8,8)
        m_dec4_m1 = self.ldm_model.swinunetr.encoder10_1(m_hidden_states_out_m1).to(self.ldm_model.device_list[1])
        m_dec4_m2 = self.ldm_model.swinunetr.encoder10_2(m_hidden_states_out_m2).to(self.ldm_model.device_list[1])
        m_dec4_m3 = self.ldm_model.swinunetr.encoder10_3(m_hidden_states_out_m3)
        m_dec4_m4 = self.ldm_model.swinunetr.encoder10_4(m_hidden_states_out_m4)
        
        missing_modality_features = torch.cat((m_dec4_m1, m_dec4_m2, m_dec4_m3, m_dec4_m4), dim=1) # ([B, 4 * 768, 8, 8])
        """
        random_modality_features = generate_random_features(self.ldm_model.generation,
                                                            missing_modality_features[0][0],
                                                            self.ldm_model.device_list)
        
        for ch in self.channels:
            missing_modality_features[:, ch] = random_modality_features  # Set selected channels
        """
        
        #missing_modality_features[:, 0] = random_modality_features  # Set selected channels
        print(missing_modality_features.shape)
        missing_modality_features[:, :384] = self.mean_features[:,:384,...]
        
        
        missing_modality_features = missing_modality_features.reshape(self.ldm_model.bs,self.ldm_model.fs*16,self.ldm_model.image_size,self.ldm_model.image_size)

        missing_modality_features = missing_modality_features.to(self.ldm_model.device_list[1])
        generated_features = self.ldm_model.diffusion.diff_sample(missing_modality_features).to(self.ldm_model.device_list[0])
        
        generated_features= torch.chunk(generated_features.reshape(self.ldm_model.bs, 4* 16*self.ldm_model.fs, 4,4,4), 4, dim=1)
        c_f_m_generated_logits = self.ldm_model.swinunetr(missing_modality_image,
                                                          bottleneck=generated_features)
        missing_img_logits = self.ldm_model.swinunetr(missing_modality_image)

        return [missing_img_logits, c_f_m_generated_logits]