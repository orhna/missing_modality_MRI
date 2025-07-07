import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from nets.multimodal_swinunetr import Multimodal_SwinUNETR
from nets.recon_nets import MRIFeatureReconstructor
from monai.transforms import NormalizeIntensity

class mm_3Dunet(torch.nn.Module):

    def __init__(self,
                config,
                device):
        super(mm_3Dunet, self).__init__()

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
            dec_upsample=config.dec_upsample,
            recon_level=config.recon_level).to(self.device)

        # recon related
        feature_dim_map = {"0":1,
                            "1":1,
                            "2":2,
                            "3":4,
                            "4":8,
                            "5":16,}
        feature_dim = self.fs * feature_dim_map[str(config.recon_level[0])]
        print(feature_dim)
        self.recon = MRIFeatureReconstructor(feature_dim=feature_dim * self.multiplier).to(self.device)


        self.swinunetr.is_training=False
        self.swinunetr.diff_on = self.diff_on

    def forward(self, missing_modality_features):
        
        recon_complete_features= self.recon(missing_modality_features)
        
        return recon_complete_features

    def load_swinunetr_weights(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=self.device)["model_state_dict"]
        #checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.swinunetr.load_state_dict(checkpoint)
        print("model weights loaded successfully!")
        for param in self.swinunetr.parameters():
            param.requires_grad = False
        print("model weights are frozen")

def extract_complete_modality_features(model, complete_modality_image, recon_level, diff_on):
    
    with torch.no_grad():

        if recon_level == 1:
            c_enc1_m1 = model.encoder1_list[0](model.swinViTs[0](complete_modality_image[:,0:1,:,:], normalize=True)[0])
            c_enc1_m2 = model.encoder1_list[1](model.swinViTs[1](complete_modality_image[:,1:2,:,:], normalize=True)[0])
            c_enc1_m3 = model.encoder1_list[2](model.swinViTs[2](complete_modality_image[:,2:3,:,:], normalize=True)[0])
            c_enc1_m4 = model.encoder1_list[3](model.swinViTs[3](complete_modality_image[:,3:4,:,:], normalize=True)[0])

            if diff_on == "combined":
                complete_modality_features = model.channel_reductions[1](torch.cat((c_enc1_m1, c_enc1_m2, c_enc1_m3, c_enc1_m4), dim=1))
            elif diff_on == "separate":
                complete_modality_features  = torch.cat((c_enc1_m1, c_enc1_m2, c_enc1_m3, c_enc1_m4), dim=1) 

        elif recon_level == 2:
            c_enc2_m1 = model.encoder2_list[0](model.swinViTs[0](complete_modality_image[:,0:1,:,:], normalize=True)[1])
            c_enc2_m2 = model.encoder2_list[1](model.swinViTs[1](complete_modality_image[:,1:2,:,:], normalize=True)[1])
            c_enc2_m3 = model.encoder2_list[2](model.swinViTs[2](complete_modality_image[:,2:3,:,:], normalize=True)[1])
            c_enc2_m4 = model.encoder2_list[3](model.swinViTs[3](complete_modality_image[:,3:4,:,:], normalize=True)[1])

            if diff_on == "combined":
                complete_modality_features = model.channel_reductions[2](torch.cat((c_enc2_m1, c_enc2_m2, c_enc2_m3, c_enc2_m4), dim=1))
            elif diff_on == "separate":
                complete_modality_features  = torch.cat((c_enc2_m1, c_enc2_m2, c_enc2_m3, c_enc2_m4), dim=1) 

        elif recon_level == 3:
            c_enc3_m1 = model.encoder3_list[0](model.swinViTs[0](complete_modality_image[:,0:1,:,:], normalize=True)[2])
            c_enc3_m2 = model.encoder3_list[1](model.swinViTs[1](complete_modality_image[:,1:2,:,:], normalize=True)[2])
            c_enc3_m3 = model.encoder3_list[2](model.swinViTs[2](complete_modality_image[:,2:3,:,:], normalize=True)[2])
            c_enc3_m4 = model.encoder3_list[3](model.swinViTs[3](complete_modality_image[:,3:4,:,:], normalize=True)[2])

            if diff_on == "combined":
                complete_modality_features = model.channel_reductions[3](torch.cat((c_enc3_m1, c_enc3_m2, c_enc3_m3, c_enc3_m4), dim=1))
            elif diff_on == "separate":
                complete_modality_features  = torch.cat((c_enc3_m1, c_enc3_m2, c_enc3_m3, c_enc3_m4), dim=1) 

        elif recon_level == 4:
            c_enc4_m1 = model.encoder4_list[0](model.swinViTs[0](complete_modality_image[:,0:1,:,:], normalize=True)[3])
            c_enc4_m2 = model.encoder4_list[1](model.swinViTs[1](complete_modality_image[:,1:2,:,:], normalize=True)[3])
            c_enc4_m3 = model.encoder4_list[2](model.swinViTs[2](complete_modality_image[:,2:3,:,:], normalize=True)[3])
            c_enc4_m4 = model.encoder4_list[3](model.swinViTs[3](complete_modality_image[:,3:4,:,:], normalize=True)[3])

            if diff_on == "combined":
                complete_modality_features = model.channel_reductions[4](torch.cat((c_enc4_m1, c_enc4_m2, c_enc4_m3, c_enc4_m4), dim=1))
            elif diff_on == "separate":
                complete_modality_features  = torch.cat((c_enc4_m1, c_enc4_m2, c_enc4_m3, c_enc4_m4), dim=1) 

        elif recon_level == 5:
            c_dec4_m1 = model.encoder10_list[0](model.swinViTs[0](complete_modality_image[:,0:1,:,:], normalize=True)[4])
            c_dec4_m2 = model.encoder10_list[1](model.swinViTs[1](complete_modality_image[:,1:2,:,:], normalize=True)[4])
            c_dec4_m3 = model.encoder10_list[2](model.swinViTs[2](complete_modality_image[:,2:3,:,:], normalize=True)[4])
            c_dec4_m4 = model.encoder10_list[3](model.swinViTs[3](complete_modality_image[:,3:4,:,:], normalize=True)[4])

            if diff_on == "combined":
                complete_modality_features = model.channel_reductions[5](torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1))
            elif diff_on == "separate":
                complete_modality_features  = torch.cat((c_dec4_m1, c_dec4_m2, c_dec4_m3, c_dec4_m4), dim=1) # ([B, 4 * fs*16, 4, 4, 4])

    return complete_modality_features



def extract_features_from_swinvit(model, image, recon_level):
    
    with torch.no_grad():

        pre_x_out_m1   = model.swinViTs[0](image[:,0:1,:,:], normalize=True)[1][recon_level[0]-1]
        pre_x_out_m2   = model.swinViTs[1](image[:,1:2,:,:], normalize=True)[1][recon_level[0]-1]
        pre_x_out_m3   = model.swinViTs[2](image[:,2:3,:,:], normalize=True)[1][recon_level[0]-1]
        pre_x_out_m4   = model.swinViTs[3](image[:,3:4,:,:], normalize=True)[1][recon_level[0]-1]
        concat_features  = torch.cat((pre_x_out_m1, pre_x_out_m2, pre_x_out_m3, pre_x_out_m4), dim=1) # ([B, 4 * fs*16, 4, 4, 4])

    return concat_features


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

def drop_modality_feature_channel(_input, recon_level, method, idx_to_drop, fs, pc_mean_features):

    feature_dim_map = {"0":1,
                        "1":1,
                        "2":2,
                        "3":4,
                        "4":8,
                        "5":16,}
    multiplier = feature_dim_map[str(recon_level[0])]
    
    for idx in idx_to_drop:
        start_i= idx *fs*multiplier
        end_i= start_i+ fs*multiplier

        if method == "whole_mean":
            mean_value = torch.mean(_input)
            _input[:,start_i:end_i] = mean_value
        elif method == "zero":
            _input[:,start_i:end_i] = 0
        elif method == "gnoise":
            _input[:, start_i:end_i, :, :, :] = torch.randn_like(_input[:, start_i:end_i, :, :, :])
        elif method == "pcmean_features":
            _input[:, start_i:end_i] = pc_mean_features[:,start_i:end_i,...]
        elif method == "no_replacement":
            _input = _input

    return _input

#  dynamic wrapper for sliding window inference 
class mm_3DUnetWrapper:
    def __init__(self, model):
        self.rec_model = model
        self.channels_to_drop = 0
        self.normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
        self.recon_level = 0

    def __call__(self, x):

        missing_modality_image = x

        with torch.no_grad():
            
            if self.recon_level == 1:
                m_enc1_m1 = self.rec_model.swinunetr.encoder1_list[0](self.rec_model.swinunetr.swinViTs[0](missing_modality_image[:,0:1,:,:], normalize=True)[0])
                m_enc1_m2 = self.rec_model.swinunetr.encoder1_list[1](self.rec_model.swinunetr.swinViTs[1](missing_modality_image[:,1:2,:,:], normalize=True)[0])
                m_enc1_m3 = self.rec_model.swinunetr.encoder1_list[2](self.rec_model.swinunetr.swinViTs[2](missing_modality_image[:,2:3,:,:], normalize=True)[0])
                m_enc1_m4 = self.rec_model.swinunetr.encoder1_list[3](self.rec_model.swinunetr.swinViTs[3](missing_modality_image[:,3:4,:,:], normalize=True)[0])
                missing_modality_features = torch.cat((m_enc1_m1, m_enc1_m2, m_enc1_m3, m_enc1_m4), dim=1) # ([B, 4 * 768, 8, 8])
            
            if self.recon_level == 2:
                m_enc2_m1 = self.rec_model.swinunetr.encoder2_list[0](self.rec_model.swinunetr.swinViTs[0](missing_modality_image[:,0:1,:,:], normalize=True)[1])
                m_enc2_m2 = self.rec_model.swinunetr.encoder2_list[1](self.rec_model.swinunetr.swinViTs[1](missing_modality_image[:,1:2,:,:], normalize=True)[1])
                m_enc2_m3 = self.rec_model.swinunetr.encoder2_list[2](self.rec_model.swinunetr.swinViTs[2](missing_modality_image[:,2:3,:,:], normalize=True)[1])
                m_enc2_m4 = self.rec_model.swinunetr.encoder2_list[3](self.rec_model.swinunetr.swinViTs[3](missing_modality_image[:,3:4,:,:], normalize=True)[1])
                missing_modality_features = torch.cat((m_enc2_m1, m_enc2_m2, m_enc2_m3, m_enc2_m4), dim=1) # ([B, 4 * 768, 8, 8])
            
            if self.recon_level == 3:
                m_enc3_m1 = self.rec_model.swinunetr.encoder3_list[0](self.rec_model.swinunetr.swinViTs[0](missing_modality_image[:,0:1,:,:], normalize=True)[2])
                m_enc3_m2 = self.rec_model.swinunetr.encoder3_list[1](self.rec_model.swinunetr.swinViTs[1](missing_modality_image[:,1:2,:,:], normalize=True)[2])
                m_enc3_m3 = self.rec_model.swinunetr.encoder3_list[2](self.rec_model.swinunetr.swinViTs[2](missing_modality_image[:,2:3,:,:], normalize=True)[2])
                m_enc3_m4 = self.rec_model.swinunetr.encoder3_list[3](self.rec_model.swinunetr.swinViTs[3](missing_modality_image[:,3:4,:,:], normalize=True)[2])
                missing_modality_features = torch.cat((m_enc3_m1, m_enc3_m2, m_enc3_m3, m_enc3_m4), dim=1) # ([B, 4 * 768, 8, 8])
            
            if self.recon_level == 4:
                m_enc4_m1 = self.rec_model.swinunetr.encoder4_list[0](self.rec_model.swinunetr.swinViTs[0](missing_modality_image[:,0:1,:,:], normalize=True)[3])
                m_enc4_m2 = self.rec_model.swinunetr.encoder4_list[1](self.rec_model.swinunetr.swinViTs[1](missing_modality_image[:,1:2,:,:], normalize=True)[3])
                m_enc4_m3 = self.rec_model.swinunetr.encoder4_list[2](self.rec_model.swinunetr.swinViTs[2](missing_modality_image[:,2:3,:,:], normalize=True)[3])
                m_enc4_m4 = self.rec_model.swinunetr.encoder4_list[3](self.rec_model.swinunetr.swinViTs[3](missing_modality_image[:,3:4,:,:], normalize=True)[3])
                missing_modality_features = torch.cat((m_enc4_m1, m_enc4_m2, m_enc4_m3, m_enc4_m4), dim=1) # ([B, 4 * 768, 8, 8])
            
            if self.recon_level == 5:
                m_dec4_m1 = self.rec_model.swinunetr.encoder10_list[0](self.rec_model.swinunetr.swinViTs[0](missing_modality_image[:,0:1,:,:], normalize=True)[4])
                m_dec4_m2 = self.rec_model.swinunetr.encoder10_list[1](self.rec_model.swinunetr.swinViTs[1](missing_modality_image[:,1:2,:,:], normalize=True)[4])
                m_dec4_m3 = self.rec_model.swinunetr.encoder10_list[2](self.rec_model.swinunetr.swinViTs[2](missing_modality_image[:,2:3,:,:], normalize=True)[4])
                m_dec4_m4 = self.rec_model.swinunetr.encoder10_list[3](self.rec_model.swinunetr.swinViTs[3](missing_modality_image[:,3:4,:,:], normalize=True)[4])
                missing_modality_features = torch.cat((m_dec4_m1, m_dec4_m2, m_dec4_m3, m_dec4_m4), dim=1) # ([B, 4 * 768, 8, 8])
            

            if self.rec_model.diff_on == "separate":
                missing_modality_features = drop_modality_feature_channel(missing_modality_features,
                                                                        self.recon_level,
                                                                        self.rec_model.config.generation_mode,
                                                                        self.channels_to_drop,
                                                                        self.rec_model.fs,
                                                                        self.rec_model.mean_features)
            elif self.rec_model.diff_on == "combined":
                missing_modality_features = self.rec_model.swinunetr.channel_reductions[self.recon_level](missing_modality_features)

            #missing_modality_features = self.normalize(missing_modality_features)
            recon_complete_features = self.rec_model.recon(missing_modality_features)
            
            c_f_m_generated_logits = self.rec_model.swinunetr(missing_modality_image, bottleneck=[recon_complete_features,self.channels_to_drop])
            missing_img_logits = self.rec_model.swinunetr(missing_modality_image)

        return [missing_img_logits, c_f_m_generated_logits]


class mm_3DUnetWrapper_swinvit:
    def __init__(self, model):
        self.rec_model = model
        self.channels_to_drop = 0
        #self.normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
        self.recon_level = 0

    def __call__(self, x):

        missing_modality_image = x

        with torch.no_grad():

            pre_x_out_m1 = self.rec_model.swinunetr.swinViTs[0](missing_modality_image[:,0:1,:,:], normalize=True)[1][self.recon_level[0] - 1]
            pre_x_out_m2 = self.rec_model.swinunetr.swinViTs[1](missing_modality_image[:,1:2,:,:], normalize=True)[1][self.recon_level[0] - 1]
            pre_x_out_m3 = self.rec_model.swinunetr.swinViTs[2](missing_modality_image[:,2:3,:,:], normalize=True)[1][self.recon_level[0] - 1]
            pre_x_out_m4 = self.rec_model.swinunetr.swinViTs[3](missing_modality_image[:,3:4,:,:], normalize=True)[1][self.recon_level[0] - 1]
            missing_modality_features  = torch.cat((pre_x_out_m1, pre_x_out_m2, pre_x_out_m3, pre_x_out_m4), dim=1) # ([B, 4 * fs*16, 4, 4, 4])

            #missing_modality_features = self.normalize(missing_modality_features)
            recon_complete_features = self.rec_model.recon(missing_modality_features)
            
            c_f_m_generated_logits = self.rec_model.swinunetr(missing_modality_image, reconstructed=[recon_complete_features,self.channels_to_drop])
            missing_img_logits = self.rec_model.swinunetr(missing_modality_image)

        #return torch.stack([missing_img_logits, c_f_m_generated_logits], dim=0)
        #return torch.cat([missing_img_logits, c_f_m_generated_logits], dim=1)  # shape: [B, 2*C, D, H, W]
        return [missing_img_logits, c_f_m_generated_logits]