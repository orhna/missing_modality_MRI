# legacy code
# used in experiments that didn't yield decent results

import torch
import torch.nn.functional as F
from nets.multimodal_swinunetr import Multimodal_SwinUNETR
from monai.transforms import NormalizeIntensity

class memoryNet(torch.nn.Module):

    def __init__(self,
                config,
                device):
        super(memoryNet, self).__init__()

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


        self.memory_size = 50 
        self.feature_dim = (768, 4, 4, 4)
        flat_dim = torch.tensor(self.feature_dim).prod().item()  
        
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
        self.memory = torch.empty((self.memory_size, flat_dim)) 
        self.memory = torch.nn.init.normal_(self.memory, mean=0, std=1)
        self.memory = torch.nn.Parameter(self.memory, requires_grad=True).to(self.device)
        self.modality_size = 192 * 4 * 4 * 4  # each modality contributes 192x4x4x4
        self.num_modalities = 4

        
        self.swinunetr.is_training=False
        self.swinunetr.diff_on = self.diff_on

    def forward(self, feature_input, available_modalities):
        
        B, C, H, W, D = feature_input.shape
        # flattenning
        flat_input = feature_input.view(B, -1)

        # masking 
        total_dim = C * H * W * D 
        modality_dim = self.modality_size 
        mask = torch.zeros((1, total_dim), device=feature_input.device)

        for idx in available_modalities:
            start = idx * modality_dim
            end = (idx + 1) * modality_dim
            mask[:, start:end] = 1

        masked_input = flat_input * mask
        # normalization
        input_norm = F.normalize(masked_input, dim=1)
        memory_norm = F.normalize(self.memory, dim=1)
        #calculating cosine similarity
        sim = torch.mm(input_norm, memory_norm.t())  # (B, memory_size)
        #print("sim:",sim.shape)
        addressing = F.softmax(sim, dim=1)
        #print("addressing:",addressing)
        # retrieve feature vector from memory
        memory_readout = torch.mm(addressing, self.memory) 
        #print("memory_readout:",memory_readout)
 
        # replace the missing modality features in the original feature vector
        output_flat = flat_input.clone()
        for idx in range(self.num_modalities):
            if idx not in available_modalities:
                start = idx * modality_dim
                end = (idx + 1) * modality_dim
                output_flat[:, start:end] = memory_readout[:, start:end]

        reconstructed_features = output_flat.view(B, C, H, W, D)  # (B, 768, 4, 4, 4)
        return reconstructed_features
    
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
class memoryNetWrapper:
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