import torch
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import utils.utils as utils
import config
import argparse

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU
from monai.networks.utils import one_hot
from nets.unet import res_unet
from nets.swinunetrldm import SwinUNETRWithLDM
from nets.multimodal_swinunetr import Multimodal_SwinUNETR
from nets.multimodal_swinunetr_shard import Multimodal_SwinUNETR_shard

from tensorboardX import SummaryWriter

if __name__ == "__main__":
    
    torch.multiprocessing.set_sharing_strategy('file_system') 
    #command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=str)
    args = parser.parse_args()

    device_list = list(map(int, args.device_id.split(',')))   
    
    if len(device_list) > 1:
        model_sharding = True
    else:
        model_sharding = False

    Test_config = config.Test_config()
    Database_config = config.Database_config()

    chosen_ds = Test_config.dataset
    # path initialization 
    img_path=Database_config.img_path[chosen_ds]
    seg_path=Database_config.seg_path[chosen_ds]
    split_path=Database_config.split_path[chosen_ds]["test"]

    n_of_channels = len(Test_config.modalities)
    channel_indices = []
    modalities_to_train = Test_config.modalities
    for _m in modalities_to_train:
        channel_indices.append(Database_config.channels[chosen_ds].index(_m))
    print("Channel indices to be loaded :",channel_indices)

    test_loader = utils.get_test_loader(img_path, seg_path, split_path, channel_indices)

    # initialize loss and metrics according to output channel
    loss_function, dice_metric, sensitivity_metric, precision_metric, IOU_metric, post_trans = utils.initialize_loss_metric(new_eval=True)

    if model_sharding:
        device_id_list = utils.initialize_GPUs(device_list)
        device = device_id_list[0]
    else:
        device = utils.initialize_GPU(device_list[0])

    if Test_config.model== "SwinUNETR":
        print("Testing SwinUNETR model:",Test_config.model_file_path )
        model = SwinUNETR(
            img_size=(Test_config.crop_size, Test_config.crop_size, Test_config.crop_size),
            in_channels=n_of_channels, 
            out_channels=Test_config.n_of_output_c,
            feature_size=48,
            use_checkpoint=True).to(device)
        checkpoint = torch.load(Test_config.model_file_path)
        model.load_state_dict(checkpoint)
    elif Test_config.model == "MultiModal_SwinUNETR":
        print("TRAINING WITH MultiModal_SwinUNETR")
        if not model_sharding:
            model = Multimodal_SwinUNETR(
                img_size=(Test_config.crop_size, Test_config.crop_size, Test_config.crop_size),
                in_channels=1, 
                out_channels=Test_config.n_of_output_c,
                feature_size=Test_config.feature_size,
                use_checkpoint=True,
                cross_attention=Test_config.cross_attention,
                deep_supervision=Test_config.deep_supervision,
                t1c_spec=Test_config.t1c_spec,
                sep_dec=Test_config.sep_dec).to(device)
        else:
            model = Multimodal_SwinUNETR_shard(
                img_size=(Test_config.crop_size, Test_config.crop_size, Test_config.crop_size),
                in_channels=1, 
                out_channels=Test_config.n_of_output_c,
                device_list=device_id_list,
                feature_size=Test_config.feature_size,
                use_checkpoint=True,
                cross_attention=Test_config.cross_attention,
                deep_supervision=Test_config.deep_supervision,
                t1c_spec=Test_config.t1c_spec,
                sep_dec=Test_config.sep_dec)
        checkpoint = torch.load(Test_config.model_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.is_training=False
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        val_outputs = None
        metric={}
    
        for val_data in tqdm(test_loader):
            val_input = val_data[0]
            val_input = val_input.to(device)
            val_labels = val_data[1].to(device)                        
            roi_size = (Test_config.crop_size, Test_config.crop_size, Test_config.crop_size)
            sw_batch_size = 1
            
            #cropping and normalizing
            
            ##################
            
            #using sliding window for the whole 3D image
            val_outputs = sliding_window_inference(val_input, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        metric["dice"] = dice_metric.aggregate()
        print(metric["dice"])
        
        metric["dice_ET"] = dice_metric.aggregate()[2].item()
        metric["dice_TC"] = dice_metric.aggregate()[0].item()
        metric["dice_WT"] = dice_metric.aggregate()[1].item()

        dice_metric.reset()

    print(metric)
