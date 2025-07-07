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
from monai.metrics import DiceMetric
from nets.mmformer import Model, LR_Scheduler, crop, normalize, NormalizeNonZeroIntensity, mmformer_mask,get_mmformer_test_loader, softmax_weighted_loss_monai, softmax_weighted_loss
from monai.transforms import Compose, Activations, AsDiscrete

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

    test_loader = get_mmformer_test_loader(img_path, seg_path, split_path, channel_indices)

    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    if model_sharding:
        device_id_list = utils.initialize_GPUs(device_list)
        device = device_id_list[0]
    else:
        device = utils.initialize_GPU(device_list[0])

    model = Model(num_cls=4).to(device)
    checkpoint = torch.load("/mnt/disk1/hjlee/orhun/repo/mmFormer/pt_model/model_last.pth")
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}

    model.load_state_dict(new_state_dict, strict=True)

    model.eval()  # Set model to evaluation mode
    model.is_training= False
    with torch.no_grad():
        val_outputs = None
        metric={}
    
        for val_data in tqdm(test_loader):
            val_input = val_data[0]
            val_input = val_input.to(device)
            val_labels = val_data[1].to(device)                        
            roi_size = (128, 128, 128)
            sw_batch_size = 1
            
            #cropping and normalizing
            """
            x_min, x_max, y_min, y_max, z_min, z_max = crop(val_input)[0]
            val_input = val_input[:,:,x_min:x_max, y_min:y_max, z_min:z_max]
            val_labels = val_labels[:,:,x_min:x_max, y_min:y_max, z_min:z_max]
            print("after crop val_input",val_input.shape)
            print("after crop val_labels",val_labels.shape)
            val_input = normalize(val_input)
            """
            ##################

            val_outputs = sliding_window_inference(val_input, roi_size, sw_batch_size, model)
            val_outputs = torch.argmax(val_outputs, dim=1)

            #val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)][0]
            # Convert labels to one-hot encoding

            val_labels = val_labels.squeeze(0)
            et_label = (val_labels == 3).float()
            wt_label = ((val_labels == 1) | (val_labels == 2) | (val_labels == 3)).float()
            tc_label = ((val_labels == 1) | (val_labels == 3)).float()

            et_pred = (val_outputs == 3).float()
            wt_pred = ((val_outputs == 1) | (val_outputs == 2) | (val_outputs == 3)).float()
            tc_pred = ((val_outputs == 1) | (val_outputs == 3)).float()

            label_stack = torch.stack([et_label, wt_label, tc_label], dim=1)  # [1, 3, 240, 240, 155]
            pred_stack = torch.stack([et_pred, wt_pred, tc_pred], dim=1)  # [1, 3, 240, 240, 155]
            dice_metric(pred_stack, label_stack)
            
        metric["dice_ET"] = dice_metric.aggregate()[0].item()
        metric["dice_TC"] = dice_metric.aggregate()[2].item()
        metric["dice_WT"] = dice_metric.aggregate()[1].item()
        dice_metric.reset()
        
    print(metric)
