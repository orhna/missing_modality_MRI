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
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

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

    test_loader = utils.get_test_loader_BRATS21(img_path, split_path, channel_indices)

    # initialize loss and metrics according to output channel
    #loss_function, dice_metric, dice_metric_WT, dice_metric_TC, dice_metric_ET, sensitivity_metric, precision_metric, IOU_metric, post_trans = utils.initialize_loss_metric(Test_config.n_of_output_c)

    #/data/hjlee/orhun/data/BRATS21_Preprocessed/Training/Images/BRATS_00000.nii.gz
    #/data/hjlee/orhun/data/BRATS21_Preprocessed/Training/Labels/BRATS_00000.nii.gz

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    device = utils.initialize_GPU(device_list[0])

    model = SwinUNETR(
            img_size=128,
            in_channels=4,
            out_channels=3,
            feature_size=48
        )
    model_dict = torch.load(Test_config.model_file_path)["state_dict"]
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        val_outputs = None
        metric={}
    
        for val_data in tqdm(test_loader):
            val_input = val_data[0]
            val_input = val_input.to(device)
            val_labels = val_data[1].to(device)                        
            roi_size = (Test_config.crop_size, Test_config.crop_size, Test_config.crop_size)
            sw_batch_size = 1
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
