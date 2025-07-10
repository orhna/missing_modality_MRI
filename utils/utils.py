import random
import numpy as np
import json
import importlib.util
import os
import itertools
from collections import OrderedDict

from monai.transforms import (Lambda,
                               Compose, EnsureChannelFirst,
                                 RandSpatialCrop, RandRotate90, 
                                 NormalizeIntensity, RandAdjustContrast,
                                   RandZoom, RandFlip, RandGaussianNoise,
                                     RandGaussianSmooth, RandAdjustContrast,
                                     ConvertToMultiChannelBasedOnBratsClasses,
                                     RandScaleIntensity, RandShiftIntensity,
                                     SpatialCrop, SpatialPad
                                )
from monai.data import ImageDataset,DataLoader
import torch
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.losses.dice import DiceLoss
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

def initialize_GPU(device_id):
        print("Running on GPU:" + str(device_id))
        print(torch.cuda.is_available())  # Should return True if CUDA is available
        print(torch.cuda.device_count())
        cuda_id = "cuda:" + str(device_id)
        device = torch.device(cuda_id)
        torch.cuda.set_device(device_id)    
            
        return device

def initialize_GPUs(device_id_list):
        print("Running on GPUs:" + str(device_id_list[0])+" and " + str(device_id_list[1]))
        print(torch.cuda.is_available())  # Should return True if CUDA is available
        print(torch.cuda.device_count())
        cuda_id_1 = "cuda:" + str(device_id_list[0])
        cuda_id_2 = "cuda:" + str(device_id_list[1])

        device_1 = torch.device(cuda_id_1)
        device_2 = torch.device(cuda_id_2)

        return [device_1, device_2]

def separate_paths(image_list, label_list, train_file, val_file):
    
    def read_file(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    # Read train and validation sample names
    train_samples = read_file(train_file)
    val_samples = read_file(val_file)

    # Initialize lists for training and validation subsets
    train_images, train_labels = [], []
    val_images, val_labels = [], []

    # Separate paths based on sample names
    for image, label in zip(image_list, label_list):
        sample_name = os.path.basename(image).replace('.nii.gz', '')
        if sample_name in train_samples:
            train_images.append(image)
            train_labels.append(label)
        if sample_name in val_samples:
            val_images.append(image)
            val_labels.append(label)

    return train_images, train_labels, val_images, val_labels

def create_dataloader_original(images,
                                segs,
                                train_file, val_file,
                                workers,
                                train_batch_size: int,
                                cropped_input_size:list,
                                channel_indices):

    train_images, train_segs, val_images, val_segs = separate_paths(images, segs, train_file, val_file)

    def select_channels(x):
        if x.ndim == 4:
            return x[..., channel_indices]
        else:
            return x

    # image augmentation through spatial cropping to size and by randomly rotating
    train_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            RandFlip(prob=0.5,spatial_axis=[0, 1, 2]),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            RandScaleIntensity(factors=0.1, prob=0.1),
            RandShiftIntensity(offsets=0.1, prob=0.1),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ]
    )
    train_labeltrans = Compose(
        [   #Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            ConvertToMultiChannelBasedOnBratsClassesCustom(),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])

    val_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True,channel_wise=True)])
    val_segtrans = Compose([EnsureChannelFirst(),ConvertToMultiChannelBasedOnBratsClassesCustom()])
    
    # create a training data loader
    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_labeltrans)
    
    train_loader = DataLoader(train_ds,
                            batch_size=train_batch_size,
                            shuffle=True,
                            num_workers=workers,
                            pin_memory=0)
    
    # create a validation data loader
    val_ds = ImageDataset(val_images, val_segs, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    return train_loader, val_loader


def get_ldm_loaders(images,
                segs,
                train_file, val_file,
                workers,
                train_batch_size: int,
                cropped_input_size:list,
                channel_indices):

    train_images, train_segs, val_images, val_segs = separate_paths(images, segs, train_file, val_file)

    def select_channels(x):
        if x.ndim == 4:
            return x[..., channel_indices]
        else:
            return x

    train_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            NormalizeIntensity(nonzero=True,channel_wise=True),
            #RandFlip(prob=0.5,spatial_axis=[0, 1, 2]),
            #RandGaussianNoise(prob=0.15, mean=0.0, std= 0.33),
            #RandGaussianSmooth(prob=0.15, sigma_x=(0.5, 1.5),sigma_y=(0.5, 1.5),sigma_z=(0.5, 1.5)),
            #RandAdjustContrast(prob=0.15, gamma=(0.7, 1.3)),
            #RandScaleIntensity(factors=0.1, prob=1.0),
            #RandShiftIntensity(offsets=0.1, prob=1.0),
            #RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            SpatialCrop(roi_center=(66,85,70),roi_size=(128,128,128)) # 132,174,140
            #RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])
    train_labeltrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            ConvertToMultiChannelBasedOnBratsClassesCustom(),
            #RandFlip(prob=0.5,spatial_axis=[0, 1, 2]),
            #RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            SpatialCrop(roi_center=(66,85,70),roi_size=(128,128,128))
            #RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])

    val_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True,channel_wise=True),
            SpatialCrop(roi_center=(66,85,70),roi_size=(128,128,128)) # 132,174,140

        ])
    val_segtrans = Compose([
            EnsureChannelFirst(),
            ConvertToMultiChannelBasedOnBratsClassesCustom(),
            SpatialCrop(roi_center=(66,85,70),roi_size=(128,128,128)) # 132,174,140
        ])
    
    # create a training data loader
    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_labeltrans)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, drop_last=True, shuffle=True, num_workers=workers, pin_memory=0)
    
    # create a validation data loader
    val_ds = ImageDataset(val_images, val_segs, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    
    return train_loader, val_loader


def get_loaders_192(images,
                segs,
                train_file, val_file,
                workers,
                train_batch_size: int,
                channel_indices):

    train_images, train_segs, val_images, val_segs = separate_paths(images, segs, train_file, val_file)

    def select_channels(x):
        if x.ndim == 4:
            return x[channel_indices,...]
        else:
            return x

    train_imtrans = Compose(
        [   Lambda(select_channels),
            #EnsureChannelFirst(strict_check=True),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])
    train_labeltrans = Compose(
        [   #EnsureChannelFirst(strict_check=True),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])

    val_imtrans = Compose(
        [   Lambda(select_channels),
            #EnsureChannelFirst(),
        ])
    val_segtrans = Compose([
            #EnsureChannelFirst(),
        ])
    
    # create a training data loader
    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_labeltrans)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, drop_last=True, shuffle=True, num_workers=workers, pin_memory=0)
    
    # create a validation data loader
    val_ds = ImageDataset(val_images, val_segs, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    
    return train_loader, val_loader


def get_loaders(images,
                segs,
                train_file, val_file,
                workers,
                train_batch_size: int,
                cropped_input_size:list,
                channel_indices):

    train_images, train_segs, val_images, val_segs = separate_paths(images, segs, train_file, val_file)

    def select_channels(x):
        if x.ndim == 4:
            return x[..., channel_indices]
        else:
            return x

    train_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            NormalizeIntensity(nonzero=True,channel_wise=True),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])
    train_labeltrans = Compose(
        [   #Lambda(select_channels),
            EnsureChannelFirst(strict_check=True),
            ConvertToMultiChannelBasedOnBratsClassesCustom(),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            RandRotate90(prob=0.1, spatial_axes=(0, 2))
        ])

    val_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True,channel_wise=True)
        ])
    val_segtrans = Compose([
            EnsureChannelFirst(),
            ConvertToMultiChannelBasedOnBratsClassesCustom()
        ])
    
    # create a training data loader
    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_labeltrans)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, drop_last=True, shuffle=True, num_workers=workers, pin_memory=0)
    
    # create a validation data loader
    val_ds = ImageDataset(val_images, val_segs, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    
    return train_loader, val_loader

def get_test_loader(img_path,
                label_path,
                test_file,
                channel_indices):


    def select_channels(x):
        if x.ndim == 4:
            return x[..., channel_indices]
        else:
            return x

    with open(test_file, 'r') as f:
        sample_names = {line.strip() for line in f}  # Use a set for faster lookup
    
    image_dict = {}

    for file_name in os.listdir(img_path):
        if file_name.endswith('.nii.gz'):
            base_name = file_name[:-7] 
            if base_name in sample_names:
                image_dict[base_name] = os.path.join(img_path, file_name)

    common_base_names = sorted(image_dict.keys())
    image_list = [image_dict[name] for name in common_base_names]
    label_list= [img.replace(".nii.gz", "_label.nii.gz") for img in image_list]    
    label_list= [img.replace("Images", "Labels") for img in label_list]    

    test_imtrans = Compose(
        [   Lambda(select_channels),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True,channel_wise=True)
        ])
    test_segtrans = Compose([
            EnsureChannelFirst(),
            ConvertToMultiChannelBasedOnBratsClassesCustom()
        ])
    
    # create a training data loader
    test_ds = ImageDataset(image_list, label_list, transform=test_imtrans, seg_transform=test_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, drop_last=False, shuffle=False)
    
    return test_loader

def get_test_loader_BRATS21(images_folder, txt_file, channel_indices):

    def select_channels(x):
        if x.ndim == 4:
            return x[..., channel_indices]
        else:
            return x
    
    #reorder_channels = Lambda(lambda x: x[..., [2, 0, 1, 3]])
    #Lambda(select_channels)
    test_imtrans = Compose([Lambda(select_channels),EnsureChannelFirst()]) #, NormalizeIntensity(nonzero=True,channel_wise=True)
    test_segtrans = Compose([EnsureChannelFirst(), ConvertToMultiChannelBasedOnBratsClassesCustom()])

    with open(txt_file, 'r') as f:
        sample_names = {line.strip() for line in f}  # Use a set for faster lookup
    
    image_dict = {}

    for file_name in os.listdir(images_folder):
        if file_name.endswith('.nii.gz'):
            base_name = file_name[:-7] 
            if base_name in sample_names:
                image_dict[base_name] = os.path.join(images_folder, file_name)

    common_base_names = sorted(image_dict.keys())
    image_list = [image_dict[name] for name in common_base_names]
    label_list = [image_dict[name].replace("Images", "Labels") for name in common_base_names]
    
    test_ds = ImageDataset(image_list, label_list, transform=test_imtrans, seg_transform=test_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=0)
    print(len(test_ds))
    return test_loader

def save_config_from_py(file_path, logdir):
    # Load the Python file as a module
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract class attributes and convert them to dictionaries
    config = {}
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):  # Check if it's a class
            class_dict = {}
            for attr_name in dir(obj):
                if not attr_name.startswith("__"):  # Skip dunder methods
                    attr_value = getattr(obj, attr_name)
                    class_dict[attr_name] = attr_value
            config[name] = class_dict
        
    # Ensure Training_config is written first
    ordered_config = {}
    if "Training_config" in config:
        ordered_config["Training_config"] = config.pop("Training_config")
    ordered_config.update(config)  # Add the remaining classes in their original order

    output_json_path = logdir + "/config.json"
    with open(output_json_path, 'w') as json_file:
        json.dump(ordered_config, json_file, indent=4, separators=(',', ': '))
    
    print("Config saved as .json")

def rand_drop_channel(dataset_modalities: list, batch_img_data: torch.Tensor, mode: str = "zero"):
    
    for i in range(batch_img_data.shape[0]):
        number_of_dropped_modalities = np.random.randint(0, len(dataset_modalities))
        modalities_dropped = random.sample(list(np.arange(len(dataset_modalities))), number_of_dropped_modalities)
        modalities_dropped.sort()
        
        if number_of_dropped_modalities > 0:
            remaining_modalities = list(set(range(len(dataset_modalities))) - set(modalities_dropped))
            
            if mode == "zero":
                batch_img_data[i, modalities_dropped, :, :, :] = 0.
            elif mode == "modmean":
                mean_value = batch_img_data[i, remaining_modalities, :, :, :].mean(dim=0, keepdim=True)
                batch_img_data[i, modalities_dropped, :, :, :] = mean_value
            elif mode == "noise":
                mean_value = batch_img_data[i, remaining_modalities, :, :, :].mean(dim=0, keepdim=True)
                std_value = batch_img_data[i, remaining_modalities, :, :, :].std(dim=0, keepdim=True)
                noise = torch.normal(mean=mean_value, std=std_value)
                batch_img_data[i, modalities_dropped, :, :, :] = noise
            else:
                raise ValueError("Invalid mode. Choose from 'zero', 'mean', or 'noise'.")

    if len(modalities_dropped) > 0:
        modalities_dropped = tuple(modalities_dropped)
    else:
        modalities_dropped = "no_drop"
    
    return batch_img_data, modalities_dropped

def create_counter_dict():

    lst = [0, 1, 2, 3]
    counter = {}
    for r in range(1, len(lst) + 1):
        combinations = itertools.combinations(lst, r)
        for comb in combinations:
            for perm in itertools.permutations(comb):
                sorted_perm = tuple(sorted(perm))  
                counter[sorted_perm] = 0  

    counter["no_drop"] = 0
    return counter

def export_counter_dict(dict, path):
    output_json_path = path + "/random_counter.json"
    dict_with_str_keys = {str(key): value for key, value in dict.items()}
    with open(output_json_path, 'w') as json_file:
        json.dump(dict_with_str_keys, json_file, indent=4)

def initialize_loss_metric():
  
    loss_function = DiceLoss(softmax=True, to_onehot_y= True)
    
    sensitivity_metric = ConfusionMatrixMetric(include_background=True, metric_name='sensitivity', reduction="mean", get_not_nans=False)
    precision_metric = ConfusionMatrixMetric(include_background=True, metric_name='precision', reduction="mean", get_not_nans=False)
    IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])

    return loss_function, dice_metric, sensitivity_metric, precision_metric, IOU_metric, post_trans

def log_metrics(writer, metric, epoch, dataset_name):
    writer.add_scalar(f"Validation/{dataset_name}/IOU", metric["IOU"] , epoch)
    writer.add_scalar(f"Validation/{dataset_name}/dice_WT", metric["dice_WT"], epoch)
    writer.add_scalar(f"Validation/{dataset_name}/dice_TC", metric["dice_TC"], epoch)
    writer.add_scalar(f"Validation/{dataset_name}/dice_ET", metric["dice_ET"], epoch)
    writer.add_scalar(f"Validation/{dataset_name}/HD_ET", metric["HD_ET"], epoch)
    writer.add_scalar(f"Validation/{dataset_name}/HD_WT", metric["HD_WT"], epoch)
    writer.add_scalar(f"Validation/{dataset_name}/HD_TC", metric["HD_TC"], epoch)


def get_rgb(tensor):
    # Create a blank RGB volume
    rgb_volume = torch.zeros((3, *tensor.shape), dtype=torch.uint8)
    rgb_volume[0][tensor == 1] = 255  # Red for label 1
    rgb_volume[1][tensor == 2] = 255  # Green for label 2
    rgb_volume[2][tensor == 3] = 255  # Blue for label 3
    return rgb_volume

def get_rgb_overlay(input, label):
    # Create a blank RGB volume
    label = label.squeeze(0).squeeze(0)
    rgb_volume = input.repeat(3,1,1,1)
    #rgb_volume[0][label == 1] = 255  # Red for label 1
    #rgb_volume[1][label == 2] = 255  # Green for label 2
    #rgb_volume[2][label == 3] = 255  # Blue for label 3
    return rgb_volume

def log_visuals(modalities_to_train, input, outputs, labels, epoch, writer, dataset_name):
 
    for k in range(len(modalities_to_train)):
        plot_2d_or_3d_image([input[0,k].unsqueeze(0)], step=epoch, writer=writer, index=0, tag=f"{dataset_name}/{modalities_to_train[k]}")
    rgb_pred = get_rgb(outputs).permute(1,0,2,3,4)
    rgb_gt = get_rgb(labels).permute(2,1,0,3,4,5).squeeze(0)
    plot_2d_or_3d_image(rgb_pred, step=epoch, writer=writer, index=0, max_channels=3, tag=f"{dataset_name}/A_Pred_RGB")
    plot_2d_or_3d_image(rgb_gt, step=epoch, writer=writer, index=0, max_channels=3, tag=f"{dataset_name}/A_GT_RGB")

def set_random_seed(x):
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    random.seed(x)
    np.random.seed(x)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class ConvertToMultiChannelBasedOnBratsClassesCustom(ConvertToMultiChannelBasedOnBratsClasses):

    def __call__(self, img):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        #result = [ img == 3, (img == 1) | (img == 3), (img == 1) | (img == 3) | (img == 2)]
        result = [(img == 1) | (img == 3), (img == 1) | (img == 3) | (img == 2), img == 3]
        
        # order: ET, TC, WT
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

def get_test_scenarios():

    scenarios = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [1,1,0,0],
            [1,0,1,0],
            [1,0,0,1],
            [0,1,1,0],
            [0,1,0,1],
            [0,0,1,1],
            [1,1,1,0],
            [1,1,0,1],
            [1,0,1,1],
            [0,1,1,1],
            [1,1,1,1]]
    
    return scenarios

def load_n_duplicate_mm_weights(source_state_dict, target_state_dict, load_decoder=False, load_out=False):

    for name, param in source_state_dict.items():
        if "swinViT" in name:  
            if "patch_embed.proj.weight" in name:  
                print("skipping first layer weight")
                continue 
            if "patch_embed.proj.bias" in name:
                print("skipping first layer bias")
                continue  
            for i in range(4):  
                target_name = name.replace("swinViT", f"swinViT_{i+1}")
                if target_name in target_state_dict:
                    target_state_dict[target_name] = param
        """
        if "encoder1" in name:
            for i in range(4): 
                target_name = name.replace("encoder1", f"encoder1_{i+1}")
                print("duplicating encoder1 weights to multi encoders:", f"encoder1_{i+1}")
                if target_name in target_state_dict:
                    target_state_dict[target_name] = param
        """
        if "encoder2" in name:
            for i in range(4):  
                target_name = name.replace("encoder2", f"encoder2_{i+1}")
                if target_name in target_state_dict:    
                    target_state_dict[target_name] = param
        if "encoder3" in name:
            for i in range(4): 
                target_name = name.replace("encoder3", f"encoder3_{i+1}")
                if target_name in target_state_dict:
                    target_state_dict[target_name] = param
        if "encoder4" in name:
            for i in range(4): 
                target_name = name.replace("encoder4", f"encoder4_{i+1}")
                if target_name in target_state_dict:
                    target_state_dict[target_name] = param
        if "encoder10" in name:
            for i in range(4): 
                target_name = name.replace("encoder10", f"encoder10_{i+1}")
                if target_name in target_state_dict:
                    target_state_dict[target_name] = param
        if load_decoder == True:
            if "decoder" in name:
                if target_name in target_state_dict:
                    target_state_dict[name] = param
        if load_out == True:
            if "out" in name:
                if target_name in target_state_dict:
                        target_state_dict[name] = param

    prefixes_to_remove = ("swinViT.", "encoder1.", "encoder2.","encoder3.","encoder4.","encoder10.")
    target_state_dict = OrderedDict({k: v for k, v in target_state_dict.items() if not k.startswith(prefixes_to_remove)})

    return  target_state_dict

def test_random_dropping(_input, method, idx_to_drop, remaining_modalities, pc_mean_tensor=None ):
 
    if method == "whole_mean":
        mean_value = torch.mean(_input)
        _input[:,idx_to_drop,...] = mean_value
    elif method == "modality_mean":
        mean_value = torch.mean(_input[:, remaining_modalities, :, :, :],dim=1,keepdim=True)
        _input[:,idx_to_drop,...] = mean_value
    elif method == "pc_modality_mean":
        _input[:,idx_to_drop,...] = pc_mean_tensor[idx_to_drop]
    elif method == "zero":
        _input[:,idx_to_drop,...] = 0
    return _input

def get_dict_key(index_list):

    binary_key = ['0'] * 4
    for index in index_list:
        binary_key[index] = '1'
    binary_key = ''.join(binary_key) 
    
    return binary_key

def random_scenario():
    elements = {0, 1, 2, 3}

    valid_subsets = [list(subset) for i in range(1, len(elements))
                    for subset in itertools.combinations(elements, i)]

    return random.choice(valid_subsets)

def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print('#' * 100)
        ecart = (128 - (xmax - xmin)) // 2
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax

def crop(vol):
    """
    Crop a 5D tensor (B, C, H, W, D) to the smallest bounding box that contains all nonzero elements,
    while ensuring a minimum size of 128 per dimension.
    """
    assert len(vol.shape) == 5  # Ensure input has shape (B, C, H, W, D)
    B, C, H, W, D = vol.shape
    cropped_regions = []
    
    for b in range(B):
        vol_b = vol[b]  # Shape (C, H, W, D)
        vol_b = torch.amax(vol_b, dim=0)  # Max over the channel dimension, shape (H, W, D)
        
        nonzero_indices = torch.nonzero(vol_b, as_tuple=True)
        
        if len(nonzero_indices[0]) == 0:
            # If the volume is completely empty, return full dimensions
            cropped_regions.append((0, H, 0, W, 0, D))
            continue
        
        x_min, x_max = sup_128(torch.min(nonzero_indices[0]).item(), torch.max(nonzero_indices[0]).item())
        y_min, y_max = sup_128(torch.min(nonzero_indices[1]).item(), torch.max(nonzero_indices[1]).item())
        z_min, z_max = sup_128(torch.min(nonzero_indices[2]).item(), torch.max(nonzero_indices[2]).item())
        
        cropped_regions.append((x_min, x_max, y_min, y_max, z_min, z_max))
    
    return cropped_regions

def normalize(vol):

    """
    Normalize a 5D tensor (B, C, H, W, D) independently for each batch element and channel.
    """
    assert len(vol.shape) == 5  # Ensure input has shape (B, C, H, W, D)
    B, C, H, W, D = vol.shape
    
    for b in range(B):
        for k in range(C):
            x = vol[b, k, ...]
            mask = x > 0  # Compute mask for non-zero elements
            if mask.sum() > 0:  # Avoid division by zero
                y = x[mask]
                x = (x - y.mean()) / (y.std() + 1e-8)  # Add small epsilon to avoid division by zero
                vol[b, k, ...] = x
    
    return vol

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
