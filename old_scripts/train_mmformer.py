import torch
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import utils.utils as utils
import config
import argparse
from nets.mmformer import Model, LR_Scheduler, mmformer_mask, get_mmformer_loaders, softmax_weighted_loss, dice_loss
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU
from monai.networks.utils import one_hot
from torch.optim.lr_scheduler import StepLR
from monai.transforms import Compose, Activations, AsDiscrete
from monai.losses.dice import DiceLoss, GeneralizedDiceLoss

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
    #load config
    Training_config = config.Training_config()
    Database_config = config.Database_config()
    _date = datetime.now().strftime("%d-%H-%M")
    log_dir = f"./logs/mmformer/{Training_config.experiment_name}_{_date}"
    writer = SummaryWriter(log_dir=log_dir)

    
    #save config
    utils.save_config_from_py("config.py",log_dir)
    #set seed
    utils.set_random_seed(Training_config.seed)

    cropped_input_size = Training_config.cropped_input_size
    epochs = Training_config.epoch
    print("lr: ",Training_config.lr)
    print("Workers: ", Training_config.workers)
    print("Batch size: ",Training_config.train_batch_size)

    #set index
    img_index = 0
    label_index = 1 

    # Set the data size and total modalities
    chosen_ds = Training_config.dataset_to_train[0]
    channels= Training_config.modalities_to_train
    train_size=Database_config.train_size[chosen_ds]
    total_size=Database_config.total_size[chosen_ds]

    print("modalities to train: ", channels)

    # path initialization 
    img_path=Database_config.img_path[chosen_ds]
    seg_path=Database_config.seg_path[chosen_ds]

    model_save_path = Training_config.model_save_path + Training_config.experiment_name + "/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    load_model_path = Training_config.load_model_path

    images= sorted(glob(os.path.join(img_path, "*.*")))
    segs = sorted(glob(os.path.join(seg_path,"*.*"))) 

    # create dict to count dropped modalities frequency
    counter_dict = utils.create_counter_dict()

    channel_indices = []
    modalities_to_train = Training_config.modalities_to_train
    for _m in modalities_to_train:
        channel_indices.append(Database_config.channels[chosen_ds].index(_m))
    print("Channel index order to be loaded :",channel_indices)

    train_loader, val_loader = get_mmformer_loaders(
                                images=images,
                                segs=segs,
                                train_file=Database_config.split_path[chosen_ds]["train"],
                                val_file=Database_config.split_path[chosen_ds]["val"] ,
                                workers=Training_config.workers,
                                train_batch_size=Training_config.train_batch_size,
                                cropped_input_size=cropped_input_size,
                                channel_indices=channel_indices)

    # initialize GPU
    if model_sharding:
        device_id_list = utils.initialize_GPUs(device_list)
        device = device_id_list[0]
    else:
        device = utils.initialize_GPU(device_list[0])
    
    num_cls = 4
    model = Model(num_cls=num_cls).to(device)

    if Training_config.continue_training:
        checkpoint = torch.load("/mnt/disk1/hjlee/orhun/repo/mmFormer/pt_model/model_last.pth")
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}

        
        new_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith("decoder_fuse")}
        
        model.load_state_dict(new_state_dict, strict=False)
        print("model weights are loaded")
    
    
    lr_schedule = LR_Scheduler(Training_config.lr, Training_config.epoch)
    train_params = [{'params': model.parameters(), 'lr': Training_config.lr, 'weight_decay':1e-6}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=Training_config.lr, weight_decay=1e-6)

    #validation related
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    #loss_function = DiceLoss(softmax=True, to_onehot_y= False)
    dice_loss_fn = DiceLoss(include_background=True, to_onehot_y=True, reduction="mean")  
    generalized_dice_loss_fn = GeneralizedDiceLoss(include_background=True, to_onehot_y=True)
    
    epoched = 0
    #training
    for epoch in tqdm(range(epoched,epochs)):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        step_lr = lr_schedule(optimizer, epoch)

        for batch in tqdm(train_loader):
            step += 1
            label = batch[label_index].to(device)  
            batch[img_index], mask = mmformer_mask(modalities_to_train, batch[img_index])
            input_data = batch[img_index].to(device)

            model.is_training = True
            fuse_pred, sep_preds, prm_preds = model(input_data, mask)
            assert not torch.isnan(fuse_pred).any(), "fuse_pred contains NaNs!"
            assert not torch.isnan(sep_preds[0]).any(), "sep_preds contains NaNs!"
            #assert not torch.isnan(prm_preds[0]).any(), "Output contains NaNs!"

            #label = one_hot(label, num_classes=4)
            
            ###Loss compute
            #fuse_cross_loss = softmax_weighted_loss(fuse_pred, label, num_cls=num_cls)
            #fuse_dice_loss = dice_loss(fuse_pred, label, num_cls=num_cls)
            fuse_cross_loss = generalized_dice_loss_fn(fuse_pred, label)
            fuse_dice_loss = dice_loss_fn(fuse_pred, label)

            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).to(device)
            sep_dice_loss = torch.zeros(1).to(device)
            for sep_pred in sep_preds:
                #sep_cross_loss += softmax_weighted_loss(sep_pred, label, num_cls=num_cls)
                #sep_dice_loss += dice_loss(sep_pred, label, num_cls=num_cls)
                sep_cross_loss += generalized_dice_loss_fn(sep_pred, label)
                sep_dice_loss += dice_loss_fn(sep_pred, label)
                sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).to(device)
            prm_dice_loss = torch.zeros(1).to(device)
            for prm_pred in prm_preds:
                #prm_cross_loss += softmax_weighted_loss(prm_pred, label, num_cls=num_cls)
                #prm_dice_loss += dice_loss(prm_pred, label, num_cls=num_cls)
                prm_cross_loss += generalized_dice_loss_fn(prm_pred, label)
                prm_dice_loss += dice_loss_fn(prm_pred, label)
            prm_loss = prm_cross_loss + prm_dice_loss

            loss = fuse_loss + sep_loss + prm_loss
            
            optimizer.zero_grad()
            loss.backward()
            """
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} - Gradient mean: {param.grad.mean()}, Gradient max: {param.grad.max()}")
            """
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len = train_size  // Training_config.train_batch_size

            print("loss:",loss.item())
            ###log
            writer.add_scalar('Training/BatchLoss', loss.item(), global_step=step + epoch*epoch_len)
            #writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), step + epoch*epoch_len)
            #writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), step + epoch*epoch_len)
            #writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), step + epoch*epoch_len)
            #writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), step + epoch*epoch_len)
            #writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), step + epoch*epoch_len)
            #writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), step + epoch*epoch_len)
        
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("Training/EpochLoss", epoch_loss, epoch)
        #add epoch loss here
        
        if not epoch == 0 and epoch % Training_config.val_interval == 0: #
            model.eval()
            model.is_training=False
            with torch.no_grad():
                val_outputs = None
                metric={}
                for val_data in val_loader:
                    val_input = val_data[0]
                    val_input = val_input.to(device)
                    val_labels = val_data[1].to(device)                        
                    roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                    sw_batch_size = 1

                    val_outputs = sliding_window_inference(val_input, roi_size, sw_batch_size, model)
                    #val_outputs = torch.argmax(val_outputs, dim=1)

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
                #print("ET:",metric["dice_ET"])
                #print("TC:",metric["dice_TC"])
                #print("WT:",metric["dice_WT"])
                writer.add_scalar(f"Validation/BRATS18/dice_ET", metric["dice_ET"] , epoch)
                writer.add_scalar(f"Validation/BRATS18/dice_TC", metric["dice_TC"] , epoch)
                writer.add_scalar(f"Validation/BRATS18/dice_WT", metric["dice_WT"] , epoch)
