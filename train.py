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
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from nets.nnunet import CustomUNet
from nets.multimodal_swinunetr import Multimodal_SwinUNETR 
from nets.multimodal_swinunetr_shard import Multimodal_SwinUNETR_shard
from torch.optim.lr_scheduler import StepLR, LambdaLR
from monai.losses.dice import GeneralizedDiceLoss, DiceCELoss

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
    log_dir = f"./logs/testasval/{Training_config.experiment_name}_{_date}"
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

    train_loader, val_loader = utils.get_loaders(images=images,
                                                segs=segs,
                                                train_file=Database_config.split_path[chosen_ds]["train"],
                                                val_file=Database_config.split_path[chosen_ds]["val"] ,
                                                workers=Training_config.workers,
                                                train_batch_size=Training_config.train_batch_size,
                                                cropped_input_size=cropped_input_size,
                                                channel_indices=channel_indices)
  
    print("size of train",len(train_loader.dataset))
    print("size of val",len(val_loader.dataset))

    # initialize GPU
    if model_sharding:
        device_id_list = utils.initialize_GPUs(device_list)
        device = device_id_list[0]
    else:
        device = utils.initialize_GPU(device_list[0])

    # initialize loss and metrics according to output channel
    loss_function, dice_metric, sensitivity_metric, precision_metric, IOU_metric, post_trans = utils.initialize_loss_metric(Training_config.weighted_dice, Training_config.new_eval)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", get_not_nans=False, percentile=95)
    #loss_function = GeneralizedDiceLoss(to_onehot_y=False, sigmoid=True)
    loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False)

    if Training_config.model_type == "SwinUNETR":
        print("TRAINING WITH SwinUNETR")
        model = SwinUNETR(
            img_size=(cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),
            in_channels=len(channels), 
            out_channels=Training_config.output_channel,
            feature_size=Training_config.feature_size,
            use_checkpoint=True).to(device)
        if Training_config.continue_training:
            print("Continuing training from checkpoint")
            model_dict = torch.load(Training_config.checkpoint_path)["state_dict"]
            model.load_state_dict(model_dict)
            model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Training_config.lr)
        epoched=0
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    elif Training_config.model_type == "MultiModal_SwinUNETR":
        epoched=0
        if not model_sharding:
            print("TRAINING WITH MultiModal_SwinUNETR")
            model = Multimodal_SwinUNETR(
                img_size=(cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),
                in_channels=1, 
                out_channels=Training_config.output_channel,
                feature_size=Training_config.feature_size,
                use_checkpoint=True,
                deep_supervision=Training_config.deep_supervision,
                sep_dec=Training_config.sep_dec,
                tp_conv=Training_config.tp_conv,
                dec_upsample=Training_config.dec_upsample).to(device)
        else:
            print("TRAINING WITH MultiModal_SwinUNETR shards")
            model = Multimodal_SwinUNETR_shard(
                img_size=(cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),
                in_channels=1, 
                out_channels=Training_config.output_channel,
                device_list=device_id_list,
                feature_size=Training_config.feature_size,
                use_checkpoint=True,
                cross_attention=Training_config.cross_attention,
                deep_supervision=Training_config.deep_supervision,
                t1c_spec=Training_config.t1c_spec)
        
        #optimizer = torch.optim.Adam(model.parameters(), lr=Training_config.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Training_config.lr, weight_decay=Training_config.weight_decay)
        scheduler = StepLR(optimizer, step_size=Training_config.sch_step_size, gamma=Training_config.sch_gamma)

        if Training_config.continue_training:
            print("Continuing training from checkpoint")
            checkpoint = torch.load(Training_config.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoched=checkpoint['epoch']+1   
    elif Training_config.model_type == "nnunet":
        epoched=0
        model = CustomUNet(in_channels=4, out_channels=3).to(device)
        #optimizer = torch.optim.SGD(model.parameters(), lr=Training_config.lr, momentum=0.99, nesterov=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=Training_config.lr)
        #optimizer = torch.optim.AdamW(model.parameters(), lr=Training_config.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)


    # initialize the best metric
    best_metric_ET=-1
    best_metric_WT=-1
    best_metric_TC=-1
    best_metric_epoch=-1
    metric_values = list()

    #training
    for epoch in tqdm(range(epoched,epochs)):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        model.is_training = True
        epoch_loss = 0
        epoch_ds_loss=0
        epoch_t1c_spec_loss=0
        epoch_sep_dec_loss=0
        epoch_tp_conv_loss=0
        epoch_dec_upsample_loss=0
        step = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            step += 1
            label = batch[label_index].to(device)
            # initalize input data
            if Training_config.random_modality_drop:
                batch[img_index], d_m  = utils.rand_drop_channel_new(modalities_to_train,
                                                                      batch[img_index],
                                                                      mode="modmean")
                input_data = batch[img_index].to(device)    
                counter_dict[d_m] +=1 
                model.d_m = d_m
            else:
                input_data = batch[img_index].to(device)


            if Training_config.deep_supervision:
                if Training_config.sep_dec:
                    if Training_config.tp_conv and model.d_m == "no_drop":
                        out, ds_outs, sep_out, tp_conv_losses = model(input_data)
                    elif Training_config.dec_upsample and model.d_m == "no_drop":
                        out, ds_outs, sep_out, dec_upsample_losses = model(input_data)
                    else:
                        out, ds_outs, sep_out = model(input_data)

                    loss = loss_function(out, label)
                    ds_loss_0 = loss_function(ds_outs[0], label)
                    ds_loss_1 = loss_function(ds_outs[1], label)
                    ds_loss_2 = loss_function(ds_outs[2], label)
                    ds_loss_3 = loss_function(ds_outs[3], label)
                    ds_loss_4 = loss_function(ds_outs[4], label)
                    
                    f_loss = loss_function(sep_out[0], label)
                    t1c_loss = loss_function(sep_out[1], label)
                    t1_loss = loss_function(sep_out[2], label)
                    t2_loss = loss_function(sep_out[3], label)
                    sep_losses = f_loss + t1c_loss + t1_loss +t2_loss
                    _w = 0.2
                    ds_loss = _w* ds_loss_0 + _w*ds_loss_1 + _w*ds_loss_2 + _w*ds_loss_3 + _w*ds_loss_4
                    total_loss = loss + ds_loss + sep_losses

                    if Training_config.tp_conv and model.d_m == "no_drop":
                        tp_conv_loss_list=[] 
                        for mse_loss_modlist in tp_conv_losses:
                            tp_conv_loss_list.append(0.25 * torch.stack(mse_loss_modlist).sum())
                        tp_conv_loss = 0.25* torch.stack(tp_conv_loss_list).sum()
                        total_loss += tp_conv_loss

                    elif Training_config.dec_upsample and model.d_m == "no_drop":
                        dec_upsample_loss = 0.2* torch.stack(dec_upsample_losses).sum()
                        total_loss += dec_upsample_loss

                else:
                    out, ds_outs = model(input_data)
                    loss = loss_function(out, label)
                    ds_loss_0 = loss_function(ds_outs[0], label)
                    ds_loss_1 = loss_function(ds_outs[1], label)
                    ds_loss_2 = loss_function(ds_outs[2], label)
                    ds_loss_3 = loss_function(ds_outs[3], label)
                    ds_loss_4 = loss_function(ds_outs[4], label)
                    _w = 0.2
                    ds_loss = _w* ds_loss_0 + _w*ds_loss_1 + _w*ds_loss_2 + _w*ds_loss_3 + _w*ds_loss_4
                    total_loss = loss + ds_loss

            else:
                if Training_config.t1c_spec:
                    out, t1c_out = model(input_data)
                    loss = loss_function(out, label)
                    t1c_loss = loss_function(t1c_out, label[:,1,...].unsqueeze(0))
                    total_loss = loss + t1c_loss
                elif Training_config.sep_dec:
                    out, sep_out = model(input_data)
                    loss = loss_function(out, label)
                    f_loss = loss_function(sep_out[0], label)
                    t1c_loss = loss_function(sep_out[1], label)
                    t1_loss = loss_function(sep_out[2], label)
                    t2_loss = loss_function(sep_out[3], label)
                    sep_losses = f_loss + t1c_loss + t1_loss + t2_loss
                    total_loss = loss + (sep_losses/4)
                else:
                    out = model(input_data)
                    total_loss = loss_function(out, label)

            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_len = train_size  // Training_config.train_batch_size

            if Training_config.deep_supervision:
                epoch_ds_loss += ds_loss.item()
            if Training_config.sep_dec:
                epoch_sep_dec_loss += sep_losses.item()
            if Training_config.tp_conv and model.d_m == "no_drop":
                epoch_tp_conv_loss += tp_conv_loss.item()
            if Training_config.dec_upsample and model.d_m == "no_drop":
                epoch_dec_upsample_loss += dec_upsample_loss.item()

            print(f"{step}/{epoch_len}, train_loss: {total_loss.item():.4f}")
            #writer.add_scalar("Training/BatchLoss", total_loss.item(), step + epoch*epoch_len)
        
        if Training_config.lr_scheduler:
            scheduler.step()
            writer.add_scalar("Training/EpochLR", optimizer.param_groups[0]['lr'], epoch) 

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("Training/EpochLoss", epoch_loss, epoch)
        
        if Training_config.deep_supervision:
            epoch_ds_loss /= step
            writer.add_scalar("Training/EpochLossDS", epoch_ds_loss, epoch)
        if Training_config.sep_dec:
            epoch_sep_dec_loss /= step
            writer.add_scalar("Training/EpochLossSD", epoch_sep_dec_loss, epoch)
        if Training_config.tp_conv and model.d_m == "no_drop":
            epoch_tp_conv_loss /= step
            writer.add_scalar("Training/EpochLossTPconv", epoch_tp_conv_loss, epoch)
        if Training_config.dec_upsample and model.d_m == "no_drop":
            epoch_dec_upsample_loss /= step
            writer.add_scalar("Training/EpochLossDecUpsample", epoch_dec_upsample_loss, epoch)

        # save model
        if not epoch == 0 and epoch % 5 == 0:            
            model_save_name = model_save_path + Training_config.experiment_name + "_Epoch_" + str(epoch) + ".pth"
            opt_save_name=model_save_path + Training_config.experiment_name + "_checkpoint_Epoch_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), model_save_name)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss ,
            }, opt_save_name)
            print("Saved Model")
        
        # validation
        if epoch % Training_config.val_interval == 0:# not epoch == 0 and
            model.eval()
            model.is_training = False
            with torch.no_grad():
                val_outputs = None
                metric={}
            
                for val_data in val_loader:
                    val_input = val_data[0]
                    val_input = val_input.to(device)
                    val_labels = val_data[1].to(device)                        
                    roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                    sw_batch_size = 1
                    #using sliding window for the whole 3D image
                    val_outputs = sliding_window_inference(val_input, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    sensitivity_metric(y_pred=val_outputs, y=val_labels)
                    precision_metric(y_pred=val_outputs, y=val_labels)      
                    IOU_metric(y_pred=val_outputs, y=val_labels)
                    hausdorff_metric(y_pred=val_outputs, y=val_labels)
                
                metric["dice"] = dice_metric.aggregate()
                metric["hausdorff_distance"] = hausdorff_metric.aggregate()
                metric["sensitivity"] = sensitivity_metric.aggregate()[0].item()
                metric["precision"] = precision_metric.aggregate()[0].item()                    
                metric["IOU"] = IOU_metric.aggregate().item()  
                metric["dice_ET"] = dice_metric.aggregate()[2].item()
                metric["dice_TC"] = dice_metric.aggregate()[0].item()
                metric["dice_WT"] = dice_metric.aggregate()[1].item()
                metric["HD_ET"] = hausdorff_metric.aggregate()[2].item()
                metric["HD_TC"] = hausdorff_metric.aggregate()[0].item()
                metric["HD_WT"] = hausdorff_metric.aggregate()[1].item()
                utils.log_visuals(modalities_to_train, val_input, val_outputs[0], val_labels, epoch, writer, chosen_ds)
                utils.log_metrics(writer, metric, epoch, chosen_ds)

                dice_metric.reset()
                hausdorff_metric.reset()
                sensitivity_metric.reset()
                precision_metric.reset()                    
                IOU_metric.reset()

                if metric["dice_TC"] > best_metric_TC:
                    best_metric = metric["dice_TC"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_TC.pth"
                        torch.save(model.state_dict(), model_save_name)                   
                if metric["dice_WT"] > best_metric_WT:
                    best_metric = metric["dice_WT"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_WT.pth"
                        torch.save(model.state_dict(), model_save_name)                   
                if metric["dice_ET"] > best_metric_ET:
                    best_metric = metric["dice_ET"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_ET.pth"
                        torch.save(model.state_dict(), model_save_name)                   

    writer.close() 
    utils.export_counter_dict(counter_dict, log_dir)
    print(Training_config.experiment_name)