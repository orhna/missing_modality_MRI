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
from monai.losses import DiceLoss
from monai.transforms import (
    AsDiscrete,
    Activations, Compose
)
from monai.networks.nets import SwinUNETR
from monai.networks.utils import one_hot
from monai.utils.enums import MetricReduction
from nets.unet import res_unet
from nets.swinunetrldm import SwinUNETRWithLDM
from nets.multimodal_swinunetr import Multimodal_SwinUNETR 
from nets.multimodal_swinunetr_shard import Multimodal_SwinUNETR_shard
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    
    torch.multiprocessing.set_sharing_strategy('file_system') 

    #command line argument
    parser = argparse.ArgumentParser()
    #parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
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
    log_dir = f"./logs/{Training_config.experiment_name}_{_date}"
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

    print("data_size",train_size)
    print("modalities to train: ", channels)

    # path initialization 
    img_path=Database_config.img_path[chosen_ds]
    seg_path=Database_config.seg_path[chosen_ds]

    model_save_path = Training_config.model_save_path + Training_config.experiment_name + "/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    load_model_path = Training_config.load_model_path

    val_size = total_size-train_size
    images= sorted(glob(os.path.join(img_path, "*.*")))
    segs = sorted(glob(os.path.join(seg_path,"*.*"))) 

    # create dict to count dropped modalities frequency
    counter_dict = utils.create_counter_dict()

    channel_indices = []
    modalities_to_train = Training_config.modalities_to_train
    for _m in modalities_to_train:
        channel_indices.append(Database_config.channels[chosen_ds].index(_m))
    print("Channel indices to be loaded :",channel_indices)

    train_loader, val_loader = utils.create_dataloader_original(images=images,
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

    model = SwinUNETR(
        img_size=Training_config.cropped_input_size[0],
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True).to(device)
    epoched=0
    # initialize loss and metrics according to output channel
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Training_config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # initialize the best metric
    best_metric=-1
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
        epoch_loss = 0
        step = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            step += 1
            label = batch[label_index].to(device)
            # initalize input data
            if Training_config.random_modality_drop:
                batch[img_index], d_m  = utils.rand_drop_channel(modalities_to_train, batch[img_index])
                input_data = batch[img_index].to(device)    
                counter_dict[d_m] +=1 
            else:
                input_data = batch[img_index].to(device)

            out = model(input_data)
            #print(out.shape) #[1, 3, 128, 128, 128]
            #print(out[:, 0].shape) # [1, 128, 128, 128]

            loss = dice_loss(out, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_len = train_size  // Training_config.train_batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("Training/BatchLoss", loss.item(), step + epoch*epoch_len)
        
        if Training_config.lr_scheduler:
            scheduler.step()
            writer.add_scalar("Training/EpochLR", optimizer.param_groups[0]['lr'], epoch) 

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("Training/EpochLoss", loss.item(), epoch)

        # save model
        if not epoch == 0 and epoch % 50 == 0:            
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
        if not epoch == 0 and epoch % Training_config.val_interval == 0:

            run_acc = utils.AverageMeter()
            model.eval()
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
                    dice_acc(y_pred=val_outputs, y=val_labels)
                    acc, not_nans = dice_acc.aggregate()
                    run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                dice_et = run_acc.avg[0]
                dice_tc = run_acc.avg[1]
                dice_wt = run_acc.avg[2] 
                
                
                #metric["dice"] = dice_acc.aggregate()
                #metric["dice_ET"] = dice_acc[0]
                #metric["dice_TC"] = dice_acc[1]
                #metric["dice_WT"] = dice_acc[2]

                # val_input : torch.Size([1, 4, 240, 240, 155])
                # val_outputs[0] : torch.Size([1, 240, 240, 155])
                # val_labels : torch.Size([1, 1, 240, 240, 155])
                #writer.add_scalar(f"Validation/BRATS18/dice", metric["dice"], epoch)
                writer.add_scalar(f"Validation/BRATS18/dice_ET", dice_et, epoch)
                writer.add_scalar(f"Validation/BRATS18/dice_TC", dice_tc, epoch)
                writer.add_scalar(f"Validation/BRATS18/dice_WT", dice_wt, epoch)

                dice_acc.reset()
                """
                if metric["dice"] > best_metric:
                    best_metric_TC = metric["dice"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_avg_dice.pth"
                        torch.save(model.state_dict(), model_save_name)   
                
                if metric["dice_TC"] > best_metric_TC:
                    best_metric_TC = metric["dice_TC"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_TC.pth"
                        torch.save(model.state_dict(), model_save_name)                   
                if metric["dice_WT"] > best_metric_WT:
                    best_metric_WT = metric["dice_WT"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_WT.pth"
                        torch.save(model.state_dict(), model_save_name)                   
                if metric["dice_ET"] > best_metric_ET:
                    best_metric_ET = metric["dice_ET"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_ET.pth"
                        torch.save(model.state_dict(), model_save_name)                   
                
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice {:.4f} at epoch {}".format(
                        epoch + 1, metric["dice"], best_metric, best_metric_epoch
                    )
                ) 
                """  
    writer.close() 
    utils.export_counter_dict(counter_dict, log_dir)
    print(Training_config.experiment_name)