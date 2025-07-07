import os
import json
import shutil
import tempfile
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from datetime import datetime
from glob import glob

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)
import utils.utils as utils
import config
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial
from tensorboardX import SummaryWriter

import torch


print_config()

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


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system') 

    #command line argument
    parser = argparse.ArgumentParser()
    #parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--device_id", help="ID of the GPU", type=str)
    args = parser.parse_args()
    device_list = list(map(int, args.device_id.split(',')))   
    device = utils.initialize_GPU(device_list[0])
    #load config
    Training_config = config.Training_config()
    Database_config = config.Database_config()
    _date = datetime.now().strftime("%d-%H-%M")
    log_dir = f"./logs/{Training_config.experiment_name}_{_date}"
    writer = SummaryWriter(log_dir=log_dir)

    
    model_save_path = Training_config.model_save_path + Training_config.experiment_name + "/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    load_model_path = Training_config.load_model_path
    #save config
    utils.save_config_from_py("config.py",log_dir)
    #set seed
    utils.set_random_seed(Training_config.seed)

    cropped_input_size = Training_config.cropped_input_size
    epochs = Training_config.epoch
    # path initialization 
    chosen_ds = Training_config.dataset_to_train[0]
    img_path=Database_config.img_path[chosen_ds]
    seg_path=Database_config.seg_path[chosen_ds]
    channel_indices = []
    modalities_to_train = Training_config.modalities_to_train
    for _m in modalities_to_train:
        channel_indices.append(Database_config.channels[chosen_ds].index(_m))
    print("Channel indices to be loaded :",channel_indices)
    images= sorted(glob(os.path.join(img_path, "*.*")))
    segs = sorted(glob(os.path.join(seg_path,"*.*"))) 

    #set index
    img_index = 0
    label_index = 1 
    # Set the data size and total modalities
    chosen_ds = Training_config.dataset_to_train[0]
    channels= Training_config.modalities_to_train
    train_size=Database_config.train_size[chosen_ds]
    total_size=Database_config.total_size[chosen_ds]
    train_loader, val_loader = utils.create_dataloader_original(images=images,
                                                            segs=segs,
                                                            train_file=Database_config.split_path[chosen_ds]["train"],
                                                            val_file=Database_config.split_path[chosen_ds]["val"] ,
                                                            workers=Training_config.workers,
                                                            train_batch_size=Training_config.train_batch_size,
                                                            cropped_input_size=cropped_input_size,
                                                            channel_indices=channel_indices)
    
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True).to(device)

    torch.backends.cudnn.benchmark = True
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[128, 128, 128],
        sw_batch_size=4,
        predictor=model,
        overlap=0.5,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def train_epoch(model, loader, optimizer, epoch, loss_func):
        model.train()
        start_time = time.time()
        run_loss = AverageMeter()
        for idx, batch_data in enumerate(loader):
            data, target = batch_data[img_index].to(device), batch_data[label_index].to(device)
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
            run_loss.update(loss.item(), n=2)
            print(
                "Epoch {}/{} {}/{}".format(epoch, epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
        return run_loss.avg


    def val_epoch(
        model,
        loader,
        epoch,
        acc_func,
        model_inferer=None,
        post_sigmoid=None,
        post_pred=None,
    ):
        model.eval()
        start_time = time.time()
        run_acc = AverageMeter()

        with torch.no_grad():
            for idx, batch_data in enumerate(loader):
                data, target = batch_data[img_index].to(device), batch_data[label_index].to(device)
                logits = model_inferer(data)
                val_labels_list = decollate_batch(target)
                val_outputs_list = decollate_batch(logits)
                val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
                acc_func.reset()
                acc_func(y_pred=val_output_convert, y=val_labels_list)
                acc, not_nans = acc_func.aggregate()
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                dice_tc = run_acc.avg[0]
                dice_wt = run_acc.avg[1]
                dice_et = run_acc.avg[2] 
                print(
                    "Val {}/{} {}/{}".format(epoch, epochs, idx, len(loader)),
                    ", dice_tc:",
                    dice_tc,
                    ", dice_wt:",
                    dice_wt,
                    ", dice_et:",
                    dice_et,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
                start_time = time.time()

        return run_acc.avg

    def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    ):
        val_acc_max = 0.0
        dices_tc = []
        dices_wt = []
        dices_et = []
        dices_avg = []
        loss_epochs = []
        trains_epoch = []
        for epoch in range(start_epoch, epochs):
            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                epoch=epoch,
                loss_func=loss_func,
            )
            print(
                "Final training  {}/{}".format(epoch, epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            if (epoch + 1) % Training_config.val_interval == 0 or epoch == 0:
                loss_epochs.append(train_loss)
                trains_epoch.append(int(epoch))
                epoch_time = time.time()
                val_acc = val_epoch(
                    model,
                    val_loader,
                    epoch=epoch,
                    acc_func=acc_func,
                    model_inferer=model_inferer,
                    post_sigmoid=post_sigmoid,
                    post_pred=post_pred,
                )
                dice_tc = val_acc[0]
                dice_wt = val_acc[1]
                dice_et = val_acc[2]
                val_avg_acc = np.mean(val_acc)
                print(
                    "Final validation stats {}/{}".format(epoch, epochs - 1),
                    ", dice_tc:",
                    dice_tc,
                    ", dice_wt:",
                    dice_wt,
                    ", dice_et:",
                    dice_et,
                    ", Dice_Avg:",
                    val_avg_acc,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                dices_tc.append(dice_tc)
                dices_wt.append(dice_wt)
                dices_et.append(dice_et)
                dices_avg.append(val_avg_acc)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    model_save_name = model_save_path + Training_config.experiment_name + "_Epoch_" + str(epoch) + ".pth"
                    opt_save_name=model_save_path + Training_config.experiment_name + "_checkpoint_Epoch_" + str(epoch) + ".pt"
                    torch.save(model.state_dict(), model_save_name)
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }, opt_save_name)
                    print("Saved Model")
                scheduler.step()
        print("Training Finished !, Best Accuracy: ", val_acc_max)
        return (
            val_acc_max,
            dices_tc,
            dices_wt,
            dices_et,
            dices_avg,
            loss_epochs,
            trains_epoch,
        )

start_epoch = 0

(
    val_acc_max,
    dices_tc,
    dices_wt,
    dices_et,
    dices_avg,
    loss_epochs,
    trains_epoch,
) = trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_func=dice_loss,
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_sigmoid=post_sigmoid,
    post_pred=post_pred,
)