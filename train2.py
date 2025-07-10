import torch
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
import utils.utils as utils
import config
import argparse
import json
import itertools
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from nets.mm_memory2 import Multimodal_SwinUNETR
from torch.optim.lr_scheduler import StepLR
from monai.losses.dice import DiceCELoss
from tensorboardX import SummaryWriter

def train_one_epoch(model, train_loader, optimizer, loss_function, device, epoch, writer, training_config):
    """
    Trains the model for one epoch.
    """
    model.train()
    model.is_training = True
    epoch_loss = 0
    epoch_ds_loss = 0
    epoch_sep_dec_loss = 0
    step = 0
    epoch_len = len(train_loader)
    
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{training_config.epoch}"):
        optimizer.zero_grad()
        step += 1
        label = batch[1].to(device)
        
        selected_input_data = batch[0][:, 1:3, :, :, :]
        processed_input_data, d_m = utils.rand_drop_channel(
            ["T1c", "T1"], 
            selected_input_data,
            mode="modmean"
        )
        input_data = processed_input_data.to(device)

        out, ds_outs, sep_out = model(input_data, modalities_dropped_info=d_m)
        
        loss = loss_function(out, label)
        ds_losses = [loss_function(ds_out, label) for ds_out in ds_outs]
        ds_loss = sum(ds_losses) * 0.2

        sep_losses_list = [loss_function(sep, label) for sep in sep_out]
        sep_losses = sum(sep_losses_list)

        total_loss = loss + ds_loss + sep_losses
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_ds_loss += ds_loss.item()
        epoch_sep_dec_loss += sep_losses.item()

        print(f"{step}/{epoch_len}, train_loss: {total_loss.item():.4f}")

    if training_config.lr_scheduler:
        writer.add_scalar("Training/EpochLR", optimizer.param_groups[0]['lr'], epoch)

    epoch_loss /= step
    epoch_ds_loss /= step
    epoch_sep_dec_loss /= step
    
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    writer.add_scalar("Training/EpochLoss", epoch_loss, epoch)
    writer.add_scalar("Training/EpochLossDS", epoch_ds_loss, epoch)
    writer.add_scalar("Training/EpochLossSD", epoch_sep_dec_loss, epoch)

    return epoch_loss

def validate_one_epoch(model, val_loader, post_trans, dice_metric_c, hausdorff_metric, sensitivity_metric, 
                       precision_metric, IOU_metric, dice_dict_m, epoch, writer, device, training_config, 
                       model_save_path, best_metrics):
    """
    Validates the model for one epoch.
    """
    model.eval()
    model.is_training = False
    best_metric_ET, best_metric_WT, best_metric_TC = best_metrics

    with torch.no_grad():
        metric_c = {}
        metric_m = {}
        dict_to_save = {}
        
        for val_data in tqdm(val_loader, desc="Validating"):
            val_input_all_modalities = val_data[0]
            current_val_input = val_input_all_modalities[:, 1:3, :, :, :].to(device)
            val_labels = val_data[1].to(device)
            roi_size = training_config.cropped_input_size

            c_logits = sliding_window_inference(current_val_input, roi_size, sw_batch_size=1, 
                                                predictor=lambda x: model(x, modalities_dropped_info=[]))
            c_outputs = [post_trans(i) for i in decollate_batch(c_logits)]
            
            dice_metric_c(y_pred=c_outputs, y=val_labels)
            sensitivity_metric(y_pred=c_outputs, y=val_labels)
            precision_metric(y_pred=c_outputs, y=val_labels)
            IOU_metric(y_pred=c_outputs, y=val_labels)
            hausdorff_metric(y_pred=c_outputs, y=val_labels)

            scenarios = {
                "10": [1],  # T1 missing
                "01": [0]   # T1c missing
            }
            for key, drop_indices in scenarios.items():
                remaining_indices = [i for i in range(2) if i not in drop_indices]
                missing_mod_img = utils.drop_modality_image_channel(
                    current_val_input, method="modality_mean", 
                    idx_to_drop=drop_indices, remaining_modalities=remaining_indices
                )
                m_logits = sliding_window_inference(
                    missing_mod_img, roi_size, sw_batch_size=1,
                    predictor=lambda x: model(x, modalities_dropped_info=drop_indices)
                )
                m_outputs = [post_trans(i) for i in decollate_batch(m_logits)]
                dice_dict_m[key](y_pred=m_outputs, y=val_labels)

        for sc_key in dice_dict_m:
            metric_m["dice_" + sc_key] = dice_dict_m[sc_key].aggregate()
            writer.add_scalar(f"Val/{sc_key}/Dice_M_ET", metric_m["dice_" + sc_key][2].item(), epoch)
            writer.add_scalar(f"Val/{sc_key}/Dice_M_TC", metric_m["dice_" + sc_key][0].item(), epoch)
            writer.add_scalar(f"Val/{sc_key}/Dice_M_WT", metric_m["dice_" + sc_key][1].item(), epoch)
            dict_to_save[sc_key] = {
                "Dice_M_ET": metric_m["dice_" + sc_key][2].item(),
                "Dice_M_TC": metric_m["dice_" + sc_key][0].item(),
                "Dice_M_WT": metric_m["dice_" + sc_key][1].item()
            }
            dice_dict_m[sc_key].reset()

        metric_c["dice"] = dice_metric_c.aggregate()
        metric_c["hausdorff_distance"] = hausdorff_metric.aggregate()
        metric_c["sensitivity"] = sensitivity_metric.aggregate()[0].item()
        metric_c["precision"] = precision_metric.aggregate()[0].item()
        metric_c["IOU"] = IOU_metric.aggregate().item()
        metric_c["dice_ET"] = metric_c["dice"][2].item()
        metric_c["dice_TC"] = metric_c["dice"][0].item()
        metric_c["dice_WT"] = metric_c["dice"][1].item()
        metric_c["HD_ET"] = metric_c["hausdorff_distance"][2].item()
        metric_c["HD_TC"] = metric_c["hausdorff_distance"][0].item()
        metric_c["HD_WT"] = metric_c["hausdorff_distance"][1].item()
        
        utils.log_metrics(writer, metric_c, epoch, training_config.dataset_to_train[0])

        dict_to_save["complete"] = {
            "Dice_M_ET": metric_c["dice_ET"],
            "Dice_M_TC": metric_c["dice_TC"],
            "Dice_M_WT": metric_c["dice_WT"]
        }

        avg_ET = sum(d["Dice_M_ET"] for d in dict_to_save.values()) / len(dict_to_save)
        avg_TC = sum(d["Dice_M_TC"] for d in dict_to_save.values()) / len(dict_to_save)
        avg_WT = sum(d["Dice_M_WT"] for d in dict_to_save.values()) / len(dict_to_save)
        
        writer.add_scalar("Val/Avg/Dice_M_ET", avg_ET, epoch)
        writer.add_scalar("Val/Avg/Dice_M_TC", avg_TC, epoch)
        writer.add_scalar("Val/Avg/Dice_M_WT", avg_WT, epoch)
        
        dict_to_save["average"] = {"Dice_ET": avg_ET, "Dice_TC": avg_TC, "Dice_WT": avg_WT}

        dice_metric_c.reset()
        hausdorff_metric.reset()
        sensitivity_metric.reset()
        precision_metric.reset()
        IOU_metric.reset()

        json_dump_path = os.path.join(os.path.dirname(model_save_path), f"{epoch}_dice_results.json")
        with open(json_dump_path, "w") as json_file:
            json.dump(dict_to_save, json_file, indent=4)

        if avg_TC > best_metric_TC and epoch > 1:
            best_metric_TC = avg_TC
            torch.save(model.state_dict(), os.path.join(model_save_path, f"{training_config.experiment_name}_BEST_TC.pth"))
        if avg_WT > best_metric_WT and epoch > 1:
            best_metric_WT = avg_WT
            torch.save(model.state_dict(), os.path.join(model_save_path, f"{training_config.experiment_name}_BEST_WT.pth"))
        if avg_ET > best_metric_ET and epoch > 1:
            best_metric_ET = avg_ET
            torch.save(model.state_dict(), os.path.join(model_save_path, f"{training_config.experiment_name}_BEST_ET.pth"))
            
        return best_metric_ET, best_metric_WT, best_metric_TC

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=str, default="0")
    args = parser.parse_args()
    device_list = list(map(int, args.device_id.split(',')))

    training_config = config.Training_config()
    database_config = config.Database_config()

    _date = datetime.now().strftime("%d-%H-%M")
    log_dir = f"./logs/memory_joint/{training_config.experiment_name}_{_date}"
    writer = SummaryWriter(log_dir=log_dir)
    utils.save_config_from_py("config.py", log_dir)

    utils.set_random_seed(training_config.seed)

    print(f"lr: {training_config.lr}")
    print(f"Workers: {training_config.workers}")
    print(f"Batch size: {training_config.train_batch_size}")

    chosen_ds = training_config.dataset_to_train[0]
    modalities_for_model = ["T1c", "T1"]
    num_model_modalities = len(modalities_for_model)
    print(f"Modalities to train: {modalities_for_model}")

    img_path = database_config.img_path[chosen_ds]
    seg_path = database_config.seg_path[chosen_ds]
    images = sorted(glob(os.path.join(img_path, "*.*")))
    segs = sorted(glob(os.path.join(seg_path, "*.*")))
    
    channel_indices = [database_config.channels[chosen_ds].index(m) for m in training_config.modalities_to_train]
    print(f"Channel index order to be loaded: {channel_indices}")

    train_loader, val_loader = utils.get_loaders(
        images=images, segs=segs,
        train_file=database_config.split_path[chosen_ds]["train"],
        val_file=database_config.split_path[chosen_ds]["val"],
        workers=training_config.workers,
        train_batch_size=training_config.train_batch_size,
        cropped_input_size=training_config.cropped_input_size,
        channel_indices=channel_indices
    )
    print(f"Size of train dataset: {len(train_loader.dataset)}")
    print(f"Size of val dataset: {len(val_loader.dataset)}")

    device = utils.initialize_GPU(device_list[0])
    model = Multimodal_SwinUNETR(
        img_size=training_config.cropped_input_size,
        in_channels=1,
        out_channels=training_config.output_channel,
        feature_size=training_config.feature_size,
        num_modalities=num_model_modalities,
        use_checkpoint=True,
        device=device,
        use_memory=training_config.use_memory,
        memory_size=128,
        memory_feature_dim=512
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=training_config.sch_step_size, gamma=training_config.sch_gamma)

    loss_function, dice_metric_c, sensitivity_metric, precision_metric, IOU_metric, post_trans = utils.initialize_loss_metric(training_config.weighted_dice, training_config.new_eval)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", get_not_nans=False, percentile=95)
    
    binary_combinations = ["10", "01"]
    dice_dict_m = {comb: DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False) for comb in binary_combinations}

    start_epoch = 0
    if training_config.continue_training:
        print("Continuing training from checkpoint")
        checkpoint = torch.load(training_config.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    best_metric_ET, best_metric_WT, best_metric_TC = -1, -1, -1
    model_save_path = os.path.join(training_config.model_save_path, training_config.experiment_name)
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(start_epoch, training_config.epoch):
        train_one_epoch(model, train_loader, optimizer, loss_function, device, epoch, writer, training_config)
        
        if training_config.lr_scheduler:
            scheduler.step()

        if epoch % 5 == 0 and epoch != 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(model_save_path, f"{training_config.experiment_name}_checkpoint_Epoch_{epoch}.pt"))
            print("Saved Checkpoint")

        if epoch % training_config.val_interval == 0:
            best_metric_ET, best_metric_WT, best_metric_TC = validate_one_epoch(
                model, val_loader, post_trans, dice_metric_c, hausdorff_metric, sensitivity_metric,
                precision_metric, IOU_metric, dice_dict_m, epoch, writer, device, training_config,
                model_save_path, (best_metric_ET, best_metric_WT, best_metric_TC)
            )

    writer.close()
    print(f"Finished training for experiment: {training_config.experiment_name}")

if __name__ == "__main__":
    main()
