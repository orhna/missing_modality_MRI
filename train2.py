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

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=str)
    return parser.parse_args()

def main():
    """
    Main training and validation loop for multimodal SwinUNETR with memory.
    """
    torch.multiprocessing.set_sharing_strategy('file_system') 
    args = parse_args()
    device_list = list(map(int, args.device_id.split(',')))   

    # Load config
    training_config = config.Training_config()
    database_config = config.Database_config()
    _date = datetime.now().strftime("%d-%H-%M")
    log_dir = f"./logs/memory_joint/{training_config.experiment_name}_{_date}"
    writer = SummaryWriter(log_dir=log_dir)

    # Save config and set seed
    utils.save_config_from_py("config.py", log_dir)
    utils.set_random_seed(training_config.seed)

    cropped_input_size = training_config.cropped_input_size
    epochs = training_config.epoch
    print("lr: ", training_config.lr)
    print("Workers: ", training_config.workers)
    print("Batch size: ", training_config.train_batch_size)

    img_index = 0
    label_index = 1 

    chosen_ds = training_config.dataset_to_train[0]

    # Define the modalities we are actually using for the model
    modalities_for_model = ["T1c", "T1"] 
    num_model_modalities = len(modalities_for_model)
    print("modalities to train: ", modalities_for_model)

    img_path = database_config.img_path[chosen_ds]
    seg_path = database_config.seg_path[chosen_ds]
    model_save_path = training_config.model_save_path + training_config.experiment_name + "/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    images = sorted(glob(os.path.join(img_path, "*.*")))
    segs = sorted(glob(os.path.join(seg_path, "*.*"))) 

    channel_indices = []
    modalities_to_train = training_config.modalities_to_train
    for _m in modalities_to_train:
        channel_indices.append(database_config.channels[chosen_ds].index(_m))
    print("Channel index order to be loaded :", channel_indices)

    train_loader, val_loader = utils.get_loaders(
        images=images,
        segs=segs,
        train_file=database_config.split_path[chosen_ds]["train"],
        val_file=database_config.split_path[chosen_ds]["val"],
        workers=training_config.workers,
        train_batch_size=training_config.train_batch_size,
        cropped_input_size=cropped_input_size,
        channel_indices=channel_indices
    )

    train_size = len(train_loader.dataset)
    print("size of train", len(train_loader.dataset))
    print("size of val", len(val_loader.dataset))

    device = utils.initialize_GPU(device_list[0])

    # Initialize loss and metrics
    loss_function, dice_metric_c, sensitivity_metric, precision_metric, IOU_metric, post_trans = utils.initialize_loss_metric()
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", get_not_nans=False, percentile=95)
    loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False)

    binary_combinations_for_missing_scenarios = [
        "".join(map(str, bits)) for bits in itertools.product([0, 1], repeat=num_model_modalities)
    ]
    binary_combinations_for_missing_scenarios = [
        comb for comb in binary_combinations_for_missing_scenarios if comb not in ["0" * num_model_modalities, "1" * num_model_modalities]
    ]
    dice_dict_m = {comb: DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False) 
                   for comb in binary_combinations_for_missing_scenarios}

    epoched = 0
    model = Multimodal_SwinUNETR(
        img_size=(cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),
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

    if training_config.continue_training:
        print("Continuing training from checkpoint")
        checkpoint = torch.load(training_config.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoched = checkpoint['epoch'] + 1   

    best_metric_ET = -1
    best_metric_WT = -1
    best_metric_TC = -1

    # Training loop
    for epoch in tqdm(range(epoched, epochs)):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        model.is_training = True
        epoch_loss = 0
        epoch_ds_loss = 0
        epoch_sep_dec_loss = 0
        step = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            step += 1
            label = batch[label_index].to(device)
            # Select T1c and T1 for the model
            selected_input_data = batch[img_index][:, 1:3, :, :, :] # Shape: (B, 2, D, H, W)
            processed_input_data, d_m = utils.rand_drop_channel(
                modalities_for_model,
                selected_input_data,
                mode="modmean"
            )
            input_data = processed_input_data.to(device)    

            out, ds_outs, sep_out = model(input_data, modalities_dropped_info=d_m)
            loss = loss_function(out, label)
            ds_loss_0 = loss_function(ds_outs[0], label)
            ds_loss_1 = loss_function(ds_outs[1], label)
            ds_loss_2 = loss_function(ds_outs[2], label)
            ds_loss_3 = loss_function(ds_outs[3], label)
            ds_loss_4 = loss_function(ds_outs[4], label)
            t1c_loss = loss_function(sep_out[0], label)
            t1_loss = loss_function(sep_out[1], label)
            sep_losses = t1c_loss + t1_loss 
            _w = 0.2
            ds_loss = _w * ds_loss_0 + _w * ds_loss_1 + _w * ds_loss_2 + _w * ds_loss_3 + _w * ds_loss_4
            total_loss = loss + ds_loss + sep_losses

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            epoch_len = train_size // training_config.train_batch_size
            epoch_ds_loss += ds_loss.item()
            epoch_sep_dec_loss += sep_losses.item()
            print(f"{step}/{epoch_len}, train_loss: {total_loss.item():.4f}")

        if training_config.lr_scheduler:
            scheduler.step()
            writer.add_scalar("Training/EpochLR", optimizer.param_groups[0]['lr'], epoch) 

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("Training/EpochLoss", epoch_loss, epoch)
        epoch_ds_loss /= step
        writer.add_scalar("Training/EpochLossDS", epoch_ds_loss, epoch)
        epoch_sep_dec_loss /= step
        writer.add_scalar("Training/EpochLossSD", epoch_sep_dec_loss, epoch)

        # Save model
        if not epoch == 0 and epoch % 5 == 0:            
            model_save_name = model_save_path + training_config.experiment_name + "_Epoch_" + str(epoch) + ".pth"
            opt_save_name = model_save_path + training_config.experiment_name + "_checkpoint_Epoch_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), model_save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }, opt_save_name)
            print("Saved Model")

        # Validation
        if epoch % training_config.val_interval == 0:
            model.eval()
            model.is_training = False
            with torch.no_grad():
                metric_c = {} 
                metric_m = {}
                dict_to_save = {}
                for val_data in val_loader:
                    val_input_all_modalities = val_data[0] # (B, 4, D, H, W)
                    current_val_input = val_input_all_modalities[:, 1:3, :, :, :].to(device) # (B, 2, D, H, W)
                    val_labels = val_data[1].to(device)                        
                    roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                    c_logits = sliding_window_inference(current_val_input, roi_size=roi_size, sw_batch_size=1, predictor = lambda x: model(x, modalities_dropped_info=[]))
                    c_outputs = [post_trans(i) for i in decollate_batch(c_logits)]
                    dice_metric_c(y_pred=c_outputs, y=val_labels)
                    sensitivity_metric(y_pred=c_outputs, y=val_labels)
                    precision_metric(y_pred=c_outputs, y=val_labels)      
                    IOU_metric(y_pred=c_outputs, y=val_labels)
                    hausdorff_metric(y_pred=c_outputs, y=val_labels)

                    # Scenario: Only T1c present (T1 missing)
                    modalities_to_drop_t1_missing = [1] 
                    t1c_only_image = utils.drop_modality_image_channel(
                        current_val_input, method="modality_mean",
                        idx_to_drop=modalities_to_drop_t1_missing, remaining_modalities=[0]
                    )
                    m_logits_t1c_only = sliding_window_inference(
                        t1c_only_image, roi_size=roi_size, sw_batch_size=1,
                        predictor=lambda x: model(x, modalities_dropped_info=modalities_to_drop_t1_missing)
                    )
                    m_outputs_t1c_only = [post_trans(i) for i in decollate_batch(m_logits_t1c_only)]
                    dice_dict_m["10"](y_pred=m_outputs_t1c_only, y=val_labels)

                    # Scenario: Only T1 present (T1c missing)
                    modalities_to_drop_t1c_missing = [0]
                    t1_only_image = utils.drop_modality_image_channel(
                        current_val_input, method="modality_mean",
                        idx_to_drop=modalities_to_drop_t1c_missing, remaining_modalities=[1]
                    )
                    m_logits_t1_only = sliding_window_inference(
                        t1_only_image, roi_size=roi_size, sw_batch_size=1,
                        predictor=lambda x: model(x, modalities_dropped_info=modalities_to_drop_t1c_missing)
                    )
                    m_outputs_t1_only = [post_trans(i) for i in decollate_batch(m_logits_t1_only)]
                    dice_dict_m["01"](y_pred=m_outputs_t1_only, y=val_labels)

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
                for sc_key in dice_dict_m:
                    dice_dict_m[sc_key].reset()

                metric_c["dice"] = dice_metric_c.aggregate()
                metric_c["hausdorff_distance"] = hausdorff_metric.aggregate()
                metric_c["sensitivity"] = sensitivity_metric.aggregate()[0].item()
                metric_c["precision"] = precision_metric.aggregate()[0].item()                    
                metric_c["IOU"] = IOU_metric.aggregate().item()  
                metric_c["dice_ET"] = dice_metric_c.aggregate()[2].item()
                metric_c["dice_TC"] = dice_metric_c.aggregate()[0].item()
                metric_c["dice_WT"] = dice_metric_c.aggregate()[1].item()
                metric_c["HD_ET"] = hausdorff_metric.aggregate()[2].item()
                metric_c["HD_TC"] = hausdorff_metric.aggregate()[0].item()
                metric_c["HD_WT"] = hausdorff_metric.aggregate()[1].item()
                utils.log_metrics(writer, metric_c, epoch, chosen_ds)

                dict_to_save["complete"] = {
                    "Dice_M_ET": metric_c["dice_ET"],
                    "Dice_M_TC": metric_c["dice_TC"],
                    "Dice_M_WT": metric_c["dice_WT"]
                }
                values = [inner_dict["Dice_M_ET"] for inner_dict in dict_to_save.values()]
                avg_ET = sum(values) / len(values)
                values = [inner_dict["Dice_M_TC"] for inner_dict in dict_to_save.values()]
                avg_TC = sum(values) / len(values)
                values = [inner_dict["Dice_M_WT"] for inner_dict in dict_to_save.values()]
                avg_WT = sum(values) / len(values)
                writer.add_scalar(f"Val/Avg/Dice_M_ET", avg_ET, epoch)
                writer.add_scalar(f"Val/Avg/Dice_M_TC", avg_TC, epoch)
                writer.add_scalar(f"Val/Avg/Dice_M_WT", avg_WT, epoch)
                dict_to_save["average"] = {
                    "Dice_ET": avg_ET,
                    "Dice_TC": avg_TC,
                    "Dice_WT": avg_WT
                }

                dice_metric_c.reset()
                hausdorff_metric.reset()
                sensitivity_metric.reset()
                precision_metric.reset()                    
                IOU_metric.reset()

                json_dump_path = os.path.dirname(model_save_path) + f"/{str(epoch)}_dice_results.json"
                with open(json_dump_path, "w") as json_file:
                    json.dump(dict_to_save, json_file, indent=4)

                if avg_TC > best_metric_TC:
                    best_metric_TC = avg_TC
                    if epoch > 1:
                        model_save_name = model_save_path + training_config.experiment_name + "_BEST_TC.pth"
                        torch.save(model.state_dict(), model_save_name)                   
                if avg_WT > best_metric_WT:
                    best_metric_WT = avg_WT
                    if epoch > 1:
                        model_save_name = model_save_path + training_config.experiment_name + "_BEST_WT.pth"
                        torch.save(model.state_dict(), model_save_name)                   
                if avg_ET > best_metric_ET:
                    best_metric_ET = avg_ET
                    if epoch > 1:
                        model_save_name = model_save_path + training_config.experiment_name + "_BEST_ET.pth"
                        torch.save(model.state_dict(), model_save_name)                   

    writer.close() 
    print(training_config.experiment_name)

if __name__ == "__main__":
    main()