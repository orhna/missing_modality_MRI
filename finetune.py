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
from monai.transforms import Activations, AsDiscrete, Compose
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses.dice import GeneralizedDiceLoss, DiceCELoss
from nets.mm_3Dunet import mm_3Dunet, mm_3DUnetWrapper, drop_modality_image_channel, extract_complete_modality_features, drop_modality_feature_channel
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from monai.transforms import NormalizeIntensity

if __name__ == "__main__":

    #command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=str)
    args = parser.parse_args()
    device_list = list(map(int, args.device_id.split(',')))   
    device = utils.initialize_GPU(device_list[0])
        
    Training_config = config.Training_config()
    Database_config = config.Database_config()
    _date = datetime.now().strftime("%d-%H-%M")
    log_dir = f"./logs/recon/{Training_config.experiment_name}_{_date}"
    writer = SummaryWriter(log_dir=log_dir)

    #save config
    utils.save_config_from_py("config.py",log_dir)
    #set seed
    utils.set_random_seed(Training_config.seed)

    cropped_input_size = Training_config.cropped_input_size
    epochs = Training_config.epoch
    img_index, label_index = 0, 1

    chosen_ds = Training_config.dataset_to_train[0]
    channels = Training_config.modalities_to_train
    train_size = Database_config.train_size[chosen_ds]
    _bs = Training_config.train_batch_size 

    img_path=Database_config.img_path[chosen_ds]
    seg_path=Database_config.seg_path[chosen_ds]

    images= sorted(glob(os.path.join(img_path, "*.*")))
    segs = sorted(glob(os.path.join(seg_path,"*.*"))) 

    model_save_path = Training_config.model_save_path + Training_config.experiment_name + "/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    load_model_path = Training_config.load_model_path

    channel_indices = []
    modalities_to_train = Training_config.modalities_to_train
    for _m in modalities_to_train:
        channel_indices.append(Database_config.channels[chosen_ds].index(_m))
    print("Channel index order to be loaded :",channel_indices)

    train_loader, val_loader = utils.get_loaders(images=images,
                                            segs=segs,
                                            train_file=Database_config.split_path[chosen_ds]["train"],
                                            val_file=Database_config.split_path[chosen_ds]["val"],
                                            workers=Training_config.workers,
                                            train_batch_size=_bs,
                                            cropped_input_size=cropped_input_size,
                                            channel_indices=channel_indices)

    train_size = len(train_loader.dataset)
    print("size of train",len(train_loader.dataset))
    print("size of val",len(val_loader.dataset))

    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)

    rec_model = mm_3Dunet(Training_config, device=device)
    rec_model.load_swinunetr_weights(load_model_path)
    wrapper = mm_3DUnetWrapper(rec_model)
    wrapper.recon_level = Training_config.recon_level
    
    criterionMSE = torch.nn.MSELoss()
    criterionDiceCE = DiceCELoss(sigmoid=True, to_onehot_y=False)

    optimizer = torch.optim.Adam(rec_model.parameters(), lr=Training_config.lr)
    scheduler = StepLR(optimizer, step_size=Training_config.sch_step_size, gamma=Training_config.sch_gamma)

    dice_metric_c = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    hd_metric_c = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", get_not_nans=False, percentile=95)
    binary_combinations = [''.join(map(str, bits)) for bits in itertools.product([0, 1], repeat=4)]
    binary_combinations = [comb for comb in binary_combinations if comb != '0000' and comb != '1111']

    hd_dict_c_f_m_g = {comb: HausdorffDistanceMetric(include_background=True, reduction="mean_batch", get_not_nans=False, percentile=95) for comb in binary_combinations}
    dice_dict_m = {comb: DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False) for comb in binary_combinations}
    dice_dict_c_f_m_g = {comb: DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False) for comb in binary_combinations}
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    
    epoched=0
    epoch_len = train_size  // _bs
    best_diff_ET, best_diff_TC, best_diff_WT = 1,1,1

    for epoch in tqdm(range(epoched,epochs)):
        rec_model.train()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        epoch_mse_loss = 0
        epoch_dice_loss = 0

        step = 0
        # training
        for batch in tqdm(train_loader):
            step += 1
            optimizer.zero_grad()
            complete_modality_image= batch[img_index].to(device) # ([1, 4, 128, 128, 128])  
            label = batch[label_index].to(device)
            channels_to_drop= utils.random_scenario()
            remaining_channels = [x for x in [0,1,2,3] if x not in channels_to_drop]

            missing_modality_image = drop_modality_image_channel(complete_modality_image,
                                        method="modality_mean",
                                        idx_to_drop = channels_to_drop,
                                        remaining_modalities = remaining_channels )

            complete_modality_features = extract_complete_modality_features(rec_model.swinunetr,
                                                                        complete_modality_image,
                                                                        Training_config.recon_level,
                                                                        rec_model.diff_on)
            missing_modality_features = extract_complete_modality_features(rec_model.swinunetr,
                                                                        missing_modality_image,
                                                                        Training_config.recon_level,
                                                                        rec_model.diff_on)

            missing_modality_features = drop_modality_feature_channel(missing_modality_features,
                                                                      Training_config.recon_level,
                                                                      method="whole_mean",
                                                                      idx_to_drop = channels_to_drop,
                                                                      fs = Training_config.feature_size,
                                                                      pc_mean_features = rec_model.mean_features)
            
            recon_complete_modality = rec_model(missing_modality_features) 

            mse_loss = criterionMSE(recon_complete_modality, complete_modality_features)

            
            with torch.no_grad(): 
                c_f_m_generated_logits = rec_model.swinunetr(missing_modality_image, bottleneck=[recon_complete_modality,channels_to_drop])
            dice_loss = criterionDiceCE(c_f_m_generated_logits, label) 
            
            total_loss = 0.2 * mse_loss + 0.8 * dice_loss  
            total_loss.backward()
            optimizer.step()
            
            #print(f"MSE loss: {mse_loss.item():.4f}")
            #print(f"Dice loss: {dice_loss.item():.4f}")

            epoch_mse_loss += mse_loss.item()
            epoch_dice_loss += dice_loss.item()

        epoch_mse_loss /= step
        epoch_dice_loss /= step

        epoch_total_loss = epoch_mse_loss + epoch_dice_loss 
        writer.add_scalar("Training/EpochTotalLoss", epoch_total_loss, epoch)
        writer.add_scalar("Training/EpochMSELoss", epoch_mse_loss, epoch)
        writer.add_scalar("Training/EpochDiceLoss", epoch_dice_loss, epoch)
        
        if Training_config.lr_scheduler:
            scheduler.step()
            writer.add_scalar("Training/EpochLR", optimizer.param_groups[0]['lr'], epoch) 

        if not epoch == 0 and epoch % 5 == 0:            
            model_save_name = model_save_path + Training_config.experiment_name + "_Epoch_" + str(epoch) + ".pth"
            opt_save_name=model_save_path + Training_config.experiment_name + "_checkpoint_Epoch_" + str(epoch) + ".pt"
            torch.save(rec_model.state_dict(), model_save_name)
            torch.save({
            'epoch': epoch,
            'model_state_dict': rec_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, opt_save_name)
            print("Saved Model")

        # evaluation
        if not epoch == 0 and epoch % Training_config.val_interval == 0: # 
            rec_model.eval()
            rec_model.ldm_eval = True
            print("validation starts")
            with torch.no_grad():
                metric_c = {} 
                metric_m = {}
                metric_c_f_m_g = {}                
                metric_diff = {}
                metric_hd_c_f_m_g = {}
                metric_c_hd = {}

                dict_to_save = {}
                dict_to_save_hd = {}

                for val_data in tqdm(val_loader):
                    val_labels = val_data[label_index].to(device) 
                    complete_modality_image = val_data[img_index].to(device) # [1, 4, 240, 240, 155]
                    complete_modality_image = complete_modality_image[0].unsqueeze(0)
                    # evaluation of complete modality image
                    c_logits = sliding_window_inference(complete_modality_image, roi_size=(128,128,128), sw_batch_size=1, predictor= wrapper.rec_model.swinunetr)
                    c_outputs = [post_trans(i) for i in decollate_batch(c_logits)]
                    dice_metric_c(y_pred=c_outputs, y=val_labels)
                    hd_metric_c(y_pred=c_outputs, y=val_labels)

                    # evaluation of missing modality scenarios
                    for r in range(1, 4):       
                        for channels in itertools.combinations(range(4), r):  # Generate all valid subsets
                            
                            remaining_channels = [x for x in [0,1,2,3] if x not in channels]
                            missing_modality_image = drop_modality_image_channel(complete_modality_image,
                                                                                method="modality_mean",
                                                                                idx_to_drop = channels,
                                                                                remaining_modalities = remaining_channels )
                            wrapper.channels_to_drop=channels
                            _logits = sliding_window_inference(
                                inputs=missing_modality_image,
                                roi_size=(128,128,128),
                                sw_batch_size=1,
                                predictor=wrapper)
                            
                            m_logits = _logits[0]
                            c_f_m_generated_logits = _logits[1]
                            m_outputs = [post_trans(i) for i in decollate_batch(m_logits)]
                            c_f_m_generated_outputs = [post_trans(i) for i in decollate_batch(c_f_m_generated_logits)]

                            _key = utils.get_dict_key(remaining_channels)
                            dice_dict_m["".join(_key)](y_pred=m_outputs, y=val_labels)
                            dice_dict_c_f_m_g["".join(_key)](y_pred=c_f_m_generated_outputs, y=val_labels)
                            hd_dict_c_f_m_g["".join(_key)](y_pred=c_f_m_generated_outputs, y=val_labels)

                for sc_key in dice_dict_m:
                    metric_m["dice_"+sc_key] = dice_dict_m[sc_key].aggregate()
                    metric_c_f_m_g["dice_"+sc_key] = dice_dict_c_f_m_g[sc_key].aggregate()
                    metric_hd_c_f_m_g["hd_"+sc_key] = hd_dict_c_f_m_g[sc_key].aggregate()

                    metric_diff[sc_key+"_CGfM-M_ET"] = dice_dict_c_f_m_g[sc_key].aggregate()[2].item() - dice_dict_m[sc_key].aggregate()[2].item()
                    metric_diff[sc_key+"_CGfM-M_TC"] = dice_dict_c_f_m_g[sc_key].aggregate()[0].item() - dice_dict_m[sc_key].aggregate()[0].item()
                    metric_diff[sc_key+"_CGfM-M_WT"] = dice_dict_c_f_m_g[sc_key].aggregate()[1].item() - dice_dict_m[sc_key].aggregate()[1].item()

                    writer.add_scalar(f"Val/{sc_key}/Dice_M_ET", metric_m["dice_"+sc_key][2].item(), epoch)
                    writer.add_scalar(f"Val/{sc_key}/Dice_M_TC", metric_m["dice_"+sc_key][0].item(), epoch)
                    writer.add_scalar(f"Val/{sc_key}/Dice_M_WT", metric_m["dice_"+sc_key][1].item(), epoch)
                    writer.add_scalar(f"Val/{sc_key}/Dice_CGfM_ET", metric_c_f_m_g["dice_"+sc_key][2].item(), epoch)
                    writer.add_scalar(f"Val/{sc_key}/Dice_CGfM_TC", metric_c_f_m_g["dice_"+sc_key][0].item(), epoch)
                    writer.add_scalar(f"Val/{sc_key}/Dice_CGfM_WT", metric_c_f_m_g["dice_"+sc_key][1].item(), epoch)

                    dict_to_save[sc_key] = {"Dice_CGfM_ET":metric_c_f_m_g["dice_"+sc_key][2].item(),
                                            "Dice_CGfM_TC":metric_c_f_m_g["dice_"+sc_key][0].item(),
                                            "Dice_CGfM_WT":metric_c_f_m_g["dice_"+sc_key][1].item() }
                    
                    writer.add_scalar(f"Val/{sc_key}/Dice_CGfM-M_ET", metric_diff[sc_key+"_CGfM-M_ET"], epoch)
                    writer.add_scalar(f"Val/{sc_key}/Dice_CGfM-M_TC", metric_diff[sc_key+"_CGfM-M_TC"], epoch)
                    writer.add_scalar(f"Val/{sc_key}/Dice_CGfM-M_WT", metric_diff[sc_key+"_CGfM-M_WT"], epoch)

                    writer.add_scalar(f"Val/{sc_key}/HD_CGfM_ET", metric_hd_c_f_m_g["hd_"+sc_key][2].item(), epoch)
                    writer.add_scalar(f"Val/{sc_key}/HD_CGfM_TC", metric_hd_c_f_m_g["hd_"+sc_key][0].item(), epoch)
                    writer.add_scalar(f"Val/{sc_key}/HD_CGfM_WT", metric_hd_c_f_m_g["hd_"+sc_key][1].item(), epoch)

                    dict_to_save_hd[sc_key] = {"HD_CGfM_ET":metric_hd_c_f_m_g["hd_"+sc_key][2].item(),
                                            "HD_CGfM_TC":metric_hd_c_f_m_g["hd_"+sc_key][0].item(),
                                            "HD_CGfM_WT":metric_hd_c_f_m_g["hd_"+sc_key][1].item() }
                    
                for sc_key in dice_dict_m:
                    dice_dict_m[sc_key].reset()
                for sc_key in dice_dict_c_f_m_g:
                    dice_dict_c_f_m_g[sc_key].reset()
                for sc_key in hd_dict_c_f_m_g:
                    hd_dict_c_f_m_g[sc_key].reset()

                metric_c["dice_metric_c"] = dice_metric_c.aggregate()
                metric_c_hd["hd_metric_c"] = hd_metric_c.aggregate()

                writer.add_scalar(f"Val/Dice_C_ET", metric_c["dice_metric_c"][2].item(), epoch)
                writer.add_scalar(f"Val/Dice_C_TC", metric_c["dice_metric_c"][0].item(), epoch)
                writer.add_scalar(f"Val/Dice_C_WT", metric_c["dice_metric_c"][1].item(), epoch)

                dict_to_save["complete"]= {"Dice_CGfM_ET":metric_c["dice_metric_c"][2].item(),
                                           "Dice_CGfM_TC":metric_c["dice_metric_c"][0].item(),
                                           "Dice_CGfM_WT":metric_c["dice_metric_c"][1].item()}

                values = [inner_dict["Dice_CGfM_ET"] for inner_dict in dict_to_save.values()]
                avg_ET = sum(values) / len(values)
                values = [inner_dict["Dice_CGfM_TC"] for inner_dict in dict_to_save.values()]
                avg_TC = sum(values) / len(values)
                values = [inner_dict["Dice_CGfM_WT"] for inner_dict in dict_to_save.values()]
                avg_WT = sum(values) / len(values)
                
                writer.add_scalar(f"Val/Avg/Dice_CGfM_ET", avg_ET, epoch)
                writer.add_scalar(f"Val/Avg/Dice_CGfM_TC", avg_TC, epoch)
                writer.add_scalar(f"Val/Avg/Dice_CGfM_WT", avg_WT, epoch)

                dict_to_save_hd["complete"]= {"HD_CGfM_ET":metric_c_hd["hd_metric_c"][2].item(),
                                              "HD_CGfM_TC":metric_c_hd["hd_metric_c"][0].item(),
                                              "HD_CGfM_WT":metric_c_hd["hd_metric_c"][1].item()}

                values = [inner_dict["HD_CGfM_ET"] for inner_dict in dict_to_save_hd.values()]
                avg_ET_hd = sum(values) / len(values)
                values = [inner_dict["HD_CGfM_TC"] for inner_dict in dict_to_save_hd.values()]
                avg_TC_hd  = sum(values) / len(values)
                values = [inner_dict["HD_CGfM_WT"] for inner_dict in dict_to_save_hd.values()]
                avg_WT_hd  = sum(values) / len(values)


                dict_to_save["average"]= {"Dice_ET":avg_ET,
                                           "Dice_TC":avg_TC,
                                           "Dice_WT":avg_WT}
                
                dict_to_save_hd["average"]= {"Dice_ET":avg_ET_hd,
                                           "Dice_TC":avg_TC_hd,
                                           "Dice_WT":avg_WT_hd}
                dice_metric_c.reset()
                hd_metric_c.reset()

                json_dump_path = os.path.dirname(model_save_path) + f"/{str(epoch)}_dice_results.json"
                with open(json_dump_path, "w") as json_file:
                    json.dump(dict_to_save, json_file, indent=4)
                json_dump_path = os.path.dirname(model_save_path) + f"/{str(epoch)}_hd_results.json"
                with open(json_dump_path, "w") as json_file:
                    json.dump(dict_to_save_hd, json_file, indent=4)
    
    writer.close() 
    print(Training_config.experiment_name)                 