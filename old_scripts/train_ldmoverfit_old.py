import torch
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
import utils.utils as utils
import config
import argparse
import itertools
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from monai.metrics import DiceMetric
from nets.mm_3Dldm import mm_3Dldm, mm_3DLDMWrapper, drop_modality_image_channel
from nets.mm_finalldm import mm_finalldm
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from monai.losses.dice import  DiceCELoss

if __name__ == "__main__":

    #command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=str)
    args = parser.parse_args()
    device_list = list(map(int, args.device_id.split(',')))   
    
    device = utils.initialize_GPU(device_list[0])
    device_id_list = [device]
        
    Training_config = config.Training_config()
    Database_config = config.Database_config()
    _date = datetime.now().strftime("%d-%H-%M")
    log_dir = f"./logs/ldm_final/{Training_config.experiment_name}_{_date}"
    writer = SummaryWriter(log_dir=log_dir)

    #save config
    utils.save_config_from_py("config.py",log_dir)
    #set seed
    utils.set_random_seed(Training_config.seed)

    cropped_input_size = Training_config.cropped_input_size
    epochs = Training_config.epoch
    img_index, label_index = 0, 1

    chosen_ds = Training_config.dataset_to_train[0]
    channels= Training_config.modalities_to_train
    train_size=Database_config.train_size[chosen_ds]
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

    # overfit override
    Database_config.split_path[chosen_ds]["train"] ="/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/train_overfit.txt"
    Database_config.split_path[chosen_ds]["val"] ="/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/val_overfit.txt"

    train_loader, val_loader = utils.get_ldm_loaders(images=images,
                                            segs=segs,
                                            train_file=Database_config.split_path[chosen_ds]["train"],
                                            val_file=Database_config.split_path[chosen_ds]["val"] ,
                                            workers=Training_config.workers,
                                            train_batch_size=_bs,
                                            cropped_input_size=cropped_input_size,
                                            channel_indices=channel_indices)

    train_size = len(train_loader.dataset)
    print("size of train",len(train_loader.dataset))
    print("size of val",len(val_loader.dataset))

    ldm_model = mm_finalldm(Training_config, device_list=device_id_list)
    ldm_model.load_swinunetr_weights(load_model_path)
    #wrapper = mm_3DLDMWrapper(ldm_model)
    #wrapper.ldm_model.swinunetr.ldm_eval= True

    optimizer = torch.optim.Adam(ldm_model.parameters(), lr=Training_config.lr)
    scheduler = StepLR(optimizer, step_size=Training_config.sch_step_size, gamma=Training_config.sch_gamma)

    dice_metric_m = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    dice_metric_c = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    dice_metric_c_f_m_g = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])

    epoched=0
    epoch_len = train_size  // _bs
    best_diff_ET, best_diff_TC, best_diff_WT = 1,1,1

    for epoch in tqdm(range(epoched,epochs)):
        
        ldm_model.train()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        epoch_diff_loss = 0
        step = 0

        # training
        for batch in tqdm(train_loader):
            
            step += 1
            optimizer.zero_grad()
            complete_modality = batch[img_index].to(device) # ([1, 4, 128, 128, 128])  
            label = batch[label_index].to(device)

            diff_loss = ldm_model(complete_modality)

            total_loss = diff_loss 
            
            total_loss.backward()
            optimizer.step()
            
            print(f"Diffusion loss: {diff_loss.item():.4f}")
            epoch_diff_loss += diff_loss.item()

        epoch_diff_loss /= step

        epoch_total_loss = epoch_diff_loss 
        writer.add_scalar("Training/EpochTotalLoss", epoch_total_loss, epoch)
        writer.add_scalar("Training/EpochDiffLoss", epoch_diff_loss, epoch)

        if Training_config.lr_scheduler:
            scheduler.step()
            writer.add_scalar("Training/EpochLR", optimizer.param_groups[0]['lr'], epoch) 

        # evaluation
        if not epoch == 0 and epoch % Training_config.val_interval == 0: # 
            ldm_model.eval()
            print("validation starts")
            with torch.no_grad():
                metric={}
                for val_data in val_loader:

                    val_labels = val_data[label_index].to(device) 
                    complete_modality_image = val_data[img_index].to(device) # [1, 4, 240, 240, 155]
                    complete_modality_image = complete_modality_image[0].unsqueeze(0)
                    # evaluation of complete modality image
                    c_logits = sliding_window_inference(complete_modality_image, roi_size=(128,128,128), sw_batch_size=1, predictor= wrapper.ldm_model.swinunetr)
                    c_outputs = [post_trans(i) for i in decollate_batch(c_logits)]
                    dice_metric_c(y_pred=c_outputs, y=val_labels)

                    # evaluation of missing modality scenarios
                    channels_to_drop= [0]
                    remaining_channels = [x for x in [0,1,2,3] if x not in channels_to_drop]
                    #print(channels_to_drop)
                    #print(remaining_channels)

                    missing_modality_image = drop_modality_image_channel(complete_modality_image,
                                                                    Training_config.drop_mode,
                                                                    channels_to_drop,
                                                                    remaining_channels,
                                                                    device_list)
                    
                    #print(missing_modality_image.shape)
                    cat_m_c_image = torch.cat([missing_modality_image,complete_modality_image], dim=1)

                    _logits = sliding_window_inference(
                                inputs=cat_m_c_image,
                                roi_size=(128,128,128),
                                sw_batch_size=1,
                                predictor=wrapper)
                    
                    m_logits = _logits[0]
                    c_f_m_generated_logits = _logits[1]
                    #m_w_c_generated_logits = _logits[2]
                    m_outputs = [post_trans(i) for i in decollate_batch(m_logits)]
                    c_f_r_generated_outputs = [post_trans(i) for i in decollate_batch(c_f_m_generated_logits)]
                    #m_w_c_generated_outputs = [post_trans(i) for i in decollate_batch(m_w_c_generated_logits)]

                    dice_metric_m(y_pred=m_outputs, y=val_labels)
                    dice_metric_c_f_m_g(y_pred=c_f_r_generated_outputs, y=val_labels)
                    #dice_metric_m_w_c_g(y_pred=m_w_c_generated_outputs, y=val_labels)

                    # evaluation of missing modality images
                    """
                    
                    for r in range(1, 4):       
                        for channels in itertools.combinations(range(4), r):  # Generate all valid subsets
                            missing_modality_image = complete_modality_image.clone()
                            wrapper.channels=channels
                            for ch in channels:
                                missing_modality_image[:, ch] = random_mod_channel  # Set selected channels

                            #print(missing_modality_image.shape)
                            _logits = sliding_window_inference(
                                inputs=missing_modality_image,
                                roi_size=(128,128,128),
                                sw_batch_size=1,
                                predictor=wrapper)
                            
                            m_logits = _logits[0]
                            c_f_m_generated_logits = _logits[1]
                            
                            m_outputs = [post_trans(i) for i in decollate_batch(m_logits)]
                            c_f_r_generated_outputs = [post_trans(i) for i in decollate_batch(c_f_m_generated_logits)]
                            
                            dice_metric_m(y_pred=m_outputs, y=val_labels)
                            dice_metric_c_f_m_g(y_pred=c_f_r_generated_outputs, y=val_labels)
                      """

                metric["dice_metric_m"] = dice_metric_m.aggregate()
                metric["dice_metric_c"] = dice_metric_c.aggregate()
                metric["dice_metric_c_f_m_g"] = dice_metric_c_f_m_g.aggregate()
                #metric["dice_metric_m_w_c_g"] = dice_metric_m_w_c_g.aggregate()

                metric["dice_CGfM-M_ET"] = dice_metric_c_f_m_g.aggregate()[2].item() - dice_metric_m.aggregate()[2].item()
                metric["dice_CGfM-M_TC"] = dice_metric_c_f_m_g.aggregate()[0].item() - dice_metric_m.aggregate()[0].item()
                metric["dice_CGfM-M_WT"] = dice_metric_c_f_m_g.aggregate()[1].item() - dice_metric_m.aggregate()[1].item()

                metric["dice_C-CGfM_ET"] = dice_metric_c.aggregate()[2].item() - dice_metric_c_f_m_g.aggregate()[2].item() 
                metric["dice_C-CGfM_TC"] = dice_metric_c.aggregate()[0].item() - dice_metric_c_f_m_g.aggregate()[0].item() 
                metric["dice_C-CGfM_WT"] = dice_metric_c.aggregate()[1].item() - dice_metric_c_f_m_g.aggregate()[1].item() 

                writer.add_scalar(f"Validation/{chosen_ds}/Dice_C_ET", metric["dice_metric_c"][2].item(), epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/Dice_C_TC", metric["dice_metric_c"][0].item(), epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/Dice_C_WT", metric["dice_metric_c"][1].item(), epoch)

                writer.add_scalar(f"Validation/{chosen_ds}/Dice_M_ET", metric["dice_metric_m"][2].item(), epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/Dice_M_TC", metric["dice_metric_m"][0].item(), epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/Dice_M_WT", metric["dice_metric_m"][1].item(), epoch)

                writer.add_scalar(f"Validation/{chosen_ds}/Dice_CGfM_ET", metric["dice_metric_c_f_m_g"][2].item(), epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/Dice_CGfM_TC", metric["dice_metric_c_f_m_g"][0].item(), epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/Dice_CGfM_WT", metric["dice_metric_c_f_m_g"][1].item(), epoch)

                #writer.add_scalar(f"Validation/{chosen_ds}/Dice_MwCG_ET", metric["dice_metric_m_w_c_g"][2].item(), epoch)
                #writer.add_scalar(f"Validation/{chosen_ds}/Dice_MwCG_TC", metric["dice_metric_m_w_c_g"][0].item(), epoch)
                #writer.add_scalar(f"Validation/{chosen_ds}/Dice_MwCG_WT", metric["dice_metric_m_w_c_g"][1].item(), epoch)
                
                writer.add_scalar(f"Validation/{chosen_ds}/dice_diff_CGfM-M_ET", metric["dice_CGfM-M_ET"], epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/dice_diff_CGfM-M_TC", metric["dice_CGfM-M_TC"], epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/dice_diff_CGfM-M_WT", metric["dice_CGfM-M_WT"], epoch)

                writer.add_scalar(f"Validation/{chosen_ds}/dice_diff_C-CGfM_ET", metric["dice_C-CGfM_ET"], epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/dice_diff_C-CGfM_TC", metric["dice_C-CGfM_TC"], epoch)
                writer.add_scalar(f"Validation/{chosen_ds}/dice_diff_C-CGfM_WT", metric["dice_C-CGfM_WT"], epoch)


                dice_metric_m.reset()
                dice_metric_c.reset()
                dice_metric_c_f_m_g.reset()

                if metric["dice_CGfM-M_ET"] < best_diff_ET:
                    best_diff_ET = metric["dice_CGfM-M_ET"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_ET_diff.pth"
                        torch.save(ldm_model.state_dict(), model_save_name)                   
                if metric["dice_CGfM-M_TC"] < best_diff_TC:
                    best_diff_TC = metric["dice_CGfM-M_TC"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_TC_diff.pth"
                        torch.save(ldm_model.state_dict(), model_save_name) 
                if metric["dice_CGfM-M_WT"] < best_diff_WT:
                    best_diff_WT = metric["dice_CGfM-M_WT"]
                    best_metric_epoch = epoch + 1
                    if epoch>1:
                        model_save_name = model_save_path + Training_config.experiment_name + "_BEST_WT_diff.pth"
                        torch.save(ldm_model.state_dict(), model_save_name)

    writer.close() 
    print(Training_config.experiment_name)                 