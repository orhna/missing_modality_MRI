class Training_config:
        
        experiment_name = "brats23_mm12_sd_ds_dicece_rd_5e-4"#"joint_hs3_hs4_5e-4_continued_1e-5"#"joint_hs3_hs4_5e-4"#"level5_whole_mean_1e-2_m"#"mm12_sd_ds_dicece_rd_evalall_updated"#"finetune_separate__wmeanfeaturedrop_auxdice_mm12_sd_ds_dicece_rd_E490"#"mm24_sd_ds_dicece_rd_evalall_seed55"#"mm12_sd_ds_dicece_rd_evalall"#"combined_192_unet256_sch5e-4_st50_g08_weight08"#"mm12_sd_ds_dicece_rd_EXPdecup"#"2d_192_64_d2_predx0_gnoise_cosine"#"mm12_sd_ds_dicece_rd"#"mm12_sd_ds_dicece"
        dataset_to_train = ["BRATS21"]
        modalities_to_train = ["FLAIR", "T1c", "T1", "T2"] # change this order to match the model order if fine-tuning
        epoch = 500
        seed = 55
        train_batch_size = 1
        val_interval = 20 # the number of epochs between the validation  
        lr = 5e-4 #5e-5
        random_modality_drop = True # always on for ldm training
        model_type = "MultiModal_SwinUNETR" # default model type unet {"MultiModal_SwinUNETR","SwinUNETR", "nnunet"}
        feature_size = 12
        drop_mode = "modality_mean" # {"whole_mean","modality_mean","zero"} # applies on image channels
        generation_mode = "whole_mean" # {"no_replacement","gnoise","whole_mean","zero", "pcmean_features"} # applies on feature channels
        deep_supervision = True
        sep_dec = True # adds decoders for all feature seperately, loss on all tasks
        cross_attention = False
        lr_scheduler = True
        sch_step_size = 50
        sch_gamma = 0.8
        weight_decay = 1e-4
        diff_on = "separate" # {"combined","separate"}
        recon_level = "none" # {"hs3", "hs4", "hs3_hs4", "none"}
        use_memory = False 
        #recon_level = [5] # {3,4,5} # any upper levels don't fit to memory
        # checkpoint
        continue_training = False
        checkpoint_path = "/data/hjlee/orhun/thesis/models/joint_hs3_hs4_5e-4/joint_hs3_hs4_5e-4_checkpoint_Epoch_400.pt"
        path21 = "/data/hjlee/orhun/pt_model/fold1_f48_ep300_4gpu_dice0_9059/model.pt"
        pt_mmformer_path = "/mnt/disk1/hjlee/orhun/repo/mmFormer/pt_model/model_last.pth"
        # model_save_path 
        model_save_path = "/data/hjlee/orhun/thesis/models/"
        load_model_path= "/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd_evalall_updated/mm12_sd_ds_dicece_rd_evalall_updated_checkpoint_Epoch_460.pt"#"/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd_evalall/mm12_sd_ds_dicece_rd_evalall_checkpoint_Epoch_490.pt"#"/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd/mm12_sd_ds_dicece_rd_checkpoint_Epoch_170.pt"#
        # "/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd_EXPtpconv/mm12_sd_ds_dicece_rd_EXPtpconv_BEST_ET.pth"
        # best with RD "/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd/mm12_sd_ds_dicece_rd_checkpoint_Epoch_170.pt"
        cropped_input_size = [128,128,128]   
        output_channel = 3
        workers = 2
        # not in use
        t1c_spec = False # adds a decoder for t1c features, loss on only TC
        aux = False
        recon = False
        tp_conv = False 
        dec_upsample = False
        weighted_dice = False
        new_eval = True
        # ldm configuration
        diff_domain = "3D" # {"2D", "3D"}
        ldm_dim = 64 #initial layer dimension
        ldm_dim_mults= (1, 2)
        timesteps= 500

class Database_config:
        channels={}
        # modalities for each database, the order in the pre-processed files from 0-3
        channels['BRATS18'] = ["FLAIR", "T1", "T1c", "T2"]
        channels['BRATS21'] = ["T1c", "T1", "FLAIR", "T2"]
        channels['BRATS1617'] = ["FLAIR", "T1", "T1c", "T2"]
        #size for each database
        #training set size
        train_size={}
        train_size['BRATS18'] = 200 #2
        train_size['BRATS21'] = 1051 #2
        train_size['BRATS1617'] = 429 #7

        val_size={}
        val_size["BRATS18"]= 85
        val_size["BRATS21"]= 200
        val_size["BRATS1617"]= 40 #3

        total_size={}
        total_size['BRATS18'] =  285 #8,
        total_size['BRATS21'] =  1251 #8
        total_size['BRATS1617'] = 469 #10
        
        img_path={}
        seg_path={}
        split_path={}
        split_path["BRATS18"]={}
        split_path["BRATS21"]={}
        
        img_path["BRATS18"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed_C/Training/Images" # "data/BRATS/Images"
        seg_path["BRATS18"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed_C/Training/Labels"
        split_path["BRATS18"]["train"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/train_val.txt" # train_overfit, train_val,
        split_path["BRATS18"]["val"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/val_test.txt" # train_overfit, val_test
        split_path["BRATS18"]["test"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/test.txt"

        img_path["BRATS21"] = "/data/hjlee/orhun/data/BRATS21_Processed/Training/Images" # "data/BRATS/Images"
        seg_path["BRATS21"] = "/data/hjlee/orhun/data/BRATS21_Processed/Training/Labels"
        split_path["BRATS21"]["train"] = "/data/hjlee/orhun/data/BRATS21_Processed/train.txt"
        split_path["BRATS21"]["val"] = "/data/hjlee/orhun/data/BRATS21_Processed/test.txt"
        split_path["BRATS21"]["test"] = "/data/hjlee/orhun/data/BRATS21_Processed/test.txt"

        img_path["BRATS1617"] = "/data/hjlee/orhun/data/BRATS_Preprocessed/Images" # "data/BRATS/Images"
        seg_path["BRATS1617"] = "/data/hjlee/orhun/data/BRATS_Preprocessed/MergedLabels"

#/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed_C/Images/Brats18_2013_0_1.nii.gz

class Test_config:
        dataset = "BRATS18"
        modalities = ["FLAIR", "T1c", "T1", "T2"] # order for 21: flair, t1c,t1,t2 
                                                  # order for 18: "FLAIR", "T1", "T1c", "T2"
        model= "MultiModal_SwinUNETR" # {"MultiModal_SwinUNETR","SwinUNETR"}
        crop_size = 128
        n_of_output_c = 3
        deep_supervision = True
        sep_dec = True
        dec_upsample = False
        cross_attention = False
        t1c_spec = False
        feature_size = 12
        model_file_path = "/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd_evalall/mm12_sd_ds_dicece_rd_evalall_BEST_ET.pth"
        random_dropping = "modality_mean" # {"whole_mean","modality_mean", "pc_modality_mean", "zero"} 
        pc_mean_path = "/mnt/disk1/hjlee/orhun/repo/thesis/pc_mean.pt"


# {'dice': 0.5148282051086426, 'sensitivity': 0.15686260163784027, 'precision': -0.5354330539703369, 'IOU': 0.6259039640426636, 'dice_TC': 0.6833519339561462, 'dice_WT': 0.8236450552940369, 'dice_ET': 0.6675910353660583}

# with separate encoders, 36 feature size, no pos enc
# {'dice': 0.5314303636550903, 'sensitivity': 0.15145213901996613, 'precision': -0.5108014941215515, 'IOU': 0.6438222527503967, 'dice_TC': 0.7043986320495605, 'dice_WT': 0.8408573865890503, 'dice_ET': 0.7123273611068726}

# with separate encoders, 36 feature size, pos enc #/mnt/disk1/hjlee/orhun/repo/thesis/models/multimodal_swinunetr_36_pos_sch2e-4/multimodal_swinunetr_36_pos_sch2e-4_Epoch_249.pth
# {'dice': 0.5833877325057983, 'sensitivity': 0.1492043435573578, 'precision': -0.49647679924964905, 'IOU': 0.6676676273345947, 'dice_TC': 0.762221097946167, 'dice_WT': 0.8575522899627686, 'dice_ET': 0.7322155833244324}

# with separate encoders, 36 feature size, pos enc, weighted dice  [0.1,0.4,0.1,0.4] 
# /mnt/disk1/hjlee/orhun/repo/thesis/models/multimodal_swinunetr_36_pos_sch2e-4_wloss_continued/multimodal_swinunetr_36_pos_sch2e-4_wloss_continued_BEST_WT.pth
# {'dice': 0.5854236483573914, 'sensitivity': 0.16831083595752716, 'precision': -0.6542367935180664, 'IOU': 0.6644439101219177, 'dice_TC': 0.7627357840538025, 'dice_WT': 0.8490474820137024, 'dice_ET': 0.7351805567741394}
# /mnt/disk1/hjlee/orhun/repo/thesis/models/multimodal_swinunetr_36_pos_sch2e-4_wloss_continued/multimodal_swinunetr_36_pos_sch2e-4_wloss_continued_Epoch_200.pth
# {'dice': 0.5783093571662903, 'sensitivity': 0.15719640254974365, 'precision': -0.5519970059394836, 'IOU': 0.6656265258789062, 'dice_TC': 0.7506475448608398, 'dice_WT': 0.8504215478897095, 'dice_ET': 0.7390080690383911}
# /mnt/disk1/hjlee/orhun/repo/thesis/models/multimodal_swinunetr_36_pos_sch2e-4_wloss_continued/multimodal_swinunetr_36_pos_sch2e-4_wloss_continued_BEST_ET.pth
 
# with separate encoders, 48 feature size, no pos enc 
# #/mnt/disk1/hjlee/orhun/repo/thesis/models/multimodal_swinunetr_shard_48_sch2e-4_continued/multimodal_swinunetr_shard_48_sch2e-4_continued_Epoch_299.pth
# {'dice': 0.5925178527832031, 'sensitivity': 0.15195824205875397, 'precision': -0.5289883017539978, 'IOU': 0.6796277165412903, 'dice_TC': 0.7781355977058411, 'dice_WT': 0.8795819282531738, 'dice_ET': 0.7493073344230652}
# /mnt/disk1/hjlee/orhun/repo/thesis/models/multimodal_swinunetr_shard_48_sch2e-4_continued/multimodal_swinunetr_shard_48_sch2e-4_continued_BEST_BRATS.pth

# with separate encoders, 48 feature size, pos enc
# 

# finetuning swinunetr:
# /data/hjlee/orhun/repo/thesis/models/swinunetr_finetune_noaugs_nonzeroT/swinunetr_finetune_noaugs_nonzeroT_BEST_TC.pth
# 'dice_ET': 0.7512772679328918, 'dice_TC': 0.790663480758667, 'dice_WT': 0.8788363933563232}

#/data/hjlee/orhun/repo/thesis/models/swinunetr_finetune_noaugs_nonzeroT/swinunetr_finetune_noaugs_nonzeroT_Epoch_60.pth
# 'dice_ET': 0.7479267120361328, 'dice_TC': 0.8134040832519531, 'dice_WT': 0.8922939896583557}





# mmformer 
# {'dice_ET': 0.7644659876823425, 'dice_TC': 0.8710879683494568, 'dice_WT': 0.9036794304847717}

# without cropping:
# 'dice_ET': 0.7260687947273254, 'dice_TC': 0.7491841912269592, 'dice_WT': 0.8643548488616943


# training from scratch final models to choose
# /data/hjlee/orhun/thesis/models/mm36_cross_dicece_sch3e-4_discrete04/mm36_cross_dicece_sch3e-4_discrete04_checkpoint_Epoch_50.pt
# 'dice_ET': 0.7341363430023193, 'dice_TC': 0.7908458113670349, 'dice_WT': 0.8659378290176392}

# /data/hjlee/orhun/thesis/models/mm36_dicece_sch3e-4_discrete04/mm36_dicece_sch3e-4_discrete04_checkpoint_Epoch_50.pt
# 'dice_ET': 0.7414209842681885, 'dice_TC': 0.7750723361968994, 'dice_WT': 0.8709604144096375}


#/data/hjlee/orhun/thesis/models/mm36_cross_deeps_dicece_sch3e-4_/mm36_cross_deeps_dicece_sch3e-4__checkpoint_Epoch_100.pt
#'dice_ET': 0.7587454915046692, 'dice_TC': 0.8100786209106445, 'dice_WT': 0.88616943359375}

# /data/hjlee/orhun/thesis/models/mm24_cross_ds_dicece_sch3e-4/mm24_cross_ds_dicece_sch3e-4_checkpoint_Epoch_200.pt
# 'dice_ET': 0.7553126811981201, 'dice_TC': 0.8033475279808044, 'dice_WT': 0.8882024884223938}

# /data/hjlee/orhun/thesis/models/mm12_cross_ds_sepdec_dicece_sch3e-4/mm12_cross_ds_sepdec_dicece_sch3e-4_checkpoint_Epoch_150.pt
# 'dice_ET': 0.766567587852478, 'dice_TC': 0.8047633767127991, 'dice_WT': 0.8969186544418335}


# /data/hjlee/orhun/thesis/models/mm24_cross_ds_sepdec_dicece_sch3e-4/mm24_cross_ds_sepdec_dicece_sch3e-4_checkpoint_Epoch_90.pt
# 'dice_ET': 0.7613540887832642, 'dice_TC': 0.8164069652557373, 'dice_WT': 0.8922691345214844}


# mm24_sd_ds_dicece epoch 149
# ET:77.27    TC:83.17    WT:90.22

# mm24_sd_ds_dicece epoch 195
# ET:76.90    TC:83.13    WT:90.00



# "/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd_evalall_updated/mm12_sd_ds_dicece_rd_evalall_updated_checkpoint_Epoch_460.pt
# AVG: ET:60.15   TC:74.15    WT:85.28