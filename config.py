class Training_config:
        
        experiment_name = "test"
        dataset_to_train = ["BRATS18"]
        modalities_to_train = ["FLAIR", "T1c", "T1", "T2"] 
        epoch = 500
        seed = 55
        train_batch_size = 1
        val_interval = 20 # the number of epochs between the validation  
        lr = 5e-4
        random_modality_drop = True 
        feature_size = 12
        drop_mode = "modality_mean" # {"whole_mean","modality_mean","zero"} # applies on image channels
        lr_scheduler = True
        sch_step_size = 50
        sch_gamma = 0.8
        weight_decay = 1e-4
        recon_level = "hs3_hs4" # {"hs3", "hs4", "hs3_hs4", "none"}
        use_memory = False 
        continue_training = True
        checkpoint_path = "/data/hjlee/orhun/thesis/models/joint_hs3_hs4_5e-4/joint_hs3_hs4_5e-4_checkpoint_Epoch_400.pt"
        # model_save_path 
        model_save_path = "/data/hjlee/orhun/thesis/models/"
        cropped_input_size = [128,128,128]   
        output_channel = 3
        workers = 2

class Database_config:
        channels={}
        # modalities for each database, the order in the pre-processed files from 0-3
        channels['BRATS18'] = ["FLAIR", "T1", "T1c", "T2"]
        channels['BRATS23'] = ["T1c", "T1", "FLAIR", "T2"]

        img_path={}
        seg_path={}
        split_path={}
        split_path["BRATS18"]={}
        split_path["BRATS23"]={}
        
        img_path["BRATS18"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed_C/Training/Images" # "data/BRATS/Images"
        seg_path["BRATS18"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed_C/Training/Labels"
        split_path["BRATS18"]["train"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/train_val.txt" # train_overfit, train_val,
        split_path["BRATS18"]["val"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/val_test.txt" # train_overfit, val_test
        split_path["BRATS18"]["test"] = "/data/hjlee/orhun/data/BRATS18/BRATS18_Preprocessed/test.txt"

        img_path["BRATS23"] = "/data/hjlee/orhun/data/BRATS21_Processed/Training/Images" # "data/BRATS/Images"
        seg_path["BRATS23"] = "/data/hjlee/orhun/data/BRATS21_Processed/Training/Labels"
        split_path["BRATS23"]["train"] = "/data/hjlee/orhun/data/BRATS21_Processed/train.txt"
        split_path["BRATS23"]["val"] = "/data/hjlee/orhun/data/BRATS21_Processed/test.txt"
        split_path["BRATS23"]["test"] = "/data/hjlee/orhun/data/BRATS21_Processed/test.txt"

class Test_config:
        dataset = "BRATS18"
        modalities = ["FLAIR", "T1c", "T1", "T2"] # order for 23: "FLAIR", "T1c", "T1", "T2"
                                                  # order for 18: "FLAIR", "T1", "T1c", "T2"
        crop_size = 128
        n_of_output_c = 3
        feature_size = 12
        model_file_path = "/data/hjlee/orhun/thesis/models/mm12_sd_ds_dicece_rd_evalall_updated/mm12_sd_ds_dicece_rd_evalall_updated_checkpoint_Epoch_420.pt"
        random_dropping = "modality_mean" # {"whole_mean","modality_mean", "pc_modality_mean", "zero"} 
        pc_mean_path = "/mnt/disk1/hjlee/orhun/repo/thesis/pc_mean.pt"