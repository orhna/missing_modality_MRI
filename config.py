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
        checkpoint_path = "/models/experiment_name/experiment_name_checkpoint.pt"
        # model_save_path 
        model_save_path = "/models/"
        cropped_input_size = [128,128,128]   
        output_channel = 3
        workers = 2

class Database_config:
        channels={}
        channels['BRATS18'] = ["FLAIR", "T1", "T1c", "T2"]
        channels['BRATS23'] = ["T1c", "T1", "FLAIR", "T2"]

        img_path={}
        seg_path={}
        split_path={}
        split_path["BRATS18"]={}
        split_path["BRATS23"]={}
        
        img_path["BRATS18"] = "/data/Images" 
        seg_path["BRATS18"] = "/data/Labels"
        split_path["BRATS18"]["train"] = "/datasplit/train.txt" 
        split_path["BRATS18"]["val"] = "/datasplit/val.txt" 
        split_path["BRATS18"]["test"] = "/datasplit/test.txt"

        img_path["BRATS23"] = "/data/Images"
        seg_path["BRATS23"] = "/data/Labels"
        split_path["BRATS23"]["train"] = "/datasplit/train.txt"
        split_path["BRATS23"]["val"] = "/datasplit/val.txt"
        split_path["BRATS23"]["test"] = "/datasplit/test.txt"

class Test_config:
        dataset = "BRATS18"
        modalities = ["FLAIR", "T1c", "T1", "T2"] # order for 23: "FLAIR", "T1c", "T1", "T2"
                                                  # order for 18: "FLAIR", "T1", "T1c", "T2"
        crop_size = 128
        n_of_output_c = 3
        feature_size = 12
        model_file_path = "/models/experiment_name/experiment_name_checkpoint.pt"
        random_dropping = "modality_mean" # {"whole_mean","modality_mean", "pc_modality_mean", "zero"} 
