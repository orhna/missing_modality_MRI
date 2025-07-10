import torch
import os
from tqdm import tqdm
import utils.utils as utils
import config
import argparse
import json
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from nets.multimodal_swinunetr_multirecon import Multimodal_SwinUNETR

if __name__ == "__main__":
    
    torch.multiprocessing.set_sharing_strategy('file_system') 
    #command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=str)
    args = parser.parse_args()

    device_list = list(map(int, args.device_id.split(',')))   
    
    Test_config = config.Test_config()
    Database_config = config.Database_config()

    chosen_ds = Test_config.dataset
    # path initialization 
    img_path=Database_config.img_path[chosen_ds]
    seg_path=Database_config.seg_path[chosen_ds]
    split_path=Database_config.split_path[chosen_ds]["test"]

    n_of_channels = len(Test_config.modalities)
    channel_indices = []
    modalities_to_train = Test_config.modalities
    for _m in modalities_to_train:
        channel_indices.append(Database_config.channels[chosen_ds].index(_m))
    print("Channel indices to be loaded :",channel_indices)

    pc_mean_tensor = None
    if Test_config.random_dropping == "pc_modality_mean":
        pc_mean_tensor = torch.load(Test_config.pc_mean_path)[channel_indices,...]  # Shape: (4, 240, 240, 155)

    test_loader = utils.get_test_loader(img_path, seg_path, split_path, channel_indices)

    # initialize loss and metrics according to output channel
    loss_function, dice_metric, sensitivity_metric, precision_metric, IOU_metric, post_trans = utils.initialize_loss_metric(new_eval=True)

    device = utils.initialize_GPU(device_list[0])

    model = Multimodal_SwinUNETR(
        img_size=(Test_config.crop_size, Test_config.crop_size, Test_config.crop_size),
        in_channels=1, 
        out_channels=Test_config.n_of_output_c,
        feature_size=Test_config.feature_size,
        use_checkpoint=True,
        recon_level= "none").to(device)
        
    checkpoint = torch.load(Test_config.model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # for .pt
    #model.load_state_dict(checkpoint) # for .pth
    model.eval()  # Set model to evaluation mode
    model.is_training=False

    all_results = {}
    scenarios = utils.get_test_scenarios()
    print("random dropping method:",Test_config.random_dropping)


    for scenario in scenarios:
        idx_to_drop = []
        for i in range(len(scenario)):
            if not scenario[i] == 1:
                idx_to_drop.append(i)

        with torch.no_grad():
            val_outputs = None
            metric={}

            print("to be dropped:",idx_to_drop)
            if len(idx_to_drop) >0:
                remaining_modalities = list(set(range(4)) - set(idx_to_drop))
                print("keep:",remaining_modalities)
     
            for val_data in tqdm(test_loader):
                val_input = val_data[0]
                val_input = utils.test_random_dropping(val_input, Test_config.random_dropping, idx_to_drop, remaining_modalities, pc_mean_tensor)
                val_input = val_input.to(device)
                val_labels = val_data[1].to(device)                        
                roi_size = (Test_config.crop_size, Test_config.crop_size, Test_config.crop_size)
                sw_batch_size = 1
                #using sliding window for the whole 3D image
                val_outputs = sliding_window_inference(val_input, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric["dice"] = dice_metric.aggregate().cpu().numpy().tolist()
            print(metric["dice"])
            
            metric["dice_ET"] = dice_metric.aggregate()[2].item()
            metric["dice_TC"] = dice_metric.aggregate()[0].item()
            metric["dice_WT"] = dice_metric.aggregate()[1].item()

            dice_metric.reset()

        all_results[' '.join(map(str, scenario))] = metric

    json_dump_path = os.path.dirname(Test_config.model_file_path) + "/results_Rmodmean.json"

    with open(json_dump_path, "w") as json_file:
        json.dump(all_results, json_file, indent=4)