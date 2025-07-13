# 🧠 Tackling Missing Modality Problem in Multi-modal Medical Segmentation

This project explores various strategies to improve brain tumor segmentation in the presence of missing MRI modalities using the BRATS2018 and BRATS2023 datasets. We systematically evaluate and compare several methods, promising ones are provided here for anyone who wants to replicate. Extensive experiments demonstrate that our method achieves competitive performance and generalizes well.  

## 📦 Installation

### 1. Clone the repository
### 2. Create new virtual environment
### 3. Install required packages
```bash
pip install -r requirements.txt
```
### 4. Preprocess the dataset
Download the dataset from [here](https://www.synapse.org/Synapse:syn51156910/wiki/622351), unzip the file and run  
`python preprocess.py /path/to/BRATS21/Training /path/to/BRATS21_Processed/Training`

## ⚙️ Configuration  
Use `config.py` to set up dataset, datalist and model saving paths. For example:  

`model_save_path = "./models/"`  

`img_path["BRATS18"] = "/data/Images" `  
`seg_path["BRATS18"] = "/data/Labels" `  
`split_path["BRATS18"]["train"] = "/datasplit/train.txt"`  
`split_path["BRATS18"]["val"] = "/datasplit/val.txt"`  
`split_path["BRATS18"]["test"] = "/datasplit/test.txt"`  

## 🏋️ Training  

Use `config.py` to set up experiment name:  
`experiment_name = "your_experiment_name"`  

Set `recon_level` to `"hs3_hs4"` for training with learnable token method, or use `"none"` to train without explicit feature reconstruction method.  

Then run:  
`python train.py --device_id 0`  

And for joint memory training on two modality setting, run:  
`python train2.py --device_id 0`  

Logs will be saved under `/logs/your_experiment_name`  


## 🔍 Inference  
Set the testing parameters in `config.py` -> `Test_config`, then run:  
`python test.py --device_id 0`  
