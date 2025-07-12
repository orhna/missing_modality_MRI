# 🧠 Project Title

Brief project description here (what the repo does, key goals, etc.)


## 📦 Installation

### 1. Clone the repository
### 2. Create new virtual environment
### 3. Install required packages
```bash
pip install -r requirements.txt
```

git clone https://github.com/yourusername/your-repo.git
cd your-repo

## 📦 Configuration
Use `config.py` to set up dataset, datalist and model saving paths. For example:

`model_save_path = "./models/"`


`img_path["BRATS18"] = "/data/Images" `
`seg_path["BRATS18"] = "/data/Labels" `
`split_path["BRATS18"]["train"] = "/datasplit/train.txt"` 
`split_path["BRATS18"]["val"] = "/datasplit/val.txt"`
`split_path["BRATS18"]["test"] = "/datasplit/test.txt"`

## 📦 Training

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
