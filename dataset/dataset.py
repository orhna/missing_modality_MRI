import os
import torch
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    def __init__(self, source_folder, target_folder):

        self.source_folder = source_folder
        self.target_folder = target_folder

        self.source_files = sorted(os.listdir(source_folder))
        self.target_files = sorted(os.listdir(target_folder))
        
    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self, idx):
        # Get the file names
        source_file = os.path.join(self.source_folder, self.source_files[idx])
        target_file = os.path.join(self.target_folder, self.target_files[idx])
        
        # Load the tensors
        source_tensor = torch.load(source_file).squeeze(1)  # Shape: (768, 4, 4, 4)
        target_tensor = torch.load(target_file).squeeze(1)    # Shape: (768, 4, 4, 4)
        
        # Stack them into the desired shape
        #combined_tensor = torch.stack([source_tensor, target_tensor], dim=0)  # Shape: (2, 768, 4, 4, 4)
        #return combined_tensor.squeeze(1)
        return (source_tensor, target_tensor)