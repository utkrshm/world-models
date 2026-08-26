import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class VAEDataset(Dataset):
    def __init__(self, root_dir: Path | str):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.files = sorted(glob.glob("*_observations.npy", root_dir=self.root_dir))
        
        self.index_map = []
        # Each file contains multiple observations of a single episode, so we'll index them all
        for file_idx, file in enumerate(self.files):
            data = np.load(self.root_dir / file)
            
            for frame_idx in range(len(data)):
                self.index_map.append((file_idx, frame_idx))
                
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index) -> Any:
        file_idx, frame_idx = self.index_map[index]
        data = np.load(self.root_dir / self.files[file_idx], mmap_mode="r")
        data = data[frame_idx] / 255.0
        
        # The observations are already normalized to be in the 0-1 range, so only the shape change has to happen now
        return torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)   # For now, for the VAE, only the observations are needed
