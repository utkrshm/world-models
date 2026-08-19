import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MsPacmanDataset(Dataset):
    def __init__(self, root_dir: Path | str):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.files = sorted(glob.glob("*.npz", root_dir=self.root_dir))
        
        self.index_map = []
        # Each file contains multiple observations, packed with action, reward, and lives info, so we'll index them all
        for file_idx, file in enumerate(self.files):
            data = np.load(self.root_dir / file)
            
            for frame_idx in range(len(data['observations'])):
                self.index_map.append((file_idx, frame_idx))
                
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index) -> Any:
        file_idx, frame_idx = self.index_map[index]
        data = np.load(self.root_dir / self.files[file_idx])
        
        # The observations are already normalized to be in the 0-1 range, so only the shape change has to happen now
        obs = data["observations"][frame_idx]
        return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)   # For now, for the VAE, only the observations are needed
