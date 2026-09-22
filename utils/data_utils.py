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
            frame_count = np.load(self.root_dir / file, mmap_mode="r").shape[0]
            self.index_map.extend((file_idx, frame_idx) for frame_idx in range(frame_count))
                
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index) -> Any:
        file_idx, frame_idx = self.index_map[index]
        data = np.load(self.root_dir / self.files[file_idx], mmap_mode="r")
        data = data[frame_idx]
        
        # The observations are already normalized to be in the 0-1 range, so only the shape change has to happen now
        return torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)   # For now, for the VAE, only the observations are needed

# For this dataset, sequence matters, so I gotta send the entire episode at once, instead of frame-wise chunking
class RNNDataset(Dataset):
    def __init__(self, root_dir: Path | str, n_actions: int = 9):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.obs_files = sorted(glob.glob("*_observations.npy", root_dir=root_dir))
        self.act_files = sorted(glob.glob("*_actions.npy", root_dir=self.root_dir))
        self.reward_files = sorted(glob.glob("*_rewards.npy", root_dir=self.root_dir))
        self.lives_files = sorted(glob.glob("*_lives.npy", root_dir=self.root_dir))
        
        self.n_actions = n_actions
    
    def __len__(self):
        return len(self.obs_files)
    
    def __getitem__(self, index):        
        obs = np.load(file = self.root_dir / self.obs_files[index])         # (seq_len, H, W, C)
        action = np.load(file = self.root_dir / self.act_files[index])      # (seq_len,)
        rewards = np.load(file = self.root_dir / self.reward_files[index])  # (seq_len,)
        lives = np.load(file = self.root_dir / self.lives_files[index])     # (seq_len,)
        
        return (
            torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2),        # raw, unpadded observations (seq_len, C, H, W)
            F.one_hot(torch.tensor(action), num_classes=self.n_actions),       # actions per obs, one-hot encoded (seq_len, n_actions)
            torch.tensor(rewards, dtype=torch.float32),                        # reward at each timestep (seq_len,)
            torch.tensor(lives, dtype=torch.long),                             # remaining lives at each timestep (seq_len,)
        )
