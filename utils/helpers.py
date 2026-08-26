"""Script for helpers throughout the process"""
from typing import Optional
import torch
import numpy as np
import random
import os

def seed(rng_seed, torch_seed: int | None, numpy_seed: int | None):
    """
    Only specify torch_seed and numpy_seed if you want to use separate seeds for them. 
    `rng_seed` will apply to all whose value has not been specified.
    """
    if torch_seed is None: torch_seed = rng_seed
    if numpy_seed is None: numpy_seed = rng_seed
    
    
    random.seed(rng_seed)
    
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)

    np.random.seed(numpy_seed)