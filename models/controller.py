import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, latents, hiddens, n_actions):
        """
        Args:
            latents = z from the VAE / Memory
            hiddens = h from the Memory
            n_actions = size of the output / number of actions
        """
        super().__init__()
        
        self.fc = nn.Linear(latents+hiddens, n_actions)
        
    def forward(self, z, h):
        x = torch.cat([z, h], dim=1)
        x = self.fc(x)
        return x