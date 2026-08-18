import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, latents_dim, hiddens_dim, actions_dim, n_mixtures):
        super().__init__()
        
        self.z_size = latents_dim
        self.h_size = hiddens_dim
        self.a_size = actions_dim
        self.n_mixtures = n_mixtures
        
        self.lstm = nn.LSTM(
            input_size=(self.z_size + self.a_size), hidden_size=self.h_size, num_layers=1, batch_first=True
        )
        
        self.fc = nn.Linear(self.z_size, self.n_mixtures*(2*self.z_size+1))
        # Output: For every gaussian mixture, the means of each dimension of the latent, the variances for every dimension and the weight for a particular mixture
        
    def forward(self, z, a, h):
        return
        # return _pi, _means, _vars, h_next