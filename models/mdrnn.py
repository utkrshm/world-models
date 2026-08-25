import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, latents_dim = 64, hiddens_dim = 512, actions_dim = 10, n_mixtures = 5):
        super().__init__()
        
        self.z_size = latents_dim
        self.h_size = hiddens_dim
        self.a_size = actions_dim
        self.n_mixtures = n_mixtures
        
        self.lstm = nn.LSTM(
            input_size=(self.z_size + self.a_size), hidden_size=self.h_size, num_layers=1, batch_first=True
        )
        
        self.fc = nn.Linear(self.h_size, self.n_mixtures*(2*self.z_size+1))
        # Output: For every gaussian mixture, the means of each dimension of the latent, the variances for every dimension and the weight for a particular mixture
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device         # Needed because this creates the tensors in memory, but the tensors don't know which device the model is on... got an error without running this when getting the summary

        #                 (D*num_layers, batch_size, hidden_size) for nn.LSTM
        h = torch.zeros(1, batch_size, self.h_size).to(device)
        c = torch.zeros(1, batch_size, self.h_size).to(device)
        return (h, c)
    
    def split_params(self, params):
        assert params.ndim == 3, "params does not have 3 dimensions"
        
        batch_size, seq_len, _ = params.shape
        
        params = params.view(batch_size, seq_len, self.n_mixtures, 2*self.z_size + 1)
        # 2*z_size+1 for means, stds of each component and the weight associated with each component
        
        _pi = params[:, :, :, 0]       # The single weight associated with each component for each sample
        _means = params[:, :, :, 1:self.z_size+1]
        _logstds = params[:, :, :, 1+self.z_size:]      # We assume that we get log of stds for each component for numerical stability during of the LSTM, to avoid exploding the gradients of these weights
        
        return _pi, _means, _logstds

    
    def forward(self, z, a, hiddens = None):
        x = torch.cat((z, a), dim=2)
        
        if not hiddens:
            hiddens = self.init_hidden(x.shape[0])
        
        x, hiddens = self.lstm(x, hiddens)
        x = self.fc(x)
        
        weights, mus, logsigmas = self.split_params(x)

        sigmas = torch.exp(logsigmas).clamp(min=1e-5)       # Rare case, but might need the clamping
        weights = F.softmax(weights)
                
        return weights, mus, sigmas, hiddens
    

if __name__ == "__main__":
    from torchinfo import summary
    
    latents_dim = 64            # From the VAE model
    actions_dim = 10            # For MsPacman specifically
    hiddens_dim = 512           # From the paper
    n_mixtures = 5              # From the paper
    seq_len = 20                # Random
    batch_size = 32             # Dummy for now
    
    model = Memory(latents_dim, hiddens_dim, actions_dim, n_mixtures).to(device="cpu")
    summary(
        model, 
        input_size=[
            (batch_size, seq_len, latents_dim), # For z
            (batch_size, seq_len, actions_dim), # For a
            # [(seq_len, batch_size, hiddens_dim), (seq_len, batch_size, hiddens_dim)] # For hiddens -> (h, c)
        ], 
        col_names=("input_size", "output_size", "num_params"), 
        col_width=20
    )