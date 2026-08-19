import torch
from torch import nn

"""Current implementation exactly mirrors the paper's architecture"""
class VAE(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        
        self.z_size = hidden_size
        
        self.enc_layers = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        self.mu_layer = nn.Linear(in_features=2*2*256, out_features=self.z_size)
        self.logvar_layer = nn.Linear(in_features=2*2*256, out_features=self.z_size)
        
        self.proj_dec_layer = nn.Sequential(
            nn.Linear(self.z_size, 1024),
            nn.ReLU()
        )
        self.dec_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        x = self.enc_layers(x)
        x = x.view(x.size(0), -1)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        
        return mu + eps * sigma         # Reparameterization trick
    
    def decode(self, z):
        x = self.proj_dec_layer(z)
        x = x.reshape(-1, 1024, 1, 1)
        return self.dec_layers(x)        

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


if __name__ == "__main__":
    from torchinfo import summary
    
    model = VAE()
    summary(model, input_size=(10, 3, 64, 64), col_names=("input_size", "output_size", "num_params"), col_width=20)