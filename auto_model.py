# Define Model and Loss
# --------------------------------------------
import torch
import torch.nn as nn

class EnhancedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent   

    def encode(self, x):
        return self.encoder(x)

def weighted_mse_loss(output, target):
    weights = torch.tensor([0.5, 0.2, 0.3, 0.0, 0.0], device=output.device)
    return ((weights * (output - target) ** 2).mean())