import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """Dense autoencoder over per-frame feature vectors (no CAN ID embedding).

    Architecture:
        Input (13: 4 ID-behavioral stats + 8 bytes + DLC)
            -> Linear(13, 8) -> ReLU
            -> Linear(8, 4) -> ReLU      (bottleneck)
            -> Linear(4, 8) -> ReLU
            -> Linear(8, 13) -> Tanh
    """

    def __init__(self, input_dim=13):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def get_reconstruction_error(self, x):
        self.eval()
        recon = self.forward(x)
        mse = F.mse_loss(recon, x, reduction="none").mean(dim=1)
        return mse
