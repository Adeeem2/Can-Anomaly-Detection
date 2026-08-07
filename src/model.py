import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """Conditional autoencoder: learned CAN ID embedding + ID behavioral stats.

    Architecture:
        CAN ID (int index 0-30) ──→ Embedding(31, 6) ──┐
                                                          ├── concat(6+13=19) → Encoder → Bottleneck(8)
        Features (13: 4 stats + 8 bytes + DLC) ──────────┘
                                                          ↓ latent (8)
        cat(latent(8), embedding(6)) = 14 → Decoder → 13 → Sigmoid

    The embedding is a conditioning input to both encoder and decoder; the
    decoder reconstructs ONLY the 13-dim features (all in [0,1]) via Sigmoid.
    """

    def __init__(self, num_ids=31, embedding_dim=6, input_dim=13):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        total_dim = embedding_dim + input_dim

        self.encoder = nn.Sequential(
            nn.Linear(total_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8 + embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, id_idx, feats):
        id_emb = self.id_embedding(id_idx)
        x = torch.cat([id_emb, feats], dim=1)
        latent = self.encoder(x)
        return self.decoder(torch.cat([latent, id_emb], dim=1))

    @torch.no_grad()
    def get_reconstruction_error(self, id_idx, feats):
        self.eval()
        recon = self.forward(id_idx, feats)
        mse = F.mse_loss(recon, feats, reduction="none").mean(dim=1)
        return mse


class LSTMAutoencoder(nn.Module):
    """Sequence autoencoder over windows of consecutive frames.

    Encoder LSTM compresses a (W, F) window into a hidden state; the decoder
    LSTM reconstructs the window from the repeated last hidden state. The
    per-window reconstruction error is the MSE over all W*F values.
    """

    def __init__(self, input_dim=15, hidden=64, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(z)
        return out

    @torch.no_grad()
    def get_reconstruction_error(self, x):
        self.eval()
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
