import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """Dense autoencoder with learned CAN ID embedding for anomaly detection.

    Architecture:
        CAN ID (int index 0-30) ──→ Embedding(31, 6) ──┐
                                                         ├── concat(6+9=15) → Encoder → Bottleneck(8) → Decoder → 15
        Payload (8 bytes + DLC) ────────────────────────┘
    """

    def __init__(self, num_ids=31, embedding_dim=6, payload_dim=9):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        total_dim = embedding_dim + payload_dim

        self.encoder = nn.Sequential(
            nn.Linear(total_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, total_dim),
            nn.Tanh()

        )

    def forward(self, id_idx, payload):
        id_emb = self.id_embedding(id_idx)
        x = torch.cat([id_emb, payload], dim=1)
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def get_reconstruction_error(self, id_idx, payload):
        self.eval()
        id_emb = self.id_embedding(id_idx)
        x = torch.cat([id_emb, payload], dim=1)
        recon = self.forward(id_idx, payload)
        mse = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        return mse


