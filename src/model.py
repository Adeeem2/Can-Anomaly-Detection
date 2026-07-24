import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedDNN(nn.Module):
    """
    Shared-weight DNN that maps a 64-dim frequency vector
    to a 16-dim embedding.

    Architecture (from the paper):
        Input  (64)  -> Hidden (128) -> Hidden (64) -> Hidden (32) -> Output (16)

    All three Siamese branches share the same instance of this network.
    """

    def __init__(self, input_dim=64, embedding_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, embedding_dim),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        # L2-normalize so embedding lives on the unit hypersphere
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class TripletLoss(nn.Module):
    """
    Triplet Loss:  L = max(0, ||a - p||^2 - ||a - n||^2 + margin)

    Minimizes distance between anchor and positive,
    maximizes distance between anchor and negative.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2) ** 2
        neg_dist = F.pairwise_distance(anchor, negative, p=2) ** 2
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class SiameseNetwork(nn.Module):
    """
    Wraps the shared DNN into a Siamese triplet structure.

    Forward pass:
        anchor, positive, negative -> (embedding_a, embedding_p, embedding_n)

    The triplet loss is computed from these three embeddings.
    """

    def __init__(self, input_dim=64, embedding_dim=16, margin=1.0):
        super().__init__()
        self.shared_dnn = SharedDNN(input_dim, embedding_dim)
        self.triplet_loss = TripletLoss(margin)

    def forward(self, anchor, positive, negative):
        emb_a = self.shared_dnn(anchor)
        emb_p = self.shared_dnn(positive)
        emb_n = self.shared_dnn(negative)
        return emb_a, emb_p, emb_n

    def get_embedding(self, x):
        """Single input embedding (used at detection time)."""
        return self.shared_dnn(x)
