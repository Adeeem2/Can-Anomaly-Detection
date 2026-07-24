import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.model import SiameseNetwork


DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
MODEL_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Generates triplets on-the-fly:
        anchor  = random sample from normal data
        positive = another random sample from normal data
        negative = random sample from attack data
    """

    def __init__(self, normal_features, attack_features):
        self.normal = torch.tensor(normal_features, dtype=torch.float32)
        self.attack = torch.tensor(attack_features, dtype=torch.float32)

    def __len__(self):
        return len(self.normal)

    def __getitem__(self, idx):
        anchor = self.normal[idx]

        # Positive: random different normal sample
        pos_idx = torch.randint(0, len(self.normal), (1,)).item()
        while pos_idx == idx:
            pos_idx = torch.randint(0, len(self.normal), (1,)).item()
        positive = self.normal[pos_idx]

        # Negative: random attack sample
        neg_idx = torch.randint(0, len(self.attack), (1,)).item()
        negative = self.attack[neg_idx]

        return anchor, positive, negative


# ──────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────

def load_normal_data():
    """Load all normal (attack-free) training data."""
    df = pd.read_csv(DATA_DIR / "Attack_free_training.csv")
    feat_cols = [f"freq_bit_{i}" for i in range(64)]
    return df[feat_cols].values.astype(np.float32)


def load_attack_data():
    """Load all attack training data (DoS + Fuzzy + Impersonation)."""
    frames = []
    for name in ["DoS_attack_training.csv", "Fuzzy_attack_training.csv", "Impersonation_attack_training.csv"]:
        df = pd.read_csv(DATA_DIR / name)
        feat_cols = [f"freq_bit_{i}" for i in range(64)]
        frames.append(df[feat_cols].values.astype(np.float32))
    return np.vstack(frames)


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

def train(
    epochs=50,
    batch_size=64,
    lr=0.001,
    margin=1.0,
    embedding_dim=16,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    normal_feats = load_normal_data()
    attack_feats = load_attack_data()
    print(f"Normal samples: {len(normal_feats):,}")
    print(f"Attack samples: {len(attack_feats):,}")

    # Create dataset and dataloader
    dataset = TripletDataset(normal_feats, attack_feats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Batches per epoch: {len(dataloader)}")

    # Model
    model = SiameseNetwork(input_dim=64, embedding_dim=embedding_dim, margin=margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs")
    print(f"{'='*60}")

    history = {"epoch": [], "loss": []}
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a, emb_p, emb_n = model(anchor, positive, negative)
            loss = model.triplet_loss(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "embedding_dim": embedding_dim,
                "margin": margin,
            }, MODEL_DIR / "best_model.pth")

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "embedding_dim": embedding_dim,
        "margin": margin,
    }, MODEL_DIR / "final_model.pth")

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(DATA_DIR / "training_history.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"{'='*60}")

    return model, history


if __name__ == "__main__":
    train()
