import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.model import SiameseNetwork


DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
MODEL_DIR.mkdir(exist_ok=True)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Generates triplets on-the-fly:
        anchor   = random sample from normal data
        positive = another random sample from normal data (same CAN_ID preferred,
                   per paper Section 4.1: comparisons are meaningful per-ID since
                   normal behavior is ID-specific)
        negative = random sample from attack data (same CAN_ID preferred, so the
                   network learns to separate normal-vs-attack *within* an ID,
                   rather than partly just learning to separate different IDs)
    """

    def __init__(self, normal_features, normal_ids, attack_features, attack_ids=None):
        self.normal = torch.tensor(normal_features, dtype=torch.float32)
        self.normal_ids = normal_ids
        self.attack = torch.tensor(attack_features, dtype=torch.float32)
        self.attack_ids = attack_ids

        # Build index: CAN_ID -> list of sample indices (normal)
        self.id_to_normal_indices = {}
        for i, cid in enumerate(normal_ids):
            self.id_to_normal_indices.setdefault(cid, []).append(i)

        # Build index: CAN_ID -> list of sample indices (attack), if IDs provided
        self.id_to_attack_indices = {}
        if attack_ids is not None:
            for i, cid in enumerate(attack_ids):
                self.id_to_attack_indices.setdefault(cid, []).append(i)

    def __len__(self):
        return len(self.normal)

    def __getitem__(self, idx):
        anchor = self.normal[idx]
        anchor_id = self.normal_ids[idx]

        # --- Positive: prefer same CAN_ID, else random (guarded against idx==pos_idx either way) ---
        same_id_indices = self.id_to_normal_indices.get(anchor_id, [])
        if len(same_id_indices) > 1:
            pos_idx = idx
            while pos_idx == idx:
                pos_idx = same_id_indices[np.random.randint(len(same_id_indices))]
        else:
            pos_idx = idx
            while pos_idx == idx:
                pos_idx = np.random.randint(0, len(self.normal))
        positive = self.normal[pos_idx]

        # --- Negative: prefer same CAN_ID attack sample if available, else random attack sample ---
        same_id_attack_indices = self.id_to_attack_indices.get(anchor_id, [])
        if len(same_id_attack_indices) > 0:
            neg_idx = same_id_attack_indices[np.random.randint(len(same_id_attack_indices))]
        else:
            neg_idx = np.random.randint(0, len(self.attack))
        negative = self.attack[neg_idx]

        return anchor, positive, negative


# ──────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────

def load_normal_data():
    """Load all normal (attack-free) training data with CAN_IDs."""
    df = pd.read_csv(DATA_DIR / "Attack_free_training.csv")
    feat_cols = [f"freq_bit_{i}" for i in range(64)]
    return df[feat_cols].values.astype(np.float32), df["CAN_ID"].values


def load_attack_data():
    """Load all attack training data (DoS + Fuzzy + Impersonation), with CAN_IDs."""
    frames = []
    id_frames = []
    for name in ["DoS_attack_training.csv", "Fuzzy_attack_training.csv", "Impersonation_attack_training.csv"]:
        df = pd.read_csv(DATA_DIR / name)
        feat_cols = [f"freq_bit_{i}" for i in range(64)]
        frames.append(df[feat_cols].values.astype(np.float32))
        id_frames.append(df["CAN_ID"].values)
    return np.vstack(frames), np.concatenate(id_frames)


# ──────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    losses = []
    for anchor, positive, negative in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a, emb_p, emb_n = model(anchor, positive, negative)
        loss = model.triplet_loss(emb_a, emb_p, emb_n)
        losses.append(loss.item())

    return np.mean(losses)


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

def train(
    epochs=50,
    batch_size=64,
    lr=0.001,          # matches paper Section 4.2: "the learning rate of the network
                        # training stage is 0.001" — kept as the default here
    margin=1.0,         # paper does not state a numeric margin (alpha) value;
                        # this is a chosen default, not taken from the paper
    embedding_dim=16,   # paper leaves embedding dim d unspecified; chosen default
    hidden_dims=None,   # NEW: lets you sweep hidden-layer count, matching the
                         # paper's own Figure 6 experiment (2-16 hidden layers)
    val_ratio=0.15,      # NEW: normal/attack val split
    test_ratio=0.15,     # NEW: held-out test split, untouched by training/early-stopping
    patience=10,
    seed=42,
    device=None,
):
    """
    Deviations from the paper, documented here for the project writeup:

      1. Mode/value information split (Section 4.1) is not implemented — the
         paper's own description is ambiguous and appears inconsistent with the
         bit-index ranges shown in Figure 5. The full 64-bit frequency vector is
         used uniformly at both training and detection.

      2. The paper's private CANoe-generated dataset is unavailable; a public
         substitute (Car-Hacking-style: DoS / Fuzzy / Impersonation attacks) is
         used instead. Absolute numbers should not be expected to match the
         paper's reported results — the goal is to reproduce the *trend*
         (DNN+Triplet outperforming DNN+SVM / DNN+Softmax as hidden layers
         increase, then plateauing).

      3. The paper does not state embedding dimension, margin value, batch size,
         or exact train/val/test split for the attack portion of the data — all
         of these are set here as explicit, documented defaults rather than
         values taken from the paper.
    """
    # ── Seed ──
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # ── Load data ──
    print("Loading data...")
    normal_feats, normal_ids = load_normal_data()
    attack_feats, attack_ids = load_attack_data()
    print(f"Normal samples: {len(normal_feats):,}")
    print(f"Attack samples: {len(attack_feats):,}")

    # ── Train / Validation / Test split ──
    # Paper (Section 4.1) only specifies the split for *normal* packets (70/30
    # train/validate). It does not state how attack packets are split. Here,
    # BOTH normal and attack data are split the same way, and independently of
    # each other, to avoid leaking attack samples between train/val/test
    # (see earlier review: this was a real bug in the previous version, where
    # the full unsplit attack pool was shared between train and val).
    normal_idx = np.arange(len(normal_feats))
    n_trainval, n_test = train_test_split(
        normal_idx, test_size=test_ratio, random_state=seed, shuffle=True
    )
    n_train, n_val = train_test_split(
        n_trainval, test_size=val_ratio / (1 - test_ratio), random_state=seed, shuffle=True
    )

    attack_idx = np.arange(len(attack_feats))
    a_trainval, a_test = train_test_split(
        attack_idx, test_size=test_ratio, random_state=seed, shuffle=True
    )
    a_train, a_val = train_test_split(
        a_trainval, test_size=val_ratio / (1 - test_ratio), random_state=seed, shuffle=True
    )

    normal_train_feats, normal_train_ids = normal_feats[n_train], normal_ids[n_train]
    normal_val_feats, normal_val_ids = normal_feats[n_val], normal_ids[n_val]
    normal_test_feats, normal_test_ids = normal_feats[n_test], normal_ids[n_test]

    attack_train_feats, attack_train_ids = attack_feats[a_train], attack_ids[a_train]
    attack_val_feats, attack_val_ids = attack_feats[a_val], attack_ids[a_val]
    attack_test_feats, attack_test_ids = attack_feats[a_test], attack_ids[a_test]

    print(f"\nSplit (normal / attack):")
    print(f"  Train:      {len(n_train):,} / {len(a_train):,}")
    print(f"  Validation: {len(n_val):,} / {len(a_val):,}")
    print(f"  Test:       {len(n_test):,} / {len(a_test):,}  (held out, untouched until final evaluate.py run)")

    # Save the test split to disk so evaluate.py can use the SAME held-out set
    # this run trained against (important: don't re-split randomly in evaluate.py,
    # or you risk silently testing on data the model has already seen).
    np.savez(
        DATA_DIR / "test_split.npz",
        normal_test_feats=normal_test_feats,
        normal_test_ids=normal_test_ids,
        attack_test_feats=attack_test_feats,
        attack_test_ids=attack_test_ids,
    )

    # ── Datasets & loaders ──
    train_dataset = TripletDataset(normal_train_feats, normal_train_ids, attack_train_feats, attack_train_ids)
    val_dataset = TripletDataset(normal_val_feats, normal_val_ids, attack_val_feats, attack_val_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"  Train batches:      {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # ── Model ──
    # hidden_dims exposed so you can reproduce the paper's Figure 6 sweep, e.g.:
    #   train(hidden_dims=[64, 64])          # ~2 hidden layers
    #   train(hidden_dims=[64]*7)            # ~14 hidden layers
    model_kwargs = dict(input_dim=64, embedding_dim=embedding_dim, margin=margin)
    if hidden_dims is not None:
        model_kwargs["hidden_dims"] = hidden_dims  # requires SharedDNN to accept this kwarg
    model = SiameseNetwork(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs (early stopping: patience={patience})")
    print(f"{'='*60}")

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epoch = 0  # guard in case epochs=0 is ever passed

    for epoch in range(1, epochs + 1):

        # --- Train ---
        model.train()
        train_losses = []
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a, emb_p, emb_n = model(anchor, positive, negative)
            loss = model.triplet_loss(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        avg_val_loss = validate(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        improved = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            improved = " *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
                "embedding_dim": embedding_dim,
                "hidden_dims": hidden_dims,
                "margin": margin,
                "seed": seed,
            }, MODEL_DIR / "best_model.pth")
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f}{improved}")

        # --- Early stopping ---
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── Save final model ──
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_train_loss,
        "embedding_dim": embedding_dim,
        "hidden_dims": hidden_dims,
        "margin": margin,
        "seed": seed,
    }, MODEL_DIR / "final_model.pth")

    # ── Save history ──
    history_df = pd.DataFrame(history)
    history_df.to_csv(DATA_DIR / "training_history.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Held-out test split saved to: {DATA_DIR / 'test_split.npz'} (use in evaluate.py)")
    print(f"{'='*60}")

    return model, history


if __name__ == "__main__":
    train()