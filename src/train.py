import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.model import SiameseNetwork

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
MODEL_DIR.mkdir(exist_ok=True)

TOP_K = 30
BYTE_COLS = [f"byte_{i}" for i in range(8)]


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def fit_top_ids(can_ids, k=TOP_K):
    """Learn top-k CAN IDs from training normal data only."""
    counts = pd.Series(can_ids).value_counts()
    top = counts.head(k).index.tolist()
    print(f"Top-{k} IDs: {top[:5]} ... ({len(top)} total)")
    return top


def build_features(can_ids, top_ids, byte_arr, dlc_arr):
    """Build 40-dim feature matrix: [31 one-hot | 8 bytes | 1 dlc]."""
    n = len(can_ids)
    f = np.zeros((n, 40), dtype=np.float32)
    id_map = {cid: i for i, cid in enumerate(top_ids)}
    for i, cid in enumerate(can_ids):
        f[i, id_map.get(cid, TOP_K)] = 1.0  # column TOP_K = "other"
    f[:, TOP_K + 1 : TOP_K + 9] = byte_arr  # already /255 from parser
    f[:, TOP_K + 9] = dlc_arr                # already /8 from parser
    return f

# ── Dataset ──────────────────────────────────

class TripletDataset(Dataset):
    """triplet generation with same-ID preference."""

    def __init__(self, normal_feats, normal_ids, attack_feats, attack_ids):
        self.normal = torch.tensor(normal_feats, dtype=torch.float32)
        self.attack = torch.tensor(attack_feats, dtype=torch.float32)
        self.normal_ids = normal_ids
        self.id_to_normal = self._group(normal_ids)
        self.id_to_attack = self._group(attack_ids)

    @staticmethod
    def _group(ids):
        g = {}
        for i, cid in enumerate(ids):
            g.setdefault(cid, []).append(i)
        return g

    def __len__(self):
        return len(self.normal)

    def __getitem__(self, idx):
        aid = self.normal_ids[idx]

        pos_pool = self.id_to_normal.get(aid, [])
        pos = pos_pool[np.random.randint(len(pos_pool))] if len(pos_pool) > 1 else np.random.randint(len(self.normal))

        neg_pool = self.id_to_attack.get(aid, [])
        neg = neg_pool[np.random.randint(len(neg_pool))] if neg_pool else np.random.randint(len(self.attack))

        return self.normal[idx], self.normal[pos], self.attack[neg]


# ── Data loading ─────────────────────────────

def chronological_split(n, val_ratio=0.15):
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n)
    return train_idx, val_idx


def prepare_data(val_ratio=0.15):
    """Load pre-built .npz (fast) or fall back to CSV parsing + build_features."""
    npz_path = DATA_DIR / "train_data.npz"
    if npz_path.exists():
        print("Loading pre-built train_data.npz ...")
        d = np.load(npz_path, allow_pickle=True)
        splits = {
            "train": (d["train_feats"], d["train_ids"],
                      d["train_attack_feats"], d["train_attack_ids"],
                      d["train_attack_types"]),
            "val":   (d["val_feats"], d["val_ids"],
                      d["val_attack_feats"], d["val_attack_ids"],
                      d["val_attack_types"]),
        }
        print(f"  train: {len(d['train_feats']):,} / {len(d['train_attack_feats']):,}")
        print(f"  val:   {len(d['val_feats']):,} / {len(d['val_attack_feats']):,}")
        return splits

    df = pd.read_csv(DATA_DIR / "set01_train_frames.csv")

    n_mask = df["attack"].values == 0
    a_mask = df["attack"].values == 1

    n_ids = df.loc[n_mask, "CAN_ID"].astype(str).values
    n_ts = df.loc[n_mask, "Timestamp"].values
    n_bytes = df.loc[n_mask, BYTE_COLS].values.astype(np.float32)
    n_dlc = df.loc[n_mask, "DLC"].values.astype(np.float32)

    a_ids = df.loc[a_mask, "CAN_ID"].astype(str).values
    a_ts = df.loc[a_mask, "Timestamp"].values
    a_bytes = df.loc[a_mask, BYTE_COLS].values.astype(np.float32)
    a_dlc = df.loc[a_mask, "DLC"].values.astype(np.float32)
    a_types = df.loc[a_mask, "attack_type"].values.astype(str)

    # Sort normal chronologically and split
    n_order = n_ts.argsort()
    n_ids, n_ts, n_bytes, n_dlc = (
        n_ids[n_order], n_ts[n_order], n_bytes[n_order], n_dlc[n_order]
    )
    ni, nv = chronological_split(len(n_ids), val_ratio)

    # Split attack stratified by type (chronological within each type)
    ai_idx, av_idx = [], []
    for atype in np.unique(a_types):
        type_mask = a_types == atype
        type_orig_idx = np.where(type_mask)[0]
        type_order = a_ts[type_mask].argsort()
        type_orig_idx = type_orig_idx[type_order]
        ti, tv = chronological_split(len(type_orig_idx), val_ratio)
        ai_idx.extend(type_orig_idx[ti])
        av_idx.extend(type_orig_idx[tv])
    ai = np.sort(ai_idx)
    av = np.sort(av_idx)

    # Fit top-30 on training normal only
    top_ids = fit_top_ids(n_ids[ni])
    with open(DATA_DIR / "top_can_ids.json", "w") as f:
        json.dump(top_ids, f)
    print(f"Saved top_can_ids.json")

    # Build 40-dim features
    n_feats = build_features(n_ids, top_ids, n_bytes, n_dlc)
    a_feats = build_features(a_ids, top_ids, a_bytes, a_dlc)

    splits = {
        "train": (n_feats[ni], n_ids[ni], a_feats[ai], a_ids[ai], a_types[ai]),
        "val":   (n_feats[nv], n_ids[nv], a_feats[av], a_ids[av], a_types[av]),
    }

    print(f"\nNormal: {len(n_ids):,} | Attack: {len(a_ids):,}")
    for name in ("train", "val"):
        nf, _, af, _, _ = splits[name]
        print(f"  {name:5s}: {len(nf):,} / {len(af):,}")

    return splits


# ── Threshold (validation normal data) ──────

@torch.no_grad()
def compute_threshold(model, feats, ids, device, percentile=99):
    model.eval()
    tensor = torch.tensor(feats, dtype=torch.float32).to(device)
    all_emb = []
    for i in tqdm(range(0, len(tensor), 256), desc="Threshold embed", leave=False):
        all_emb.append(model.shared_dnn(tensor[i:i + 256]).cpu().numpy())
    emb = np.vstack(all_emb)

    ids = np.array(ids, dtype=str)
    emb_mean = emb.mean(axis=0)
    dists = np.zeros(len(emb))
    for cid in tqdm(np.unique(ids), desc="Threshold centroids", leave=False):
        mask = ids == cid
        centroid = emb[mask].mean(axis=0)
        diff = emb[mask] - centroid[None, :]
        dists[mask] = np.sum(diff * diff, axis=1)
    # IDs with zero dist were not assigned a centroid (shouldn't happen here)
    unassigned = dists == 0
    if unassigned.any():
        diff = emb[unassigned] - emb_mean[None, :]
        dists[unassigned] = np.sum(diff * diff, axis=1)

    threshold = float(np.percentile(dists, percentile))
    print(f"Threshold ({percentile}th pct): {threshold:.6f}")
    return threshold


# ── Train / eval one epoch ───────────────────

def run_epoch(model, loader, device, optimizer=None, desc=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    pbar = tqdm(loader, desc=desc or ("train" if is_train else "val"), leave=False)
    with ctx:
        for a, p, n in pbar:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ea, ep, en = model(a, p, n)
            loss = model.triplet_loss(ea, ep, en)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    return np.mean(losses)


def save_checkpoint(path, model, optimizer, epoch, loss, **extra):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **extra,
    }, path)


# ── Main ─────────────────────────────────────

def train(
    epochs=5,
    batch_size=256,
    lr=0.001,
    margin=1.0,
    embedding_dim=16,
    hidden_dims=[16, 32],
    val_ratio=0.15,
    patience=10,
    seed=42,
    threshold_percentile=99,
    device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = prepare_data(val_ratio)

    input_dim = 40
    loader_kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(TripletDataset(*splits["train"][:4]), shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(TripletDataset(*splits["val"][:4]), **loader_kw)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = SiameseNetwork(
        input_dim=input_dim, embedding_dim=embedding_dim, margin=margin, hidden_dims=hidden_dims
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n{'=' * 60}\nTraining for {epochs} epochs (patience={patience})\n{'=' * 60}")

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, desc=f"E{epoch} train")
        val_loss = run_epoch(model, val_loader, device, desc=f"E{epoch} val")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            marker = " *"
            save_checkpoint(
                MODEL_DIR / "best_model.pth", model, optimizer, epoch, best_val_loss,
                embedding_dim=embedding_dim, hidden_dims=hidden_dims, margin=margin,
                seed=seed, threshold=0.0, input_dim=input_dim,
            )
        else:
            stale += 1

        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}{marker}")

        if stale >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Reload best, compute threshold on val normal data (1.6M, not 9M), save final
    best_ckpt = torch.load(MODEL_DIR / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("\nComputing threshold on validation normal data ...")
    threshold = compute_threshold(model, splits["val"][0], splits["val"][1], device, threshold_percentile)

    for tag, ep, ls in [("final_model.pth", epoch, train_loss), ("best_model.pth", best_ckpt["epoch"], best_val_loss)]:
        save_checkpoint(
            MODEL_DIR / tag, model, optimizer, ep, ls,
            embedding_dim=embedding_dim, hidden_dims=hidden_dims, margin=margin,
            seed=seed, threshold=threshold, input_dim=input_dim,
        )

    pd.DataFrame(history).to_csv(DATA_DIR / "training_history.csv", index=False)
    print(f"\nDone! Best val loss: {best_val_loss:.6f} | Threshold: {threshold:.6f}")


if __name__ == "__main__":
    train()
