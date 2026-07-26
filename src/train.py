import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.model import SiameseNetwork

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
MODEL_DIR.mkdir(exist_ok=True)

TOP_K = 30
META_COLS = {"CAN_ID", "Timestamp"}
BYTE_COLS = [f"byte_{i}" for i in range(8)]


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ── Fit / Transform ──────────────────────────

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
    f[:, TOP_K + 1 : TOP_K + 9] = byte_arr  # already /255 from features.py
    f[:, TOP_K + 9] = dlc_arr                # already /8 from features.py
    return f


# ── Dataset ──────────────────────────────────

class TripletDataset(Dataset):
    """On-the-fly triplet generation with same-ID preference."""

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

def load_frames(csv_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    return (
        df["CAN_ID"].astype(str).values,
        df["Timestamp"].values,
        df[BYTE_COLS].values.astype(np.float32),
        df["DLC"].values.astype(np.float32),
    )


def chronological_split(n, val_ratio=0.15, test_ratio=0.15):
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    return np.arange(n_train), np.arange(n_train, n_train + n_val), np.arange(n_train + n_val, n)


def prepare_data(val_ratio, test_ratio):
    """Load per-frame CSVs, sort, split, fit top-30, build features."""
    # Load
    n_ids, n_ts, n_bytes, n_dlc = load_frames("Attack_free_frames.csv")
    a_ids, a_ts, a_bytes, a_dlc = load_frames("DoS_attack_frames.csv")
    for name in ["Fuzzy_attack_frames.csv", "Impersonation_attack_frames.csv"]:
        ids, ts, by, dl = load_frames(name)
        a_ids = np.concatenate([a_ids, ids])
        a_ts = np.concatenate([a_ts, ts])
        a_bytes = np.vstack([a_bytes, by])
        a_dlc = np.concatenate([a_dlc, dl])

    # Sort chronologically
    n_order = n_ts.argsort()
    a_order = a_ts.argsort()
    n_ids, n_ts, n_bytes, n_dlc = n_ids[n_order], n_ts[n_order], n_bytes[n_order], n_dlc[n_order]
    a_ids, a_ts, a_bytes, a_dlc = a_ids[a_order], a_ts[a_order], a_bytes[a_order], a_dlc[a_order]

    # Split
    ni, nv, nt = chronological_split(len(n_ids), val_ratio, test_ratio)
    ai, av, at = chronological_split(len(a_ids), val_ratio, test_ratio)

    # Fit top-30 on training normal only
    top_ids = fit_top_ids(n_ids[ni])
    with open(DATA_DIR / "top_can_ids.json", "w") as f:
        json.dump(top_ids, f)
    print(f"Saved top_can_ids.json")

    # Build 40-dim features
    n_feats = build_features(n_ids, top_ids, n_bytes, n_dlc)
    a_feats = build_features(a_ids, top_ids, a_bytes, a_dlc)

    splits = {
        "train": (n_feats[ni], n_ids[ni], a_feats[ai], a_ids[ai]),
        "val":   (n_feats[nv], n_ids[nv], a_feats[av], a_ids[av]),
        "test":  (n_feats[nt], n_ids[nt], a_feats[at], a_ids[at]),
    }

    print(f"\nNormal: {len(n_ids):,} | Attack: {len(a_ids):,}")
    for name in ("train", "val", "test"):
        nf, _, af, _ = splits[name]
        print(f"  {name:5s}: {len(nf):,} / {len(af):,}")

    # Subsample training data for fast training
    MAX_TRAIN = 150_000
    nf, ni, af, ai = splits["train"]
    if len(nf) > MAX_TRAIN:
        idx = np.random.choice(len(nf), MAX_TRAIN, replace=False)
        nf, ni = nf[idx], ni[idx]
    if len(af) > MAX_TRAIN:
        idx = np.random.choice(len(af), MAX_TRAIN, replace=False)
        af, ai = af[idx], ai[idx]
    splits["train"] = (nf, ni, af, ai)
    print(f"Train subsampled to: {len(nf):,} / {len(af):,}")

    return splits


def save_test_split(splits):
    nf, ni, af, ai = splits["test"]
    np.savez(DATA_DIR / "test_split.npz",
             normal_test_feats=nf, normal_test_ids=ni,
             attack_test_feats=af, attack_test_ids=ai)
    print("Saved test_split.npz")


# ── Threshold (training data only) ───────────

@torch.no_grad()
def compute_threshold(model, feats, ids, device, percentile=99):
    model.eval()
    tensor = torch.tensor(feats, dtype=torch.float32).to(device)
    all_emb = []
    for i in range(0, len(tensor), 256):
        all_emb.append(model.shared_dnn(tensor[i:i + 256]).cpu().numpy())
    emb = np.vstack(all_emb)

    ids = np.array(ids, dtype=str)
    centroids = {}
    for cid in np.unique(ids):
        centroids[cid] = emb[ids == cid].mean(axis=0)

    dists = np.array([
        np.sum((e - centroids.get(c, emb.mean(axis=0))) ** 2)
        for e, c in zip(emb, ids)
    ])
    threshold = float(np.percentile(dists, percentile))
    print(f"Threshold ({percentile}th pct): {threshold:.6f}")
    return threshold


# ── Train / eval one epoch ───────────────────

def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ea, ep, en = model(a, p, n)
            loss = model.triplet_loss(ea, ep, en)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
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
    epochs=50,
    batch_size=64,
    lr=0.001,
    margin=1.0,
    embedding_dim=16,
    hidden_dims=[16, 32],
    val_ratio=0.15,
    test_ratio=0.15,
    patience=10,
    seed=42,
    threshold_percentile=99,
    device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = prepare_data(val_ratio, test_ratio)
    save_test_split(splits)

    input_dim = 40
    train_loader = DataLoader(TripletDataset(*splits["train"]), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TripletDataset(*splits["val"]), batch_size=batch_size)
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
    threshold = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            marker = " *"
            threshold = compute_threshold(model, splits["train"][0], splits["train"][1], device, threshold_percentile)
            save_checkpoint(
                MODEL_DIR / "best_model.pth", model, optimizer, epoch, best_val_loss,
                embedding_dim=embedding_dim, hidden_dims=hidden_dims, margin=margin,
                seed=seed, threshold=threshold, input_dim=input_dim,
            )
        else:
            stale += 1

        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}{marker}")

        if stale >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Reload best, recompute threshold, save final
    best_ckpt = torch.load(MODEL_DIR / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    threshold = compute_threshold(model, splits["train"][0], splits["train"][1], device, threshold_percentile)

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
