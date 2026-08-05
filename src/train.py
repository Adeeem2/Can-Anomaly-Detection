import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.model import Autoencoder
from src.features import (
    fit_id_stats, save_id_stats, build_features, compute_temporal_features, BYTE_COLS,
)

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
MODEL_DIR.mkdir(exist_ok=True)

TOP_K = 64
INPUT_DIM = 15
TRAIN_FRAC = 1.0


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def fit_top_ids(can_ids, k=TOP_K):
    counts = pd.Series(can_ids).value_counts()
    top = counts.head(k).index.tolist()
    print(f"Top-{k} IDs: {top[:5]} ... ({len(top)} total)")
    return top


def can_ids_to_indices(can_ids, top_ids):
    id_map = {cid: i for i, cid in enumerate(top_ids)}
    return np.array([id_map.get(cid, len(top_ids)) for cid in can_ids], dtype=np.int32)


def chronological_split(n, val_ratio=0.2):
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n)
    return train_idx, val_idx


# ── Dataset ──────────────────────────────────

class AEMDataset(Dataset):
    def __init__(self, id_indices, features):
        self.id_indices = torch.tensor(id_indices, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.id_indices)

    def __getitem__(self, idx):
        return self.id_indices[idx], self.features[idx]


# ── Data loading ─────────────────────────────

def prepare_data(val_ratio=0.2, train_frac=TRAIN_FRAC):
    npz_path = DATA_DIR / "ae_data.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as d:
            if "train_feats" in d and "train_id_idx" in d:
                print("Loading ae_data.npz ...")
                splits = {
                    "train": (d["train_id_idx"], d["train_feats"]),
                    "val":   (d["val_id_idx"], d["val_feats"]),
                }
                top_ids = d["top_ids"].tolist()
                print(f"  train: {len(d['train_id_idx']):,}  val: {len(d['val_id_idx']):,}  feat_dim={d['train_feats'].shape[1]}")
                return splits, top_ids

    csv_path = DATA_DIR / "set01_train_frames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing. Run parse_can_train_test.py first.")

    chunks = []
    for ch in pd.read_csv(csv_path, chunksize=2_000_000, dtype={"CAN_ID": str}):
        ch = ch.loc[ch["attack"].values == 0]
        if train_frac < 1.0:
            ch = ch.sample(frac=train_frac, random_state=42)
        chunks.append(ch)
    n_df = pd.concat(chunks, ignore_index=True)

    n_ids = n_df["CAN_ID"].astype(str).values
    n_ts = n_df["Timestamp"].values
    n_bytes = n_df[BYTE_COLS].values.astype(np.float32)
    n_dlc = n_df["DLC"].values.astype(np.float32)

    order = n_ts.argsort()
    n_ids, n_bytes, n_dlc = n_ids[order], n_bytes[order], n_dlc[order]
    n_df = n_df.iloc[order].reset_index(drop=True)

    top_ids = fit_top_ids(n_ids)
    with open(DATA_DIR / "top_can_ids.json", "w") as f:
        json.dump(top_ids, f)

    stats, global_avg = fit_id_stats(n_df)
    save_id_stats(stats, global_avg)

    payload = np.hstack([n_bytes, n_dlc.reshape(-1, 1)]).astype(np.float32)
    features = build_features(n_ids, payload, stats, global_avg)
    temporal = compute_temporal_features(n_df, stats, global_avg)
    features = np.hstack([features, temporal]).astype(np.float32)
    id_indices = can_ids_to_indices(n_ids, top_ids)

    ni, nv = chronological_split(len(features), val_ratio)

    np.savez_compressed(
        npz_path,
        train_id_idx=id_indices[ni], train_feats=features[ni],
        val_id_idx=id_indices[nv], val_feats=features[nv],
        top_ids=top_ids,
    )
    print(f"Saved ae_data.npz (train={len(ni):,}, val={len(nv):,}, feat_dim={INPUT_DIM})")

    splits = {
        "train": (id_indices[ni], features[ni]),
        "val":   (id_indices[nv], features[nv]),
    }
    return splits, top_ids


# ── Threshold (reconstruction error on val normal) ──

@torch.no_grad()
def compute_threshold(model, loader, device, percentile=99):
    model.eval()
    all_errs = []
    for id_idx, feats in tqdm(loader, desc="Threshold", leave=False):
        id_idx, feats = id_idx.to(device), feats.to(device)
        err = model.get_reconstruction_error(id_idx, feats)
        all_errs.append(err.cpu().numpy())
    all_errs = np.concatenate(all_errs)
    threshold = float(np.percentile(all_errs, percentile))
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
        for id_idx, feats in pbar:
            id_idx, feats = id_idx.to(device), feats.to(device)
            recon = model(id_idx, feats)
            loss = torch.nn.functional.mse_loss(recon, feats)
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
    epochs=30,
    batch_size=256,
    lr=1e-3,
    val_ratio=0.2,
    patience=5,
    seed=42,
    threshold_percentile=99,
    train_frac=TRAIN_FRAC,
    device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits, top_ids = prepare_data(val_ratio, train_frac)
    top_ids = list(top_ids)

    NUM_WORKERS = 2
    loader_kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(AEMDataset(*splits["train"]), shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(AEMDataset(*splits["val"]), **loader_kw)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = Autoencoder(num_ids=len(top_ids) + 1, input_dim=INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
                threshold=0.0,
            )
        else:
            stale += 1

        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}{marker}")

        if stale >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    best_ckpt = torch.load(MODEL_DIR / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("\nComputing threshold on validation normal data ...")
    threshold = compute_threshold(model, val_loader, device, threshold_percentile)

    for tag, ep, ls in [("final_model.pth", epoch, train_loss), ("best_model.pth", best_ckpt["epoch"], best_val_loss)]:
        save_checkpoint(
            MODEL_DIR / tag, model, optimizer, ep, ls,
            threshold=threshold,
        )

    pd.DataFrame(history).to_csv(DATA_DIR / "training_history.csv", index=False)
    print(f"\nDone! Best val loss: {best_val_loss:.6f} | Threshold: {threshold:.6f}")


if __name__ == "__main__":
    train()
