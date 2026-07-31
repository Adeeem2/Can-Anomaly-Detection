import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.model import Autoencoder

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
    counts = pd.Series(can_ids).value_counts()
    top = counts.head(k).index.tolist()
    print(f"Top-{k} IDs: {top[:5]} ... ({len(top)} total)")
    return top


def chronological_split(n, val_ratio=0.2):
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n)
    return train_idx, val_idx


def can_ids_to_indices(can_ids, top_ids):
    id_map = {cid: i for i, cid in enumerate(top_ids)}
    return np.array([id_map.get(cid, TOP_K) for cid in can_ids], dtype=np.int32)


# ── Dataset ──────────────────────────────────

class AEMDataset(Dataset):
    def __init__(self, id_indices, payload):
        self.id_indices = torch.tensor(id_indices, dtype=torch.long)
        self.payload = torch.tensor(payload, dtype=torch.float32)

    def __len__(self):
        return len(self.id_indices)

    def __getitem__(self, idx):
        return self.id_indices[idx], self.payload[idx]


# ── Data loading ─────────────────────────────

def prepare_data(val_ratio=0.2):
    npz_path = DATA_DIR / "ae_data.npz"
    if npz_path.exists():
        print("Loading ae_data.npz ...")
        d = np.load(npz_path, allow_pickle=True)
        splits = {
            "train": (d["train_id_idx"], d["train_payload"]),
            "val":   (d["val_id_idx"], d["val_payload"]),
        }
        top_ids = d["top_ids"].tolist()
        print(f"  train: {len(d['train_id_idx']):,}  val: {len(d['val_id_idx']):,}")
        return splits, top_ids

    df = pd.read_csv(DATA_DIR / "set01_train_frames.csv")

    n_mask = df["attack"].values == 0
    n_ids = df.loc[n_mask, "CAN_ID"].astype(str).values
    n_ts = df.loc[n_mask, "Timestamp"].values
    n_bytes = df.loc[n_mask, BYTE_COLS].values.astype(np.float32)
    n_dlc = df.loc[n_mask, "DLC"].values.astype(np.float32)

    order = n_ts.argsort()
    n_ids, n_bytes, n_dlc = n_ids[order], n_bytes[order], n_dlc[order]

    top_ids = fit_top_ids(n_ids)
    with open(DATA_DIR / "top_can_ids.json", "w") as f:
        json.dump(top_ids, f)

    id_indices = can_ids_to_indices(n_ids, top_ids)
    payload = np.hstack([n_bytes, n_dlc.reshape(-1, 1)]).astype(np.float32)

    ni, nv = chronological_split(len(n_ids), val_ratio)

    np.savez_compressed(
        npz_path,
        train_id_idx=id_indices[ni], train_payload=payload[ni],
        val_id_idx=id_indices[nv], val_payload=payload[nv],
        top_ids=top_ids,
    )
    print(f"Saved ae_data.npz")

    splits = {
        "train": (id_indices[ni], payload[ni]),
        "val":   (id_indices[nv], payload[nv]),
    }
    return splits, top_ids


# ── Threshold (reconstruction error on val normal) ──

@torch.no_grad()
def compute_threshold(model, loader, device, percentile=99):
    model.eval()
    all_errs = []
    for id_idx, payload in tqdm(loader, desc="Threshold", leave=False):
        id_idx, payload = id_idx.to(device), payload.to(device)
        recon = model(id_idx, payload)
        x = torch.cat([model.id_embedding(id_idx), payload], dim=1)
        mse = (recon - x).pow(2).mean(dim=1)
        all_errs.append(mse.cpu().numpy())
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
        for id_idx, payload in pbar:
            id_idx, payload = id_idx.to(device), payload.to(device)
            recon = model(id_idx, payload)
            x = torch.cat([model.id_embedding(id_idx), payload], dim=1)
            loss = torch.nn.functional.mse_loss(recon, x)
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
    device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits, top_ids = prepare_data(val_ratio)
    top_ids = list(top_ids)

    NUM_WORKERS = 2
    loader_kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(AEMDataset(*splits["train"]), shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(AEMDataset(*splits["val"]), **loader_kw)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = Autoencoder(num_ids=len(top_ids) + 1).to(device)
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
