import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.model import LSTMAutoencoder
from src.features import (
    fit_id_stats, save_id_stats, build_features, ByteDeltaRoller, BYTE_COLS,
)

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
MODEL_DIR.mkdir(exist_ok=True)

TOP_K = 64
INPUT_DIM = 15
TRAIN_FRAC = 1.0

WINDOW = 16
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_EPOCHS = 50
LSTM_LR = 1e-3
LSTM_PATIENCE = 10


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


# ── Window dataset ───────────────────────────

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def make_windows(feats, sessions, window):
    """Non-overlapping windows, built per-session so a window NEVER spans two
    independent recordings (each file shares the same epoch base timestamp, so a
    global slice would stitch unrelated sessions together)."""
    parts = []
    for s in np.unique(sessions):
        f = feats[sessions == s]
        n = (len(f) // window) * window
        if n:
            parts.append(f[:n].reshape(-1, window, f.shape[1]))
    return np.concatenate(parts)


# ── Data loading ─────────────────────────────

def prepare_data(val_ratio=0.2, train_frac=TRAIN_FRAC, window=WINDOW):
    npz_path = DATA_DIR / "ae_data.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as d:
            if d.get("version") == 3 and "train_feats" in d and "train_sess" in d:
                print("Loading ae_data.npz ...")
                train_w = make_windows(d["train_feats"], d["train_sess"], window)
                val_w = make_windows(d["val_feats"], d["val_sess"], window)
                top_ids = d["top_ids"].tolist()
                print(f"  train windows: {len(train_w):,}  val windows: {len(val_w):,}  "
                      f"(W={window}, F={d['train_feats'].shape[1]})")
                return (train_w, val_w), top_ids

    csv_path = DATA_DIR / "set01_train_frames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing. Run parse_can_train_test.py first.")

    # Byte-delta features must be computed per-session over the FULL stream (attack
    # frames included) so a replayed payload shows its freeze signal. The merged CSV
    # is grouped by Session and chronological within each session; the roller carries
    # the last payload per (session, id) across chunk boundaries.
    roller = ByteDeltaRoller()
    parts = []
    for ch in pd.read_csv(csv_path, chunksize=2_000_000, dtype={"CAN_ID": str}):
        if "Session" in ch.columns:
            ch = ch.sort_values(["Session", "Timestamp"]).reset_index(drop=True)
        else:
            ch = ch.sort_values("Timestamp").reset_index(drop=True)
        delta = roller.advance(ch)
        m = ch["attack"].values == 0
        keep = ch.loc[m].copy()
        keep["_delta0"] = delta[m, 0]
        keep["_delta1"] = delta[m, 1]
        if train_frac < 1.0:
            keep = keep.sample(frac=train_frac, random_state=42)
        parts.append(keep)
    n_df = pd.concat(parts, ignore_index=True)

    n_ids = n_df["CAN_ID"].astype(str).values
    n_bytes = n_df[BYTE_COLS].values.astype(np.float32)
    n_dlc = n_df["DLC"].values.astype(np.float32)

    top_ids = fit_top_ids(n_ids)
    with open(DATA_DIR / "top_can_ids.json", "w") as f:
        json.dump(top_ids, f)

    stats, global_avg = fit_id_stats(n_df)
    save_id_stats(stats, global_avg)

    payload = np.hstack([n_bytes, n_dlc.reshape(-1, 1)]).astype(np.float32)
    features = build_features(n_ids, payload, stats, global_avg)
    temporal = n_df[["_delta0", "_delta1"]].values.astype(np.float32)
    features = np.hstack([features, temporal]).astype(np.float32)

    if "Session" in n_df.columns:
        sessions = n_df["Session"].astype(str).values
    else:
        sessions = np.zeros(len(n_df), dtype=object)

    # Per-session chronological split: the last VAL_RATIO of each session's frames
    # are validation (the sessions are independent recordings with overlapping
    # timestamps, so a global-timeline split is meaningless).
    val_mask = np.zeros(len(n_df), dtype=bool)
    if "Session" in n_df.columns:
        for gi in n_df.groupby("Session").indices.values():
            gi = np.asarray(gi)
            n_val = int(len(gi) * val_ratio)
            if n_val:
                val_mask[gi[-n_val:]] = True
    else:
        n_val = int(len(n_df) * val_ratio)
        val_mask[len(n_df) - n_val:] = True

    train_feats, val_feats = features[~val_mask], features[val_mask]
    train_sess, val_sess = sessions[~val_mask], sessions[val_mask]

    np.savez_compressed(
        npz_path,
        version=3,
        train_feats=train_feats, train_sess=train_sess,
        val_feats=val_feats, val_sess=val_sess,
        top_ids=top_ids,
    )

    train_w = make_windows(train_feats, train_sess, window)
    val_w = make_windows(val_feats, val_sess, window)
    print(f"Saved ae_data.npz (train windows: {len(train_w):,}, val windows: {len(val_w):,}, "
          f"feat_dim={INPUT_DIM}, W={window})")
    return (train_w, val_w), top_ids


# ── Threshold (window reconstruction error on val normal) ──

@torch.no_grad()
def compute_threshold(model, loader, device, percentile=99):
    model.eval()
    all_errs = []
    for x in tqdm(loader, desc="Threshold", leave=False):
        x = x.to(device)
        err = model.get_reconstruction_error(x)
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
        for x in pbar:
            x = x.to(device)
            recon = model(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    return np.mean(losses)


# ── Main ─────────────────────────────────────

def train(
    epochs=LSTM_EPOCHS,
    batch_size=256,
    lr=LSTM_LR,
    val_ratio=0.2,
    patience=LSTM_PATIENCE,
    seed=42,
    threshold_percentile=99,
    train_frac=TRAIN_FRAC,
    window=WINDOW,
    device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (train_w, val_w), top_ids = prepare_data(val_ratio, train_frac, window)
    print(f"Windows: train={len(train_w):,} val={len(val_w):,} (W={window}, F={INPUT_DIM})")

    loader_kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(WindowDataset(train_w), shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(WindowDataset(val_w), **loader_kw)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = LSTMAutoencoder(input_dim=INPUT_DIM, hidden=LSTM_HIDDEN, num_layers=LSTM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n{'=' * 60}\nTraining LSTM AE for {epochs} epochs (patience={patience})\n{'=' * 60}")

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    stale = 0
    best_epoch = 0
    weights_path = MODEL_DIR / "best_model_lstm_weights.pth"

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, desc=f"LSTM E{epoch} train")
        val_loss = run_epoch(model, val_loader, device, desc=f"LSTM E{epoch} val")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, stale = val_loss, epoch, 0
            marker = " *"
            torch.save(model.state_dict(), weights_path)
        else:
            stale += 1

        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}{marker}")

        if stale >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\nReloading best weights + computing threshold on validation windows ...")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    threshold = compute_threshold(model, val_loader, device, threshold_percentile)

    torch.save({
        "arch": "lstm",
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": best_val_loss,
        "threshold": threshold,
        "input_dim": INPUT_DIM,
        "hidden": LSTM_HIDDEN,
        "num_layers": LSTM_LAYERS,
        "window": WINDOW,
    }, MODEL_DIR / "best_model_lstm.pth")

    pd.DataFrame(history).to_csv(DATA_DIR / "training_history.csv", index=False)
    print(f"\nDone! Best val loss: {best_val_loss:.6f} | Threshold: {threshold:.6f}")


if __name__ == "__main__":
    train()
