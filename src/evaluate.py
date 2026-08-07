import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve,
)

from src.model import Autoencoder, LSTMAutoencoder
from src.features import (
    load_id_stats, fit_id_stats, build_features, compute_byte_delta_features, BYTE_COLS,
)
from src.train import can_ids_to_indices


DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
FIG_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\figures")
FIG_DIR.mkdir(exist_ok=True)

INPUT_DIM = 15
LSTM_CKPT = "best_model_lstm.pth"


def _lstm_checkpoints():
    """LSTM checkpoints in MODEL_DIR (excludes intermediate *_weights files)."""
    return sorted(
        c.name for c in MODEL_DIR.glob("best_model_lstm*.pth") if "_weights" not in c.name
    )


def get_stats():
    """Load id_stats.json if present, else recompute from the local train CSV."""
    try:
        return load_id_stats()
    except (FileNotFoundError, KeyError):
        csv = DATA_DIR / "set01_train_frames.csv"
        print(f"id_stats.json missing -> recomputing stats from {csv.name} ...")
        n_df = pd.read_csv(csv, dtype={"CAN_ID": str})
        n_df = n_df.loc[n_df["attack"].values == 0]
        if "Session" in n_df.columns:
            n_df = n_df.sort_values(["Session", "Timestamp"])
        else:
            n_df = n_df.sort_values("Timestamp")
        return fit_id_stats(n_df)


def load_model(checkpoint, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_DIR / checkpoint, map_location=device, weights_only=False)
    threshold = ckpt.get("threshold", None)

    with open(DATA_DIR / "top_can_ids.json") as f:
        top_ids = json.load(f)

    if ckpt.get("arch") == "lstm":
        model = LSTMAutoencoder(
            input_dim=ckpt.get("input_dim", INPUT_DIM),
            hidden=ckpt.get("hidden", 64),
            num_layers=ckpt.get("num_layers", 1),
        ).to(device)
        meta = {"type": "lstm", "window": int(ckpt.get("window", 16))}
    else:
        num_ids = len(top_ids) + 1
        model = Autoencoder(num_ids=num_ids, input_dim=ckpt.get("input_dim", INPUT_DIM)).to(device)
        meta = {"type": "ae"}

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded {meta['type'].upper()} from {checkpoint} (epoch {ckpt['epoch']}, val_loss={ckpt['loss']:.6f})")
    if threshold is not None:
        print(f"  Threshold: {threshold:.6f}" + (f"  (best F1 {ckpt.get('best_f1', float('nan')):.4f} @ {ckpt.get('best_pct', float('nan'))}th pct)" if ckpt.get("best_f1") else ""))
    else:
        print("  WARNING: No threshold found in checkpoint!")
    return model, device, threshold, top_ids, meta


def get_reconstruction_errors(model, id_indices, features, device, batch_size=256):
    model.eval()
    all_errs = []
    with torch.no_grad():
        for i in range(0, len(id_indices), batch_size):
            idx_batch = torch.tensor(id_indices[i:i+batch_size], dtype=torch.long).to(device)
            feats_batch = torch.tensor(features[i:i+batch_size], dtype=torch.float32).to(device)
            err = model.get_reconstruction_error(idx_batch, feats_batch)
            all_errs.append(err.cpu().numpy())
    return np.concatenate(all_errs)


def get_window_errors(model, windows, device, batch_size=256):
    """Per-window reconstruction errors for the LSTM."""
    model.eval()
    all_errs = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            w = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            err = model.get_reconstruction_error(w)
            all_errs.append(err.cpu().numpy())
    return np.concatenate(all_errs)


def detect(model, n_id_idx, n_feats, a_id_idx, a_feats, a_types, device, threshold):
    normal_errs = get_reconstruction_errors(model, n_id_idx, n_feats, device)
    attack_errs = get_reconstruction_errors(model, a_id_idx, a_feats, device)

    print(f"Threshold: {threshold:.6f}")
    print(f"  Normal reconstruction error: mean={normal_errs.mean():.6f} std={normal_errs.std():.6f}")
    print(f"  Attack reconstruction error: mean={attack_errs.mean():.6f} std={attack_errs.std():.6f}")

    normal_preds = (normal_errs > threshold).astype(int)
    attack_preds = (attack_errs > threshold).astype(int)

    y_true = np.concatenate([np.zeros(len(normal_errs)), np.ones(len(attack_errs))])
    y_pred = np.concatenate([normal_preds, attack_preds])
    y_scores = np.concatenate([normal_errs, attack_errs])

    return y_true, y_pred, y_scores, normal_errs, attack_errs, a_types


def detect_windows(model, norm_windows, att_windows, att_types, device, threshold):
    norm_errs = get_window_errors(model, norm_windows, device)
    att_errs = get_window_errors(model, att_windows, device)

    print(f"Threshold: {threshold:.6f}")
    print(f"  Normal window error: mean={norm_errs.mean():.6f} std={norm_errs.std():.6f}")
    print(f"  Attack window error: mean={att_errs.mean():.6f} std={att_errs.std():.6f}")

    y_true = np.concatenate([np.zeros(len(norm_errs)), np.ones(len(att_errs))])
    y_pred = np.concatenate([(norm_errs > threshold).astype(int), (att_errs > threshold).astype(int)])
    y_scores = np.concatenate([norm_errs, att_errs])
    return y_true, y_pred, y_scores, norm_errs, att_errs, att_types


def report_per_type(norm_errs, att_errs, att_types, threshold):
    """PR-AUC / F1 / recall per attack type (each type's attacks vs all normals)."""
    print(f"\n  {'Type':<22s} {'PR-AUC':>8s} {'F1':>8s} {'Recall':>8s} {'N':>8s}")
    print(f"  {'-'*56}")
    for atype in np.unique(att_types):
        mask = att_types == atype
        n = int(mask.sum())
        at_true = np.concatenate([np.zeros(len(norm_errs)), np.ones(n)])
        at_scores = np.concatenate([norm_errs, att_errs[mask]])
        at_preds = np.concatenate([
            (norm_errs > threshold).astype(int),
            (att_errs > threshold).astype(int)[mask],
        ])
        at_pr_auc = average_precision_score(at_true, at_scores) if n else 0.0
        at_f1 = f1_score(at_true, at_preds) if n else 0.0
        at_rec = recall_score(at_true, at_preds) if n else 0.0
        print(f"  {atype:<22s} {at_pr_auc:>8.4f} {at_f1:>8.4f} {at_rec:>8.4f} {n:>8,}")


def window_stream(feats, att_mask, sessions, window, a_types=None):
    """Non-overlapping windows, computed per-session so a window NEVER spans two
    independent recordings (each file shares the same epoch base timestamp, so a
    global sort interleaves sessions). A window is 'attack' if it contains >= 1
    attack frame. When a_types (per-frame) is given, also returns the attack type
    of the FIRST attack frame for each window ('' for normal windows)."""
    fw_list, aw_list, tw_list = [], [], []
    for sess in np.unique(sessions):
        m = sessions == sess
        f, a = feats[m], att_mask[m]
        n = (len(f) // window) * window
        if n == 0:
            continue
        fw_list.append(f[:n].reshape(-1, window, f.shape[1]))
        a_w = a[:n].reshape(-1, window).any(axis=1)
        aw_list.append(a_w)
        if a_types is not None:
            at = a_types[m][:n].reshape(-1, window)
            first_att = np.argmax(at != "normal", axis=1)
            tw = np.take_along_axis(at, first_att[:, None], axis=1)[:, 0]
            tw_list.append(np.where(a_w, tw, ""))
    if not fw_list:
        empty = np.empty((0, window, feats.shape[1]), dtype=np.float32)
        if a_types is not None:
            return empty, np.empty(0, dtype=bool), np.empty(0, dtype=object)
        return empty, np.empty(0, dtype=bool)
    fw = np.concatenate(fw_list)
    aw = np.concatenate(aw_list)
    if a_types is not None:
        return fw, aw, np.concatenate(tw_list)
    return fw, aw


# ──────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────

def plot_loss_curve():
    history = pd.read_csv(DATA_DIR / "training_history.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], linewidth=2, label="Train")
    plt.plot(history["epoch"], history["val_loss"], linewidth=2, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loss_curve.png", dpi=150)
    plt.close()
    print("Saved: figures/loss_curve.png")


def plot_error_distribution(normal_errs, attack_errs, threshold, tag=""):
    plt.figure(figsize=(10, 5))
    plt.hist(normal_errs, bins=80, alpha=0.7, label="Normal", color="green", density=True)
    plt.hist(attack_errs, bins=80, alpha=0.7, label="Attack", color="red", density=True)
    plt.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.6f}")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Density")
    plt.title(f"Reconstruction Error: Normal vs Attack ({tag})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"error_distribution_{tag}.png" if tag else "error_distribution.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved: figures/{fname}")


def plot_pr_curve(y_true, y_scores, pr_auc, tag=""):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({tag})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"pr_curve_{tag}.png" if tag else "pr_curve.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved: figures/{fname}")


def plot_confusion_matrix(y_true, y_pred, tag=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    fname = f"confusion_matrix_{tag}.png" if tag else "confusion_matrix.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved: figures/{fname}")


# ──────────────────────────────────────────────
#  Main evaluation
# ──────────────────────────────────────────────

def build_full_features(df, stats, global_avg, top_ids):
    """15-dim features + id indices over the FULL stream.

    Frames are grouped by Session (each source file is an independent recording
    sharing the same epoch base), so temporal features never span two sessions.
    """
    if "Session" in df.columns:
        df = df.sort_values(["Session", "Timestamp"]).reset_index(drop=True)
    else:
        df = df.sort_values("Timestamp").reset_index(drop=True)
    ids = df["CAN_ID"].astype(str).values
    payload = np.hstack([
        df[BYTE_COLS].values.astype(np.float32),
        df["DLC"].values.astype(np.float32).reshape(-1, 1),
    ]).astype(np.float32)
    feats = build_features(ids, payload, stats, global_avg)
    delta = compute_byte_delta_features(df)
    feats = np.hstack([feats, delta]).astype(np.float32)
    id_idx = can_ids_to_indices(ids, top_ids)
    att_mask = df["attack"].values == 1
    a_types = df.loc[att_mask, "attack_type"].values.astype(str)
    a_types_full = df["attack_type"].astype(str).values
    if "Session" in df.columns:
        sessions = df["Session"].astype(str).values
    else:
        sessions = np.zeros(len(df), dtype=object)
    return feats, id_idx, att_mask, a_types, sessions, a_types_full


def evaluate(checkpoints=None, models=None):
    """Evaluate models on all test sets.

    models: list of 'lstm' and/or 'ae' (default: ['lstm'] - run ONLY the LSTM).
    """
    models = models or ["lstm"]
    checkpoints = list(checkpoints or [])
    if "lstm" in models:
        checkpoints += _lstm_checkpoints()
    if "ae" in models and (MODEL_DIR / AE_CKPT).exists():
        checkpoints.append(AE_CKPT)
    checkpoints = sorted(set(checkpoints))
    if not checkpoints:
        raise FileNotFoundError(
            f"No LSTM checkpoint found in {MODEL_DIR} (glob best_model_lstm*.pth)."
        )

    stats, global_avg = get_stats()
    with open(DATA_DIR / "top_can_ids.json") as f:
        top_ids = json.load(f)

    test_csvs = sorted(DATA_DIR.glob("set_01_test_*_frames.csv"))
    if not test_csvs:
        raise FileNotFoundError("No test CSVs found in data/. Run parse_can_train_test.py first.")

    summary = {}
    for csv_path in test_csvs:
        test_name = csv_path.stem.replace("_frames", "")
        print(f"\n{'='*64}")
        print(f"Test set: {test_name}")
        print(f"{'='*64}")
        feats, id_idx, att_mask, atypes, sessions, a_types_full = build_full_features(
            pd.read_csv(csv_path), stats, global_avg, top_ids
        )
        n_id_idx, n_feats = id_idx[~att_mask], feats[~att_mask]
        a_id_idx, a_feats = id_idx[att_mask], feats[att_mask]
        print(f"  Frames: normal={len(n_feats):,} | attack={len(a_feats):,} | Types: {np.unique(atypes)}")

        for ckpt in checkpoints:
            model, device, threshold, _top_ids, meta = load_model(ckpt)
            tag = f"{test_name}_{meta['type']}"
            if threshold is None:
                print(f"  {meta['type'].upper()}: SKIPPED (no threshold)")
                continue

            print(f"\n  [{meta['type'].upper()}] Overall:")
            if meta["type"] == "lstm":
                window = meta["window"]
                fw, aw, aw_types = window_stream(feats, att_mask, sessions, window, a_types_full)
                norm_w, att_w = fw[~aw], fw[aw]
                y_true, y_pred, y_scores, ne, ae, att_types = detect_windows(
                    model, norm_w, att_w, aw_types[aw], device, threshold
                )
            else:
                y_true, y_pred, y_scores, ne, ae, att_types = detect(
                    model, n_id_idx, n_feats, a_id_idx, a_feats, atypes, device, threshold
                )

            report_per_type(ne, ae, att_types, threshold)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_scores)

            print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
            print(f"  F1: {f1:.4f} | PR-AUC: {pr_auc:.4f}  (primary)")

            plot_error_distribution(ne, ae, threshold, tag=tag)
            plot_pr_curve(y_true, y_scores, pr_auc, tag=tag)
            plot_confusion_matrix(y_true, y_pred, tag=tag)

            summary[f"{test_name} | {meta['type']}"] = {
                "accuracy": acc, "precision": prec, "recall": rec,
                "f1": f1, "pr_auc": pr_auc,
            }

    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    print(f"{'Test Set | Model':<46s} {'PR-AUC':>8s} {'F1':>8s}")
    print(f"{'-'*64}")
    for name, res in sorted(summary.items()):
        print(f"{name:<46s} {res['pr_auc']:>8.4f} {res['f1']:>8.4f}")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate CAN anomaly models (LSTM by default)")
    ap.add_argument("--models", default="lstm", choices=["lstm", "ae", "all"],
                    help="which models to evaluate (default: lstm only)")
    args = ap.parse_args()
    evaluate(models=["lstm", "ae"] if args.models == "all" else [args.models])
