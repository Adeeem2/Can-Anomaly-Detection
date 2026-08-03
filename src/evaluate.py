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

from src.model import Autoencoder
from src.features import load_id_stats, build_features, BYTE_COLS


DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
FIG_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\figures")
FIG_DIR.mkdir(exist_ok=True)

INPUT_DIM = 13


def load_model(checkpoint="best_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_DIR / checkpoint, map_location=device, weights_only=False)
    threshold = ckpt.get("threshold", None)

    model = Autoencoder(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded autoencoder from epoch {ckpt['epoch']}, val_loss={ckpt['loss']:.6f}")
    if threshold is not None:
        print(f"Threshold: {threshold:.6f}")
    else:
        print("WARNING: No threshold found in checkpoint!")
    return model, device, threshold


def get_reconstruction_errors(model, features, device, batch_size=256):
    model.eval()
    all_errs = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            feats_batch = torch.tensor(features[i:i+batch_size], dtype=torch.float32).to(device)
            err = model.get_reconstruction_error(feats_batch)
            all_errs.append(err.cpu().numpy())
    return np.concatenate(all_errs)


def detect(model, n_feats, a_feats, device, threshold):
    normal_errs = get_reconstruction_errors(model, n_feats, device)
    attack_errs = get_reconstruction_errors(model, a_feats, device)

    print(f"Threshold: {threshold:.6f}")
    print(f"  Normal reconstruction error: mean={normal_errs.mean():.6f} std={normal_errs.std():.6f}")
    print(f"  Attack reconstruction error: mean={attack_errs.mean():.6f} std={attack_errs.std():.6f}")

    normal_preds = (normal_errs > threshold).astype(int)
    attack_preds = (attack_errs > threshold).astype(int)

    y_true = np.concatenate([np.zeros(len(normal_errs)), np.ones(len(attack_errs))])
    y_pred = np.concatenate([normal_preds, attack_preds])
    y_scores = np.concatenate([normal_errs, attack_errs])

    return y_true, y_pred, y_scores, normal_errs, attack_errs


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

def load_test_csv(csv_path, stats, global_avg):
    """Load a parsed test CSV and return 13-dim feature matrices."""
    df = pd.read_csv(csv_path)
    n_mask = df["attack"].values == 0
    a_mask = df["attack"].values == 1

    def extract(id_mask):
        ids = df.loc[id_mask, "CAN_ID"].astype(str).values
        payload = np.hstack([
            df.loc[id_mask, BYTE_COLS].values.astype(np.float32),
            df.loc[id_mask, "DLC"].values.astype(np.float32).reshape(-1, 1),
        ]).astype(np.float32)
        return build_features(ids, payload, stats, global_avg)

    n_feats = extract(n_mask)
    a_feats = extract(a_mask)
    a_types = df.loc[a_mask, "attack_type"].values.astype(str)
    print(f"  Normal={len(n_feats):,} | Attack={len(a_feats):,} | Types: {np.unique(a_types)}")
    return n_feats, a_feats, a_types


def evaluate(checkpoint="best_model.pth"):
    model, device, threshold = load_model(checkpoint)
    if threshold is None:
        raise ValueError("No threshold found in checkpoint.")

    stats, global_avg = load_id_stats()

    test_csvs = sorted(DATA_DIR.glob("set01_test_*_frames.csv"))
    if not test_csvs:
        raise FileNotFoundError("No test CSVs found in data/. Run parse_can_train_test.py first.")

    all_results = {}
    for csv_path in test_csvs:
        test_name = csv_path.stem.replace("set01_", "").replace("_frames", "")
        print(f"\n{'='*60}")
        print(f"Test set: {test_name}")
        print(f"{'='*60}")
        n_feats, a_feats, atypes = load_test_csv(csv_path, stats, global_avg)
        y_true, y_pred, y_scores, ne, ae = detect(model, n_feats, a_feats, device, threshold)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_scores)

        print(f"\nOverall:")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"  F1: {f1:.4f} | PR-AUC: {pr_auc:.4f}  (primary)")

        print(f"\n{'Type':<20s} {'PR-AUC':>8s} {'F1':>8s} {'Recall':>8s} {'N':>8s}")
        print(f"{'-'*52}")
        for atype in np.unique(atypes):
            mask = atypes == atype
            n = mask.sum()
            at_true = np.concatenate([np.zeros(len(ne)), np.ones(n)])
            at_scores = np.concatenate([ne, ae[mask]])
            at_preds = np.concatenate([
                (ne > threshold).astype(int),
                (ae > threshold).astype(int)[mask],
            ])
            at_pr_auc = average_precision_score(at_true, at_scores) if n > 0 else 0
            at_f1 = f1_score(at_true, at_preds) if n > 0 else 0
            at_rec = recall_score(at_true, at_preds) if n > 0 else 0
            print(f"{atype:<20s} {at_pr_auc:>8.4f} {at_f1:>8.4f} {at_rec:>8.4f} {n:>8,}")

        all_results[test_name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "pr_auc": pr_auc,
        }

        plot_error_distribution(ne, ae, threshold, tag=test_name)
        plot_pr_curve(y_true, y_scores, pr_auc, tag=test_name)
        plot_confusion_matrix(y_true, y_pred, tag=test_name)

    print(f"\n{'='*60}")
    print(f"  SUMMARY (all test sets)")
    print(f"{'='*60}")
    print(f"{'Test Set':<45s} {'PR-AUC':>8s} {'F1':>8s}")
    print(f"{'-'*61}")
    for name, res in all_results.items():
        print(f"{name:<45s} {res['pr_auc']:>8.4f} {res['f1']:>8.4f}")

    return all_results


if __name__ == "__main__":
    evaluate()
