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
from sklearn.manifold import TSNE

from src.model import SiameseNetwork
from src.train import build_features, BYTE_COLS, TOP_K


DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
FIG_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\figures")
FIG_DIR.mkdir(exist_ok=True)


def load_model(checkpoint="best_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_DIR / checkpoint, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", 40)
    embedding_dim = ckpt.get("embedding_dim", 16)
    hidden_dims = ckpt.get("hidden_dims", None)
    margin = ckpt.get("margin", 1.0)
    threshold = ckpt.get("threshold", None)

    siamese = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim, margin=margin, hidden_dims=hidden_dims).to(device)
    siamese.load_state_dict(ckpt["model_state_dict"])
    siamese.eval()
    model = siamese.shared_dnn

    print(f"Loaded model from epoch {ckpt['epoch']}, val_loss={ckpt['loss']:.6f}")
    if threshold is not None:
        print(f"Threshold (from training data): {threshold:.6f}")
    else:
        print("WARNING: No threshold found in checkpoint!")
    return model, device, threshold


def get_embeddings(model, features, device, batch_size=256):
    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.tensor(features[i:i+batch_size], dtype=torch.float32).to(device)
            emb = model(batch)
            all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)


def detect(model, normal_feats, normal_ids, attack_feats, attack_ids, device, threshold):
    """
    Detection logic (per-ID centroids, fixed threshold from training):
        1. Build per-ID centroids from normal test data
        2. Compute distances to own-ID centroids
        3. Apply threshold saved during training (no threshold leakage)
    """
    # Compute embeddings
    normal_emb = get_embeddings(model, normal_feats, device)
    attack_emb = get_embeddings(model, attack_feats, device)

    # Build per-ID centroids from normal test data
    id_centroids = {}
    unique_ids = np.unique(normal_ids)
    for cid in unique_ids:
        mask = normal_ids == cid
        id_centroids[cid] = normal_emb[mask].mean(axis=0)

    # Fallback: global centroid
    global_centroid = normal_emb.mean(axis=0, keepdims=True)

    # Compute distances (to own-ID centroid if available, else global)
    def compute_dist_to_centroid(emb, ids):
        dists = np.zeros(len(emb))
        for i, cid in enumerate(ids):
            if cid in id_centroids:
                centroid = id_centroids[cid]
            else:
                centroid = global_centroid[0]
            dists[i] = np.sum((emb[i] - centroid) ** 2)
        return dists

    normal_dists = compute_dist_to_centroid(normal_emb, normal_ids)
    attack_dists = compute_dist_to_centroid(attack_emb, attack_ids)

    # Apply threshold from training (no computation from test data)
    print(f"Threshold: {threshold:.6f}")

    # Predictions
    normal_preds = (normal_dists > threshold).astype(int)
    attack_preds = (attack_dists > threshold).astype(int)

    y_true = np.concatenate([np.zeros(len(normal_dists)), np.ones(len(attack_dists))])
    y_pred = np.concatenate([normal_preds, attack_preds])
    y_scores = np.concatenate([normal_dists, attack_dists])

    return y_true, y_pred, y_scores, normal_dists, attack_dists


# ──────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────

def plot_loss_curve():
    history = pd.read_csv(DATA_DIR / "training_history.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], linewidth=2, label="Train")
    plt.plot(history["epoch"], history["val_loss"], linewidth=2, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loss_curve.png", dpi=150)
    plt.close()
    print("Saved: figures/loss_curve.png")


def plot_distance_distribution(normal_dists, attack_dists, threshold):
    plt.figure(figsize=(10, 5))
    plt.hist(normal_dists, bins=80, alpha=0.7, label="Normal", color="green", density=True)
    plt.hist(attack_dists, bins=80, alpha=0.7, label="Attack", color="red", density=True)
    plt.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.4f}")
    plt.xlabel("Distance to Centroid")
    plt.ylabel("Density")
    plt.title("Distance Distribution: Normal vs Attack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distance_distribution.png", dpi=150)
    plt.close()
    print("Saved: figures/distance_distribution.png")


def plot_pr_curve(y_true, y_scores, pr_auc):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curve.png", dpi=150)
    plt.close()
    print("Saved: figures/pr_curve.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: figures/confusion_matrix.png")


def plot_tsne(normal_feats, attack_feats, model, device, n_samples=2000):
    n = min(n_samples, len(normal_feats), len(attack_feats))
    feats = np.vstack([normal_feats[:n], attack_feats[:n]])
    labels = np.array([0]*n + [1]*n)

    emb = get_embeddings(model, feats, device)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(10, 8))
    colors = ["green", "red"]
    names = ["Normal", "Attack"]
    for i in range(2):
        mask = labels == i
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=colors[i], label=names[i], alpha=0.5, s=10)
    plt.legend(fontsize=12)
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tsne_embeddings.png", dpi=150)
    plt.close()
    print("Saved: figures/tsne_embeddings.png")


# ──────────────────────────────────────────────
#  Main evaluation
# ──────────────────────────────────────────────

def load_test_csv(csv_path, top_ids):
    """Load a parsed test CSV and build 40-dim features, split normal/attack."""
    df = pd.read_csv(csv_path)
    n_mask = df["attack"].values == 0
    a_mask = df["attack"].values == 1
    n_ids = df.loc[n_mask, "CAN_ID"].astype(str).values
    n_feats = build_features(
        n_ids, top_ids,
        df.loc[n_mask, BYTE_COLS].values.astype(np.float32),
        df.loc[n_mask, "DLC"].values.astype(np.float32),
    )
    a_ids = df.loc[a_mask, "CAN_ID"].astype(str).values
    a_feats = build_features(
        a_ids, top_ids,
        df.loc[a_mask, BYTE_COLS].values.astype(np.float32),
        df.loc[a_mask, "DLC"].values.astype(np.float32),
    )
    a_types = df.loc[a_mask, "attack_type"].values.astype(str)
    print(f"  Normal={len(n_feats):,} | Attack={len(a_feats):,} | Types: {np.unique(a_types)}")
    return n_feats, n_ids, a_feats, a_ids, a_types


def evaluate(checkpoint="best_model.pth"):
    model, device, threshold = load_model(checkpoint)
    if threshold is None:
        raise ValueError("No threshold found in checkpoint. Re-run train.py to generate one.")

    with open(DATA_DIR / "top_can_ids.json") as f:
        top_ids = json.load(f)

    test_csvs = sorted(DATA_DIR.glob("set01_test_*_frames.csv"))
    if not test_csvs:
        raise FileNotFoundError("No test CSVs found in data/. Run parse_can_train_test.py first.")

    all_results = {}
    for csv_path in test_csvs:
        test_name = csv_path.stem.replace("set01_", "").replace("_frames", "")
        print(f"\n{'='*60}")
        print(f"Test set: {test_name}")
        print(f"{'='*60}")
        nf, ni, af, ai, atypes = load_test_csv(csv_path, top_ids)
        y_true, y_pred, y_scores, nd, ad = detect(model, nf, ni, af, ai, device, threshold)

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
            at_scores = ad[mask]
            at_pred = (ad > threshold).astype(int)[mask]
            at_pr_auc = average_precision_score(np.ones(n), at_scores) if n > 0 else 0
            at_f1 = f1_score(np.ones(n), at_pred) if n > 0 else 0
            at_rec = recall_score(np.ones(n), at_pred) if n > 0 else 0
            print(f"{atype:<20s} {at_pr_auc:>8.4f} {at_f1:>8.4f} {at_rec:>8.4f} {n:>8,}")

        all_results[test_name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "pr_auc": pr_auc,
        }

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
