import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE

from src.model import SharedDNN


DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
MODEL_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\models")
FIG_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\figures")
FIG_DIR.mkdir(exist_ok=True)


def load_model(checkpoint="best_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_DIR / checkpoint, map_location=device, weights_only=False)
    embedding_dim = ckpt.get("embedding_dim", 16)

    model = SharedDNN(input_dim=64, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    return model, device


def load_features(csv_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    feat_cols = [f"freq_bit_{i}" for i in range(64)]
    return df[feat_cols].values.astype(np.float32), df


def get_embeddings(model, features, device, batch_size=256):
    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.tensor(features[i:i+batch_size], dtype=torch.float32).to(device)
            emb = model(batch)
            all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)


def compute_distances(anchor_emb, query_emb):
    return np.sum((anchor_emb - query_emb) ** 2, axis=1)


def detect(model, normal_feats, attack_feats, device, threshold=None):
    """
    Detection logic:
        1. Compute anchor embedding (mean of normal training data)
        2. For each query, compute distance to anchor
        3. If distance > threshold -> attack, else normal
    """
    # Anchor = mean embedding of all normal data
    normal_emb = get_embeddings(model, normal_feats, device)
    anchor_emb = normal_emb.mean(axis=0, keepdims=True)

    # Compute distances
    normal_dists = compute_distances(anchor_emb, normal_emb)

    attack_emb = get_embeddings(model, attack_feats, device)
    attack_dists = compute_distances(anchor_emb, attack_emb)

    # Auto threshold: midpoint between mean normal dist and mean attack dist
    if threshold is None:
        threshold = (normal_dists.mean() + attack_dists.mean()) / 2
    print(f"Threshold: {threshold:.6f}")

    # Predictions
    normal_preds = (normal_dists > threshold).astype(int)  # 0=normal, 1=attack
    attack_preds = (attack_dists > threshold).astype(int)

    y_true = np.concatenate([np.zeros(len(normal_dists)), np.ones(len(attack_dists))])
    y_pred = np.concatenate([normal_preds, attack_preds])

    return y_true, y_pred, normal_dists, attack_dists, threshold


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
    print(f"Saved: figures/loss_curve.png")


def plot_distance_distribution(normal_dists, attack_dists, threshold):
    plt.figure(figsize=(10, 5))
    plt.hist(normal_dists, bins=80, alpha=0.7, label="Normal", color="green", density=True)
    plt.hist(attack_dists, bins=80, alpha=0.7, label="Attack", color="red", density=True)
    plt.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.4f}")
    plt.xlabel("Distance to Anchor")
    plt.ylabel("Density")
    plt.title("Distance Distribution: Normal vs Attack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distance_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: figures/distance_distribution.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: figures/confusion_matrix.png")


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
    print(f"Saved: figures/tsne_embeddings.png")


def load_test_split():
    """Load the held-out test split saved by train.py."""
    split_path = DATA_DIR / "test_split.npz"
    data = np.load(split_path)
    return (
        data["normal_test_feats"],
        data["normal_test_ids"],
        data["attack_test_feats"],
        data["attack_test_ids"],
    )


def evaluate(checkpoint="best_model.pth"):
    model, device = load_model(checkpoint)

    # Load the SAME held-out test split that train.py produced
    normal_feats, normal_ids, attack_feats, attack_ids = load_test_split()
    print(f"Test split: Normal={len(normal_feats):,} | Attack={len(attack_feats):,}")

    # Detect
    y_true, y_pred, normal_dists, attack_dists, threshold = detect(model, normal_feats, attack_feats, device)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*60}")

    # Plots
    plot_loss_curve()
    plot_distance_distribution(normal_dists, attack_dists, threshold)
    plot_confusion_matrix(y_true, y_pred)
    plot_tsne(normal_feats, attack_feats, model, device)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    evaluate()
