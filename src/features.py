"""ID behavioral statistics + per-frame feature construction.

Per-ID behavioral stats (computed once from training-normal data, no leak):
    - expected_period: median inter-arrival (ms), capped at 1.0ms
    - typical_dlc:     mode of raw DLC / 8
    - payload_variance: mean per-byte std over real bytes (DLC > i), /0.3 clipped
    - frequency_rank:  log(1+count) / log(1+max_count)

Unseen IDs fall back to the global-average stats across all training IDs.
Feature vector per frame = [4 stats, byte_0..7, DLC] -> 13 dims.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
BYTE_COLS = [f"byte_{i}" for i in range(8)]
STAT_NAMES = ["expected_period", "typical_dlc", "payload_variance", "frequency_rank"]

PERIOD_CAP_MS = 1.0
VAR_CAP = 0.3


def _normalize_period_ms(median_gap_ms):
    return float(min(median_gap_ms / PERIOD_CAP_MS, 1.0)) if np.isfinite(median_gap_ms) else 1.0


def fit_id_stats(df):
    """Compute per-ID behavioral stats from a normal-only DataFrame.

    df: DataFrame with CAN_ID (str), Timestamp, byte_0..byte_7 (normalized /255),
        DLC (normalized /8). Must be sorted by Timestamp.
    Returns: stats dict {can_id: [4 floats]} + global_avg list [4 floats].
    """
    df = df.copy()
    df["raw_dlc"] = (df["DLC"] * 8).round().astype(int)
    counts = df["CAN_ID"].value_counts()
    max_count = counts.max()

    stats = {}
    byte_arr = df[BYTE_COLS].values
    dlc_arr = df["raw_dlc"].values
    ts_arr = df["Timestamp"].values

    for cid, grp_idx in df.groupby("CAN_ID").groups.items():
        idx = grp_idx.to_numpy()
        g = df.loc[idx]

        ts = ts_arr[idx]
        gaps = np.diff(ts)
        gaps = gaps[gaps > 1e-6]
        period = np.median(gaps) * 1e3 if len(gaps) else float("nan")

        dlc_mode = g["raw_dlc"].mode().iloc[0]

        gb = byte_arr[idx]
        gd = dlc_arr[idx]
        stds = []
        for i in range(8):
            m = gd > i
            if m.sum() > 1:
                stds.append(gb[m, i].std())
        variance = float(np.mean(stds)) if stds else 0.0

        rank = np.log1p(counts[cid]) / np.log1p(max_count)

        stats[str(cid)] = [
            _normalize_period_ms(period),
            float(dlc_mode) / 8.0,
            float(min(variance / VAR_CAP, 1.0)),
            float(rank),
        ]

    global_avg = [float(np.mean([s[i] for s in stats.values()])) for i in range(4)]
    return stats, global_avg


def save_id_stats(stats, global_avg, path=None):
    payload = dict(stats)
    payload["global_avg"] = global_avg
    path = path or (DATA_DIR / "id_stats.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved id_stats: {len(stats)} IDs + global_avg -> {path}")


def load_id_stats(path=None):
    path = path or (DATA_DIR / "id_stats.json")
    with open(path) as f:
        payload = json.load(f)
    global_avg = payload.pop("global_avg")
    return payload, global_avg


def build_features(can_ids, payload, stats, global_avg):
    """Resolve per-ID stats into a (N, 13) feature matrix.

    can_ids: array of str CAN IDs.
    payload: (N, 9) float32 [byte_0..7, DLC].
    stats:   dict {can_id: [4 stats]} (unseen -> global_avg).
    """
    n = len(can_ids)
    vec = np.tile(np.asarray(global_avg, dtype=np.float32), (n, 1))

    table = {str(cid): np.asarray(v, dtype=np.float32) for cid, v in stats.items()}
    keys = np.array(sorted(table.keys()), dtype=object)
    table_mat = np.stack([table[k] for k in keys]).astype(np.float32)
    query = np.asarray(can_ids, dtype=object)
    loc = np.searchsorted(keys, query)
    loc = np.clip(loc, 0, len(keys) - 1)
    known = keys[loc] == query
    vec[known] = table_mat[loc[known]]

    return np.hstack([vec, payload.astype(np.float32)]).astype(np.float32)
