"""ID behavioral statistics + per-frame feature construction.

Per-ID behavioral stats (computed once from training-normal data, no leak):
    - expected_period:  median inter-arrival (ms), capped at 1.0ms
    - typical_dlc:      mode of raw DLC / 8
    - payload_variance: mean per-byte std over real bytes (DLC > i), /0.3 clipped
    - frequency_rank:   log(1+count) / log(1+max_count)
    - median_dmean:     median mean |delta byte| vs previous same-ID frame
    - median_gap_ms:    median inter-arrival (ms), uncapped

Unseen IDs fall back to the global-average stats across all training IDs.
Base feature vector per frame = [4 stats, byte_0..7, DLC] -> 13 dims.
Temporal features [dmean, gap_norm] -> 15 dims total (compute_temporal_features).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
BYTE_COLS = [f"byte_{i}" for i in range(8)]
STAT_NAMES = ["expected_period", "typical_dlc", "payload_variance", "frequency_rank",
              "median_dmean", "median_gap_ms"]

PERIOD_CAP_MS = 1.0
VAR_CAP = 0.3
GAP_CAP_MS = 10.0


def _normalize_period_ms(median_gap_ms):
    return float(min(median_gap_ms / PERIOD_CAP_MS, 1.0)) if np.isfinite(median_gap_ms) else 1.0


def fit_id_stats(df):
    """Compute per-ID behavioral stats from a normal-only DataFrame.

    df: DataFrame with CAN_ID (str), Timestamp, byte_0..byte_7 (normalized /255),
        DLC (normalized /8). Must be sorted by Timestamp.
    Returns: stats dict {can_id: [6 floats]} + global_avg list [6 floats].
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

        if len(gb) > 1:
            median_dmean = float(np.median(np.abs(np.diff(gb, axis=0)).mean(axis=1)))
            median_gap_ms = float(np.median(gaps) * 1e3) if len(gaps) else float("nan")
        else:
            median_dmean = float("nan")
            median_gap_ms = float("nan")

        stats[str(cid)] = [
            _normalize_period_ms(period),
            float(dlc_mode) / 8.0,
            float(min(variance / VAR_CAP, 1.0)),
            float(rank),
            median_dmean,
            median_gap_ms,
        ]

    global_avg = [float(np.nanmean([s[i] for s in stats.values()])) for i in range(6)]
    for cid in stats:
        for i in range(6):
            v = stats[cid][i]
            if not np.isfinite(v):
                stats[cid][i] = global_avg[i]
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
    stats:   dict {can_id: [6 stats]} (unseen -> global_avg); only [:4] used here.
    """
    n = len(can_ids)
    vec = np.tile(np.asarray(global_avg[:4], dtype=np.float32), (n, 1))

    table = {str(cid): np.asarray(v, dtype=np.float32) for cid, v in stats.items()}
    keys = np.array(sorted(table.keys()), dtype=object)
    table_mat = np.stack([table[k][:4] for k in keys]).astype(np.float32)
    query = np.asarray(can_ids, dtype=object)
    loc = np.searchsorted(keys, query)
    loc = np.clip(loc, 0, len(keys) - 1)
    known = keys[loc] == query
    vec[known] = table_mat[loc[known]]

    return np.hstack([vec, payload.astype(np.float32)]).astype(np.float32)


def compute_temporal_features(df, stats, global_avg):
    """Compute per-frame temporal features -> (N, 2) float32 [dmean, gap_norm].

    df: DataFrame with CAN_ID (str), Timestamp, byte_0..byte_7, sorted by Timestamp.
    dmean    = mean |delta byte_0..7| vs previous same-ID frame (in [0,1]).
    gap_norm = min(ms since previous same-ID frame / GAP_CAP_MS, 1.0).
    First frame of each ID (no predecessor) uses the per-ID training baselines
    (stats[idx 4] = median dmean, idx 5 = median gap ms), or global_avg for unseen IDs.
    """
    n = len(df)
    dmean = np.full(n, np.nan, dtype=np.float32)
    gap_norm = np.full(n, np.nan, dtype=np.float32)
    ts = df["Timestamp"].values
    byte_arr = df[BYTE_COLS].values.astype(np.float32)

    for cid, grp_idx in df.groupby("CAN_ID").groups.items():
        i = grp_idx.to_numpy()
        if len(i) == 0:
            continue
        s = stats.get(str(cid))
        base_dmean = s[4] if s is not None and np.isfinite(s[4]) else global_avg[4]
        base_gap = s[5] if s is not None and np.isfinite(s[5]) else global_avg[5]

        if len(i) > 1:
            dmean[i[1:]] = np.abs(np.diff(byte_arr[i], axis=0)).mean(axis=1)
            gap_norm[i[1:]] = np.minimum(np.diff(ts[i]) * 1e3 / GAP_CAP_MS, 1.0)

        dmean[i[0]] = base_dmean
        gap_norm[i[0]] = np.minimum(base_gap / GAP_CAP_MS, 1.0)

    return np.column_stack([dmean, gap_norm]).astype(np.float32)
