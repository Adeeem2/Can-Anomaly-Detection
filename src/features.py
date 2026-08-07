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
        DLC (normalized /8). When a 'Session' column is present, inter-arrival gaps
        and byte deltas are computed WITHIN each session only: the merged CSV
        interleaves independent recordings (all start at the same epoch second),
        so a global chronological sort must never produce temporal neighbors.
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

        gap_parts, dmean_parts = [], []
        if "Session" in df.columns:
            for _, sub in g.groupby("Session", sort=False):
                sub = sub.sort_values("Timestamp")
                if len(sub) < 2:
                    continue
                ts = sub["Timestamp"].values
                sg = np.diff(ts)
                sg = sg[sg > 1e-6]
                gap_parts.append(sg)
                gb = sub[BYTE_COLS].values
                dmean_parts.append(np.abs(np.diff(gb, axis=0)).mean(axis=1))
        else:
            ts = g["Timestamp"].values
            sg = np.diff(ts)
            sg = sg[sg > 1e-6]
            gap_parts.append(sg)
            if len(idx) > 1:
                dmean_parts.append(np.abs(np.diff(byte_arr[idx], axis=0)).mean(axis=1))

        gaps = np.concatenate(gap_parts) if gap_parts else np.array([], dtype=np.float64)
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

        dmeans = np.concatenate(dmean_parts) if dmean_parts else np.array([], dtype=np.float64)
        median_dmean = float(np.median(dmeans)) if len(dmeans) else float("nan")
        median_gap_ms = float(np.median(gaps) * 1e3) if len(gaps) else float("nan")

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


class ByteDeltaRoller:
    """Rolling byte-delta ("freeze signal") features, chunk-safe.

    For each frame, byte delta = current 8 bytes - previous same-ID frame's 8 bytes,
    computed over the FULL stream (attack frames included) so that a frozen payload
    (e.g. DoS replay) shows near-zero deltas. When a 'Session' column is present the
    "previous same-ID frame" is looked up WITHIN the same session only - the merged
    CSV interleaves independent recordings, so a global sort would compare frames
    from different sessions and destroy the freeze signal. Summarized over the
    "real" bytes only (index < raw DLC):
        byte_delta_mean_abs = mean |delta|
        byte_delta_std      = std of the 8 deltas
    First occurrence of an ID in a session (no predecessor) -> [0, 0].
    """

    def __init__(self):
        self.last = {}

    def advance(self, df):
        ids = df["CAN_ID"].astype(str).values
        b = df[BYTE_COLS].values.astype(np.float32)
        dlc = (df["DLC"].values * 8).round().astype(np.int64)
        n = len(df)

        if "Session" in df.columns:
            sess = df["Session"].astype(str).values
            prev = pd.DataFrame(b).groupby(
                [pd.Series(sess), pd.Series(ids)], sort=False
            ).shift(1).values.astype(np.float32)
        else:
            prev = pd.DataFrame(b).groupby(pd.Series(ids), sort=False).shift(1).values.astype(np.float32)
        is_first = np.isnan(prev).all(axis=1)

        d = np.zeros((n, 8), dtype=np.float32)
        nz = ~is_first
        d[nz] = b[nz] - prev[nz]

        # carry the last payload per (session, id) across chunk boundaries
        if "Session" in df.columns:
            for (s, cid), gi in df.groupby(["Session", "CAN_ID"], sort=False).indices.items():
                i0, ilast = gi[0], gi[-1]
                if is_first[i0] and (s, cid) in self.last:
                    d[i0] = b[i0] - self.last[(s, cid)]
                self.last[(s, cid)] = b[ilast]
        else:
            for cid in pd.unique(ids):
                if cid in self.last:
                    m = (ids == cid) & is_first
                    if m.any():
                        d[m] = b[m] - self.last[cid]
                self.last[cid] = b[ids == cid][-1]

        real = dlc[:, None] > np.arange(8)
        n_real = real.sum(axis=1)
        ad = np.where(real, np.abs(d), 0.0)
        d2 = np.where(real, d, 0.0)
        mean_abs = ad.sum(axis=1) / np.maximum(n_real, 1)
        with np.errstate(invalid="ignore"):
            mean_sq = (d2 * d2).sum(axis=1) / np.maximum(n_real, 1)
            std = np.sqrt(np.clip(mean_sq - mean_abs ** 2, 0, None))
        mean_abs = np.where(n_real > 0, mean_abs, 0.0)
        std = np.where(n_real > 1, std, 0.0)
        return np.column_stack([mean_abs, std]).astype(np.float32)


def compute_byte_delta_features(df):
    """Single-pass byte-delta features over a full DataFrame (in-memory equivalent
    of rolling a fresh ``ByteDeltaRoller`` once over ``df``)."""
    return ByteDeltaRoller().advance(df)
