"""Extract per-frame features from parsed CAN CSVs.

Output per-frame vector (40-dim after one-hot in train.py):
  - 8 data bytes normalized as byte/255 (missing bytes for DLC<8 filled with 0)
  - 1 DLC normalized as dlc/8
  - CAN_ID and Timestamp passed through for downstream splitting/one-hot

No windowing, no bit extraction.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")
BYTE_COLS = [f'DATA[{i}]' for i in range(8)]


def process_frames(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Flag'] != 100].copy().reset_index(drop=True)

    dlc = df['DLC'].values.astype(np.float32)
    bytes_raw = df[BYTE_COLS].values.astype(np.float32)

    # Fill missing bytes with -1 (outside valid [0,1] range after /255 normalization)
    for i in range(8):
        bytes_raw[dlc <= i, i] = -1.0

    out = pd.DataFrame({
        'CAN_ID': df['CAN_ID'].astype(str),
        'Timestamp': df['Timestamp'].values,
    })
    for i in range(8):
        out[f'byte_{i}'] = bytes_raw[:, i] / 255.0
    out['DLC'] = dlc / 8.0

    return out


if __name__ == "__main__":
    datasets = [
        ("Attack_free_parsed.csv", "Attack_free_frames.csv"),
        ("DoS_attack_parsed.csv", "DoS_attack_frames.csv"),
        ("Fuzzy_attack_parsed.csv", "Fuzzy_attack_frames.csv"),
        ("Impersonation_attack_parsed.csv", "Impersonation_attack_frames.csv"),
    ]
    for inp, out_name in datasets:
        print(f"Processing {inp}...")
        df = process_frames(DATA_DIR / inp)
        df.to_csv(DATA_DIR / out_name, index=False)
        print(f"  -> {out_name} ({len(df):,} frames, {df['CAN_ID'].nunique()} IDs)")
