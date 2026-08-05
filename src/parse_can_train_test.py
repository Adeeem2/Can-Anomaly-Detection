"""Parse can-train-and-test CSV files into per-frame features with labels.

Input format (per file): timestamp,arbitration_id,data_field,attack
  - data_field: hex string (e.g., "3000000430000004")
  - attack: 0 = normal, 1 = attack

Output: combined CSV sorted by timestamp
  - CAN_ID, Timestamp, byte_0..byte_7 (normalized /255),
    DLC (normalized /8), attack (0/1), attack_type (string)
  - Missing bytes (DLC < 8) padded with 0x00 before /255 → 0.0
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection")
CANTT_DIR = BASE_DIR / "can-train-and-test"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

BYTE_COLS = [f"byte_{i}" for i in range(8)]


def _hex_to_bytes(hex_series):
    """Convert Series of hex strings to (8-column DataFrame, DLC Series).

    Handles empty/missing hex strings (DLC=0, all bytes = 0).
    """
    cleaned = hex_series.fillna("").astype(str).str.strip()
    dlc = (cleaned.str.len() // 2).astype(np.float32)

    padded = cleaned.str.ljust(16, "0")
    vals = padded.apply(lambda x: int(x, 16)).values

    byte_df = pd.DataFrame(index=hex_series.index)
    for i in range(8):
        shift = 8 * (7 - i)
        byte_df[f"byte_{i}"] = np.where(
            dlc > i,
            ((vals >> shift) & 0xFF).astype(np.float32),
            0.0,
        )

    return byte_df, dlc


SET_NAMES = ["set_01", "set_02", "set_03", "set_04"]


def parse_one_file(csv_path):
    """Parse a single raw CSV into a per-frame feature DataFrame."""
    df = pd.read_csv(csv_path, dtype={"attack": np.int8})
    fname = csv_path.stem
    atype = fname.rsplit("-", 1)[0]
    if atype in ("attack-free", "accessory"):
        atype = "normal"

    byte_df, dlc = _hex_to_bytes(df["data_field"])

    out = pd.DataFrame({
        "CAN_ID": df["arbitration_id"].astype(str).str.upper(),
        "Timestamp": df["timestamp"].astype(np.float64),
        **{c: byte_df[c].values / 255.0 for c in BYTE_COLS},
        "DLC": dlc.values / 8.0,
        "attack": df["attack"].values,
        "attack_type": np.where(df["attack"].values == 1, atype, "normal"),
    }, index=df.index)
    return out


def process_set_csvs(set_dir, output_name):
    """Process all CSV files in a set directory, output combined sorted CSV."""
    chunks = []
    for csv_path in sorted(set_dir.glob("*.csv")):
        out = parse_one_file(csv_path)
        chunks.append(out)
        print(f"  {csv_path.name}: {len(out):,} rows ({out['attack'].sum():,} attack)")

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values("Timestamp").reset_index(drop=True)
    combined.to_csv(OUTPUT_DIR / output_name, index=False)
    n_attack = combined["attack"].sum()
    print(f"\nSaved {output_name} ({len(combined):,} rows, "
          f"{n_attack:,} attack, {len(combined) - n_attack:,} normal)")


def merge_train_sets():
    """Merge train_01 from all 4 sets into data/all_train_frames.csv (no global sort)."""
    all_chunks = []
    for set_name in SET_NAMES:
        train_dir = CANTT_DIR / set_name / "train_01"
        print(f"\n[{set_name}] train_01:")
        for csv_path in sorted(train_dir.glob("*.csv")):
            out = parse_one_file(csv_path)
            all_chunks.append(out)
            print(f"  {csv_path.name}: {len(out):,} rows ({out['attack'].sum():,} attack)")

    combined = pd.concat(all_chunks, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_train_frames.csv", index=False)
    n_attack = combined["attack"].sum()
    print(f"\nSaved all_train_frames.csv ({len(combined):,} rows, "
          f"{n_attack:,} attack, {len(combined) - n_attack:,} normal)")


if __name__ == "__main__":
    merge_train_sets()
