import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"D:\PROJECT\STAGEKPIT\can-anomaly-detection\data")


def load_parsed_data(csv_path):
    """Load the parsed CSV dataset."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} messages")
    return df


def extract_bit_vectors(df):
    """Convert 8 data bytes to 64-bit vector using numpy."""
    df = df[df['Flag'] != 100].copy().reset_index(drop=True)
    print(f"After filtering remote frames: {len(df):,} messages")

    data_cols = [f'DATA[{i}]' for i in range(8)]
    data_matrix = df[data_cols].values.astype(np.uint8)

    bit_matrix = np.zeros((len(data_matrix), 64), dtype=np.uint8)
    for byte_idx in range(8):
        for bit_idx in range(8):
            col = byte_idx * 8 + bit_idx
            bit_matrix[:, col] = (data_matrix[:, byte_idx] >> (7 - bit_idx)) & 1

    print(f"Bit matrix shape: {bit_matrix.shape}")
    return df, bit_matrix


def compute_bit_frequencies(df, bit_matrix, window_size=20):
    """Compute bit frequency vectors over non-overlapping windows."""
    print(f"Computing bit frequencies (window={window_size})...")

    freq_records = []

    for can_id, idx in df.groupby('CAN_ID').groups.items():
        idx = np.array(idx)
        if len(idx) < window_size:
            continue

        id_bits = bit_matrix[idx]
        id_df = df.iloc[idx]
        timestamps = id_df['Timestamp'].values
        msg_type = id_df['Type'].iloc[0]

        n_windows = len(idx) // window_size
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window = id_bits[start:end]
            freq = window.mean(axis=0)

            record = {
                'CAN_ID': can_id,
                'Type': msg_type,
                'Timestamp_Start': timestamps[start],
                'Timestamp_End': timestamps[end - 1],
            }
            for i in range(64):
                record[f'freq_bit_{i}'] = freq[i]

            freq_records.append(record)

    result = pd.DataFrame(freq_records)
    print(f"Created {len(result):,} frequency windows")
    return result


def process_dataset(input_csv, output_csv, window_size=20):
    """Full pipeline: load -> extract bits -> compute frequencies -> save."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_csv.name}")
    print(f"{'='*60}")

    df = load_parsed_data(input_csv)
    df, bit_matrix = extract_bit_vectors(df)
    freq_df = compute_bit_frequencies(df, bit_matrix, window_size)
    freq_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv.name}")

    return freq_df


if __name__ == "__main__":
    datasets = [
        "Attack_free_parsed.csv",
        "DoS_attack_parsed.csv",
        "Fuzzy_attack_parsed.csv",
        "Impersonation_attack_parsed.csv",
    ]

    all_stats = []

    for ds in datasets:
        input_csv = DATA_DIR / ds
        output_name = ds.replace("_parsed.csv", "_training.csv")
        output_csv = DATA_DIR / output_name
        freq_df = process_dataset(input_csv, output_csv)

        all_stats.append({
            'Dataset': ds.replace("_parsed.csv", ""),
            'Windows': len(freq_df),
            'IDs': freq_df['CAN_ID'].nunique(),
            'MODE': len(freq_df[freq_df['Type'] == 'MODE']),
            'VALUE': len(freq_df[freq_df['Type'] == 'VALUE']),
        })

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = pd.DataFrame(all_stats)
    print(summary.to_string(index=False))
