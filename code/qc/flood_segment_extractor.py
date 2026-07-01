"""Extract continuous nonzero flood-depth segments from sensor time series."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


def file_columns(path: str) -> list:
    """Reads only the parquet schema / the CSV header row, so it is instant even on huge files."""
    if path.endswith(".parquet"):
        return pq.read_schema(path).names
    return list(pd.read_csv(path, nrows=0).columns)


def read_depth_table(path: str) -> pd.DataFrame:
    """Read the depth input file and normalize the time column name to 'timestamp'."""
    names = file_columns(path)
    time_col = "timestamp" if "timestamp" in names else "time"
    cols = ["slug", time_col, "depth_proc_mm"]

    if path.endswith(".parquet"):  # parquet loads much faster
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols, engine="pyarrow")

    df = df.rename(columns={time_col: "timestamp"})
    df["slug"] = df["slug"].astype("category")
    return df


def identify_segments_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Identify continuous non-zero flood depth segments using vectorized NumPy operations."""

    df = df.sort_values("timestamp").reset_index(drop=True)

    depths = df["depth_proc_mm"].values

    is_nonzero = (depths > 0) & ~np.isnan(depths)

    if not is_nonzero.any():
        return pd.DataFrame(columns=["segment_id", "timestamp", "depth_proc_mm"])

    padded = np.concatenate([[False], is_nonzero, [False]])

    diff = np.diff(padded.astype(np.int8))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Boundary zeros preserve the visible start and end of each segment.
    segment_ids = np.zeros(len(df), dtype=np.int32)
    boundary_records = []

    for seg_id, (s, e) in enumerate(zip(starts, ends), 1):
        segment_ids[s:e] = seg_id

        start_time = df.iloc[s]["timestamp"]
        boundary_records.append(
            {
                "segment_id": seg_id,
                "timestamp": start_time - pd.Timedelta(minutes=1),
                "depth_proc_mm": 0.0,
            }
        )

        end_time = df.iloc[e - 1]["timestamp"]
        boundary_records.append(
            {
                "segment_id": seg_id,
                "timestamp": end_time + pd.Timedelta(minutes=1),
                "depth_proc_mm": 0.0,
            }
        )

    nonzero_df = df[is_nonzero].copy()
    nonzero_df["segment_id"] = segment_ids[is_nonzero]

    if boundary_records:
        boundary_df = pd.DataFrame(boundary_records)
        result_df = pd.concat(
            [nonzero_df[["segment_id", "timestamp", "depth_proc_mm"]], boundary_df],
            ignore_index=True,
        )
        result_df = result_df.sort_values(["segment_id", "timestamp"]).reset_index(
            drop=True
        )
    else:
        result_df = nonzero_df[["segment_id", "timestamp", "depth_proc_mm"]]

    return result_df


class FloodSegmentExtractor:
    """Extracts continuous non-zero flood depth segments from FloodNet depth data."""

    def __init__(
        self,
        input_path: str = str(
            REPO_ROOT / "data" / "02_preprocessed" / "floodnet.parquet"
        ),
        output_path: str = str(REPO_ROOT / "outputs" / "qc" / "flood_segments.csv"),
    ):

        self.input_path = input_path
        self.output_path = output_path

    def extract_all_segments(self) -> pd.DataFrame:
        """Extract segments for all sensors and save to a single CSV file."""
        print("[1/4] Loading data...")
        df = read_depth_table(self.input_path)
        print(f"      Loaded {len(df):,} rows")

        print("[2/4] Sorting data...")
        df = df.sort_values(["slug", "timestamp"])
        print("      Sorting complete")

        print("[3/4] Processing sensors...")
        result_dfs = []
        total_segments = 0
        total_records = 0

        grouped = df.groupby("slug", observed=True)
        pbar = tqdm(grouped, desc="Processing")

        for sensor, group in pbar:
            segments_df = identify_segments_vectorized(group)

            if len(segments_df) > 0:
                segments_df["slug"] = sensor
                result_dfs.append(segments_df)
                total_segments += int(segments_df["segment_id"].max())
                total_records += len(segments_df)

            pbar.set_postfix(
                {
                    "sensor": str(sensor)[:20],
                    "segments": total_segments,
                    "records": f"{total_records:,}",
                }
            )

        if result_dfs:
            print("[4/4] Saving results...")
            all_segments = pd.concat(result_dfs, ignore_index=True)
            all_segments = all_segments[
                ["slug", "segment_id", "timestamp", "depth_proc_mm"]
            ]

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            all_segments.to_csv(self.output_path, index=False)

            print(f"      Saved to {self.output_path}")
            print()
            print("=" * 50)
            print("Extraction complete:")
            print(f"  - Sensors: {len(result_dfs)}")
            print(f"  - Total segments: {total_segments:,}")
            print(f"  - Total records: {len(all_segments):,}")
            print(f"  - Output file: {self.output_path}")
            print("=" * 50)

            return all_segments

        print("No segments found")
        return pd.DataFrame(
            columns=["slug", "segment_id", "timestamp", "depth_proc_mm"]
        )

    def extract_single_sensor(self, sensor_slug: str) -> Optional[pd.DataFrame]:
        """Extract segments for a single sensor."""
        print(f"[1/2] Loading data for sensor: {sensor_slug}...")

        df = read_depth_table(self.input_path)

        sensor_df = df[df["slug"] == sensor_slug]

        if len(sensor_df) == 0:
            print(f"      Sensor '{sensor_slug}' not found")
            return None

        print(f"      Found {len(sensor_df):,} records")

        print("[2/2] Extracting segments...")
        segments_df = identify_segments_vectorized(sensor_df)

        if len(segments_df) > 0:
            segments_df["slug"] = sensor_slug
            segments_df = segments_df[
                ["slug", "segment_id", "timestamp", "depth_proc_mm"]
            ]
            print(
                f"      Found {segments_df['segment_id'].max()} segments, {len(segments_df):,} records"
            )
        else:
            print("      No non-zero segments found")

        return segments_df
