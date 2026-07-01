"""Utility functions for querying and visualizing flood segments."""

import os
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm


DEFAULT_SEGMENTS_PATH = "data/segments/flood_segments.csv"


def load_segments(path: str = DEFAULT_SEGMENTS_PATH) -> pd.DataFrame:
    """Load the flood segments CSV file."""
    return pd.read_csv(path, parse_dates=["timestamp"])


def get_sensor_segments(df: pd.DataFrame, sensor_slug: str) -> pd.DataFrame:
    """Filter segments for a specific sensor."""
    return df[df["slug"] == sensor_slug].reset_index(drop=True)


def get_segment(df: pd.DataFrame, sensor_slug: str, segment_id: int) -> pd.DataFrame:
    """Get a specific segment from a sensor's data."""
    mask = (df["slug"] == sensor_slug) & (df["segment_id"] == segment_id)
    return df[mask].reset_index(drop=True)


def list_sensors(df: pd.DataFrame) -> List[str]:
    """List all sensors in the segments data."""
    return sorted(df["slug"].unique().tolist())


def get_segment_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get overall statistics for all segments."""
    stats = {
        "total_sensors": df["slug"].nunique(),
        "total_records": len(df),
        "total_segments": df.groupby("slug")["segment_id"].max().sum(),
        "sensors": {},
    }

    for sensor in df["slug"].unique():
        sensor_df = df[df["slug"] == sensor]
        stats["sensors"][sensor] = {
            "num_segments": int(sensor_df["segment_id"].max()),
            "num_records": len(sensor_df),
            "max_depth_mm": float(sensor_df["depth_proc_mm"].max()),
            "mean_depth_mm": float(sensor_df["depth_proc_mm"].mean()),
        }

    return stats


def get_sensor_stats(df: pd.DataFrame, sensor_slug: str) -> Dict[str, Any]:
    """Get statistics for a specific sensor's segments."""
    sensor_df = df[df["slug"] == sensor_slug]

    if len(sensor_df) == 0:
        return {"sensor_slug": sensor_slug, "num_segments": 0, "segments": []}

    stats = {
        "sensor_slug": sensor_slug,
        "num_segments": int(sensor_df["segment_id"].max()),
        "total_records": len(sensor_df),
        "segments": [],
    }

    for seg_id in sensor_df["segment_id"].unique():
        seg_df = sensor_df[sensor_df["segment_id"] == seg_id]
        seg_stats = {
            "segment_id": int(seg_id),
            "num_records": len(seg_df),
            "start_time": seg_df["timestamp"].min().isoformat(),
            "end_time": seg_df["timestamp"].max().isoformat(),
            "duration_minutes": (
                seg_df["timestamp"].max() - seg_df["timestamp"].min()
            ).total_seconds()
            / 60,
            "max_depth_mm": float(seg_df["depth_proc_mm"].max()),
            "mean_depth_mm": float(seg_df["depth_proc_mm"].mean()),
            "min_depth_mm": float(seg_df["depth_proc_mm"].min()),
        }
        stats["segments"].append(seg_stats)

    return stats


def get_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of all sensors and their segment counts."""
    summary = (
        df.groupby("slug")
        .agg(
            num_segments=("segment_id", "max"),
            num_records=("segment_id", "count"),
            max_depth_mm=("depth_proc_mm", "max"),
            mean_depth_mm=("depth_proc_mm", "mean"),
        )
        .reset_index()
    )

    summary = summary.rename(columns={"slug": "sensor"})
    summary = summary.sort_values("num_segments", ascending=False).reset_index(
        drop=True
    )

    return summary


def filter_by_duration(
    df: pd.DataFrame,
    min_duration_minutes: float = 0,
    max_duration_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Filter segments by duration."""

    durations = (
        df.groupby(["slug", "segment_id"])["timestamp"]
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
        .reset_index()
    )
    durations.columns = ["slug", "segment_id", "duration"]

    valid = durations[durations["duration"] >= min_duration_minutes]

    if max_duration_minutes is not None:
        valid = valid[valid["duration"] <= max_duration_minutes]

    result = df.merge(valid[["slug", "segment_id"]], on=["slug", "segment_id"])

    return result.reset_index(drop=True)


def _plot_single_segment(args: Tuple) -> str:
    """Plot a single segment and save as PNG (worker function for parallel processing)."""
    times, depths, sensor, seg_id, duration, output_dir, figsize, dpi = args

    # Use non-interactive backend for parallel processing
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, depths, linewidth=1.5, color="#2196F3")
    ax.fill_between(times, depths, alpha=0.3, color="#2196F3")

    ax.set_xlabel("Time")
    ax.set_ylabel("Flood Depth (mm)")
    ax.set_title(f"{sensor} - Segment {seg_id} ({duration:.1f} min)")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    safe_sensor = sensor.replace("/", "_").replace("\\", "_")
    filename = f"{safe_sensor}_segment_{seg_id:04d}.png"
    filepath = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def visualize_segments(
    df: pd.DataFrame,
    output_dir: str = "visualizations/segments",
    min_duration_minutes: float = 10.0,
    figsize: Tuple[int, int] = (12, 4),
    dpi: int = 100,
    n_workers: Optional[int] = None,
) -> int:
    """Visualize segments as individual PNG plots using parallel processing."""
    os.makedirs(output_dir, exist_ok=True)

    if n_workers is None:
        n_workers = cpu_count()

    # Prepare tasks for parallel processing
    print(f"[1/3] Analyzing segments (min duration: {min_duration_minutes} min)...")
    tasks = []

    for sensor in df["slug"].unique():
        sensor_df = df[df["slug"] == sensor]

        for seg_id in sensor_df["segment_id"].unique():
            seg_df = sensor_df[sensor_df["segment_id"] == seg_id]

            duration = (
                seg_df["timestamp"].max() - seg_df["timestamp"].min()
            ).total_seconds() / 60

            if duration < min_duration_minutes:
                continue

            # Prepare serializable data for worker
            tasks.append(
                (
                    seg_df["timestamp"].tolist(),
                    seg_df["depth_proc_mm"].tolist(),
                    sensor,
                    int(seg_id),
                    duration,
                    output_dir,
                    figsize,
                    dpi,
                )
            )

    if not tasks:
        print(
            f"      No segments found with duration >= {min_duration_minutes} minutes"
        )
        return 0

    print(f"      Found {len(tasks)} segments to visualize")

    # Parallel processing
    print(f"[2/3] Generating plots with {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_plot_single_segment, tasks),
                total=len(tasks),
                desc="Saving plots",
            )
        )

    print(f"[3/3] Complete!")
    print(f"      Saved {len(results)} plots to {output_dir}")

    return len(results)
