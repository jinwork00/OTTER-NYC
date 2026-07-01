"""Create rainfall and flood-depth visualizations for event diagnostics."""

import os
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


def _plot_single_event(args: Tuple) -> str:
    """Plot a single flood event with dual Y-axis (worker function for parallel processing)."""
    (
        slug,
        segment_id,
        peak_time,
        peak_value,
        segment_times,
        segment_depths,
        precip_times,
        precip_values,
        output_dir,
        figsize,
        dpi,
    ) = args

    # Use non-interactive backend for parallel processing
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FONTSIZE = 20
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE + 2,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE - 2,
            "ytick.labelsize": FONTSIZE - 2,
            "legend.fontsize": FONTSIZE - 2,
        }
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    color_depth = "#8E44AD"
    ax1.set_xlabel("Time", fontsize=FONTSIZE)
    ax1.set_ylabel("Flood Depth (mm)", color=color_depth, fontsize=FONTSIZE)

    # Calculate max values for Y-axis scaling (prevent overlap)
    max_depth = max(segment_depths) if segment_depths else 100
    max_precip = max(precip_values) if precip_values else 10

    if segment_times and segment_depths:
        ax1.plot(
            segment_times,
            segment_depths,
            color=color_depth,
            linewidth=2,
            marker="o",
            markersize=5,
            label="Flood Depth",
        )
        ax1.tick_params(axis="y", labelcolor=color_depth, labelsize=FONTSIZE - 2)

        if segment_depths:
            peak_idx = segment_depths.index(max(segment_depths))
            ax1.scatter(
                [segment_times[peak_idx]],
                [max(segment_depths)],
                color="red",
                s=150,
                zorder=5,
                marker="*",
                label="Peak",
            )

    ax1.set_ylim(0, max_depth * 2.5)
    ax1.tick_params(axis="x", labelsize=FONTSIZE - 2, rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_rain = "#000080"
    ax2.set_ylabel("Rainfall (mm)", color=color_rain, fontsize=FONTSIZE)

    if precip_times and precip_values:
        if len(precip_times) > 1:
            import matplotlib.dates as mdates

            time_nums = mdates.date2num(precip_times)
            if len(time_nums) > 1:
                width = (time_nums[1] - time_nums[0]) * 0.8
            else:
                width = 0.005
        else:
            width = 0.005

        ax2.bar(
            precip_times,
            precip_values,
            width=width,
            color=color_rain,
            alpha=0.7,
            label="Rainfall",
        )

    ax2.set_ylim(0, max_precip * 2.5)
    ax2.tick_params(axis="y", labelcolor=color_rain, labelsize=FONTSIZE - 2)
    ax2.invert_yaxis()

    plt.title(
        f"{slug} - Segment {segment_id} | Peak: {peak_value:.1f}mm",
        fontsize=FONTSIZE + 2,
    )
    fig.autofmt_xdate()
    plt.tight_layout()

    safe_slug = slug.replace("/", "_").replace("\\", "_")
    filename = f"{safe_slug}_seg{segment_id:04d}_event.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def visualize_flood_events(
    df_peaks: pd.DataFrame,
    df_filtered: pd.DataFrame,
    df_precip: pd.DataFrame,
    output_dir: str = "visualizations/flood_events",
    valid_only: bool = True,
    max_plots: Optional[int] = None,
    lookback_hours: float = 2.0,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 100,
    n_workers: Optional[int] = None,
) -> int:
    """Visualize flood events with rainfall and flood depth on dual Y-axis."""

    if df_peaks is None or len(df_peaks) == 0:
        raise ValueError("No peaks data provided.")
    if df_filtered is None or len(df_filtered) == 0:
        raise ValueError("No filtered segment data provided.")
    if df_precip is None or len(df_precip) == 0:
        raise ValueError("No precipitation data provided.")

    df_peaks = df_peaks.copy()
    df_peaks["peak_time"] = pd.to_datetime(df_peaks["peak_time"], utc=True)
    df_peaks["start_time"] = pd.to_datetime(df_peaks["start_time"], utc=True)
    df_peaks["end_time"] = pd.to_datetime(df_peaks["end_time"], utc=True)
    df_filtered = df_filtered.copy()
    df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], utc=True)
    df_precip = df_precip.copy()
    df_precip["virtual_timestamp"] = pd.to_datetime(
        df_precip["virtual_timestamp"], utc=True
    )

    os.makedirs(output_dir, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, cpu_count() // 2)

    peaks = df_peaks.copy()
    if valid_only:
        peaks = peaks[peaks["is_valid_flood_event"]]

    if len(peaks) == 0:
        print("No events to visualize.")
        return 0

    if max_plots is not None:
        peaks = peaks.head(max_plots)

    print(f"[1/3] Preparing {len(peaks):,} events for visualization...")

    # Prepare tasks for parallel processing
    tasks = []
    lookback = pd.Timedelta(hours=lookback_hours)

    for _, peak in tqdm(peaks.iterrows(), total=len(peaks), desc="      Preparing"):
        seg_mask = (df_filtered["slug"] == peak["slug"]) & (
            df_filtered["segment_id"] == peak["segment_id"]
        )
        seg_data = df_filtered[seg_mask]

        if len(seg_data) == 0:
            continue

        window_start = peak["start_time"] - lookback
        window_end = peak["end_time"]

        precip_mask = (
            (df_precip["slug"] == peak["slug"])
            & (df_precip["virtual_timestamp"] >= window_start)
            & (df_precip["virtual_timestamp"] <= window_end)
        )
        precip_data = df_precip[precip_mask].sort_values("virtual_timestamp")

        tasks.append(
            (
                peak["slug"],
                int(peak["segment_id"]),
                peak["peak_time"],
                float(peak["peak_value"]),
                seg_data["timestamp"].tolist(),
                seg_data["depth_proc_mm"].tolist(),
                precip_data["virtual_timestamp"].tolist(),
                precip_data["precip"].tolist(),
                output_dir,
                figsize,
                dpi,
            )
        )

    if not tasks:
        print("No valid tasks to visualize.")
        return 0

    # Parallel processing
    print(f"[2/3] Generating {len(tasks):,} plots with {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_plot_single_event, tasks),
                total=len(tasks),
                desc="      Plotting",
            )
        )

    print("[3/3] Saving summary...")
    summary_df = peaks[
        [
            "slug",
            "segment_id",
            "peak_value",
            "peak_time",
            "duration_minutes",
            "is_valid_flood_event",
            "precip_1h_before",
        ]
    ].copy()
    summary_df["plot_file"] = [os.path.basename(r) for r in results]
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 50)
    print("Visualization complete:")
    print(f"  - Events visualized: {len(results):,}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Summary file: {summary_path}")
    print("=" * 50)

    return len(results)


# Shared data for sensor visualization workers
_sensor_shared = {}


def _init_sensor_worker(floodnet_dict, precip_dict, filtered_dict, valid_dict):

    global _sensor_shared
    _sensor_shared = {
        "floodnet": floodnet_dict,
        "precip": precip_dict,
        "filtered": filtered_dict,
        "valid": valid_dict,
    }


def _plot_sensor(args: Tuple) -> Optional[str]:
    """Plot a single sensor timeline with flood depth and rainfall."""
    slug, rank, max_change, output_dir, figsize, dpi = args

    # Get data from shared memory (O(1) lookup)
    sensor_df = _sensor_shared["floodnet"].get(slug)
    if sensor_df is None or len(sensor_df) == 0:
        return None

    sensor_df = sensor_df.sort_values("timestamp")
    times = sensor_df["timestamp"].values
    depths = sensor_df["depth_proc_mm"].values

    t_min, t_max = sensor_df["timestamp"].min(), sensor_df["timestamp"].max()
    precip_df = _sensor_shared["precip"].get(slug)

    if precip_df is not None and len(precip_df) > 0:
        mask = (precip_df["virtual_timestamp"] >= t_min) & (
            precip_df["virtual_timestamp"] <= t_max
        )
        precip_df = precip_df[mask].sort_values("virtual_timestamp")
        precip_times = precip_df["virtual_timestamp"].values
        precip_values = precip_df["precip"].values
    else:
        precip_times = np.array([])
        precip_values = np.array([])

    filtered_ranges = _sensor_shared["filtered"].get(slug, [])
    valid_ranges = _sensor_shared["valid"].get(slug, [])

    # Import plotting libraries inside worker
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    FONTSIZE = 20
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE + 2,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE - 2,
            "ytick.labelsize": FONTSIZE - 2,
            "legend.fontsize": FONTSIZE - 2,
        }
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    for start, end in filtered_ranges:
        ax1.axvspan(start, end, alpha=0.3, color="red")
    for start, end in valid_ranges:
        ax1.axvspan(start, end, alpha=0.3, color="green")

    color_depth = "#8E44AD"
    ax1.set_xlabel("Time", fontsize=FONTSIZE)
    ax1.set_ylabel("Flood Depth (mm)", color=color_depth, fontsize=FONTSIZE)

    max_depth = depths.max() if len(depths) > 0 and depths.max() > 0 else 100
    max_precip = (
        precip_values.max()
        if len(precip_values) > 0 and precip_values.max() > 0
        else 10
    )

    ax1.plot(times, depths, color=color_depth, linewidth=1.5, label="Flood Depth")
    ax1.tick_params(axis="y", labelcolor=color_depth, labelsize=FONTSIZE - 2)
    ax1.set_ylim(0, max_depth * 1.5)
    ax1.tick_params(axis="x", labelsize=FONTSIZE - 2, rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_rain = "#000080"
    ax2.set_ylabel("Rainfall (mm)", color=color_rain, fontsize=FONTSIZE)

    if len(precip_times) > 0:
        ax2.fill_between(
            precip_times,
            0,
            precip_values,
            step="pre",
            color=color_rain,
            alpha=0.7,
            label="Rainfall",
        )

    ax2.set_ylim(0, max_precip * 2.5)
    ax2.tick_params(axis="y", labelcolor=color_rain, labelsize=FONTSIZE - 2)
    ax2.invert_yaxis()

    legend_elements = [
        plt.Line2D([0], [0], color=color_depth, linewidth=2, label="Flood Depth")
    ]
    if filtered_ranges:
        legend_elements.append(
            Patch(facecolor="red", alpha=0.3, label="Filtered Segments")
        )
    if valid_ranges:
        legend_elements.append(
            Patch(facecolor="green", alpha=0.3, label="Valid Flood Events")
        )
    if len(legend_elements) > 1:
        ax1.legend(handles=legend_elements, loc="upper left", fontsize=FONTSIZE - 4)

    plt.title(f"{slug} - Sensor Timeline", fontsize=FONTSIZE + 2)
    fig.autofmt_xdate()
    plt.tight_layout()

    safe_slug = slug.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(
        output_dir, f"{rank:03d}_{safe_slug}_{max_change:.1f}mm.png"
    )
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _extract_segment_ranges(df: pd.DataFrame) -> Dict[str, List[Tuple]]:
    """Extract (start, end) time ranges per sensor using vectorized groupby."""
    ranges = (
        df.groupby(["slug", "segment_id"])
        .agg(start=("timestamp", "min"), end=("timestamp", "max"))
        .reset_index()
        .groupby("slug")[["start", "end"]]
        .apply(lambda g: list(zip(g["start"], g["end"])), include_groups=False)
        .to_dict()
    )
    return ranges


def visualize_sensors(
    df_floodnet: pd.DataFrame,
    df_filtered: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_precip: pd.DataFrame,
    output_dir: str = "visualizations/sensor_timelines",
    max_plots: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 8),
    dpi: int = 100,
    n_workers: Optional[int] = None,
) -> int:
    """Visualize flood depth time series per sensor with segment highlighting."""

    for name, df in [
        ("floodnet", df_floodnet),
        ("filtered", df_filtered),
        ("valid", df_valid),
        ("precip", df_precip),
    ]:
        if df is None or len(df) == 0:
            raise ValueError(f"No {name} data provided.")

    df_floodnet = df_floodnet.copy()
    df_floodnet["timestamp"] = pd.to_datetime(df_floodnet["timestamp"], utc=True)
    df_filtered = df_filtered.copy()
    df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], utc=True)
    df_valid = df_valid.copy()
    df_valid["timestamp"] = pd.to_datetime(df_valid["timestamp"], utc=True)
    df_precip = df_precip.copy()
    df_precip["virtual_timestamp"] = pd.to_datetime(
        df_precip["virtual_timestamp"], utc=True
    )

    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or max(1, cpu_count() // 2)

    print("[1/3] Calculating max absolute depth change per sensor...")

    df_sorted = df_floodnet.sort_values(["slug", "timestamp"]).copy()
    df_sorted["depth_change"] = df_sorted.groupby("slug")["depth_proc_mm"].diff()
    max_abs_change = df_sorted.groupby("slug")["depth_change"].apply(
        lambda x: x.abs().max() if len(x.dropna()) > 0 else 0.0
    )

    sensor_ranking = max_abs_change.sort_values(ascending=False).reset_index()
    sensor_ranking.columns = ["slug", "max_abs_change"]
    sensor_ranking["rank"] = range(1, len(sensor_ranking) + 1)
    sensor_ranking = sensor_ranking.set_index("slug")

    print(f"      Top 5 sensors by max change: {list(sensor_ranking.head().index)}")

    print("[2/3] Preparing data...")

    # Pre-group by slug (O(1) lookup in workers)
    floodnet_dict = dict(iter(df_floodnet.groupby("slug")))
    precip_dict = dict(iter(df_precip.groupby("slug")))

    filtered_dict = _extract_segment_ranges(df_filtered)
    valid_dict = _extract_segment_ranges(df_valid)

    ranked_slugs = sensor_ranking.sort_values("rank").index.tolist()

    slugs = [s for s in ranked_slugs if s in floodnet_dict]
    if max_plots is not None:
        slugs = slugs[:max_plots]

    tasks = [
        (
            slug,
            int(sensor_ranking.loc[slug, "rank"]),
            float(sensor_ranking.loc[slug, "max_abs_change"]),
            output_dir,
            figsize,
            dpi,
        )
        for slug in slugs
    ]
    print(
        f"      {len(slugs)} sensors, {sum(len(v) for v in filtered_dict.values())} filtered, "
        f"{sum(len(v) for v in valid_dict.values())} valid segments"
    )

    # Single-stage parallel processing
    print(f"[3/3] Generating {len(tasks)} plots with {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_sensor_worker,
        initargs=(floodnet_dict, precip_dict, filtered_dict, valid_dict),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_plot_sensor, tasks),
                total=len(tasks),
                desc="      Plotting",
            )
        )

    results = [r for r in results if r is not None]

    summary_data = [
        {
            "rank": int(sensor_ranking.loc[slug, "rank"]),
            "slug": slug,
            "max_abs_change_mm": float(sensor_ranking.loc[slug, "max_abs_change"]),
            "records": len(floodnet_dict[slug]),
            "filtered": len(filtered_dict.get(slug, [])),
            "valid": len(valid_dict.get(slug, [])),
            "start": floodnet_dict[slug]["timestamp"].min(),
            "end": floodnet_dict[slug]["timestamp"].max(),
            "filename": f"{int(sensor_ranking.loc[slug, 'rank']):03d}_{slug.replace('/', '_').replace(chr(92), '_')}_{float(sensor_ranking.loc[slug, 'max_abs_change']):.1f}mm.png",
        }
        for slug in slugs
        if slug in floodnet_dict
    ]

    summary_path = os.path.join(output_dir, "sensor_summary.csv")
    pd.DataFrame(summary_data).sort_values("rank").to_csv(summary_path, index=False)

    print()
    print("=" * 50)
    print(f"Sensor visualization complete: {len(results)} plots")
    print(f"Output: {output_dir}")
    print("=" * 50)

    return len(results)


# Shared data for segment visualization workers
_segment_shared = {}


def _init_segment_worker(valid_dict, precip_dict, lookback_hours):

    global _segment_shared
    _segment_shared = {
        "valid": valid_dict,
        "precip": precip_dict,
        "lookback": pd.Timedelta(hours=lookback_hours),
    }


def _plot_segment(args: Tuple) -> Optional[str]:
    """Plot a single flood segment with rainfall."""
    (
        slug,
        segment_id,
        start_time,
        end_time,
        duration_minutes,
        peak_value,
        output_path,
        figsize,
        dpi,
    ) = args

    # Get segment data from pre-grouped dict (O(1) lookup)
    valid_df = _segment_shared["valid"].get(slug)
    if valid_df is None:
        return None

    seg_data = valid_df[valid_df["segment_id"] == segment_id].sort_values("timestamp")
    if len(seg_data) == 0:
        return None

    times = seg_data["timestamp"].values
    depths = seg_data["depth_proc_mm"].values

    lookback = _segment_shared["lookback"]
    window_start = start_time - lookback
    precip_df = _segment_shared["precip"].get(slug)

    if precip_df is not None and len(precip_df) > 0:
        mask = (precip_df["virtual_timestamp"] >= window_start) & (
            precip_df["virtual_timestamp"] <= end_time
        )
        precip_df = precip_df[mask].sort_values("virtual_timestamp")
        precip_times = precip_df["virtual_timestamp"].values
        precip_values = precip_df["precip"].values
    else:
        precip_times = np.array([])
        precip_values = np.array([])

    # Import plotting libraries inside worker
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FONTSIZE = 20
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE + 2,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE - 2,
            "ytick.labelsize": FONTSIZE - 2,
            "legend.fontsize": FONTSIZE - 2,
        }
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    color_depth = "#8E44AD"
    ax1.set_xlabel("Time", fontsize=FONTSIZE)
    ax1.set_ylabel("Flood Depth (mm)", color=color_depth, fontsize=FONTSIZE)

    max_depth = depths.max() if len(depths) > 0 else 100
    max_precip = precip_values.max() if len(precip_values) > 0 else 10

    ax1.plot(
        times,
        depths,
        color=color_depth,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Flood Depth",
    )
    ax1.tick_params(axis="y", labelcolor=color_depth, labelsize=FONTSIZE - 2)

    peak_idx = depths.argmax()
    ax1.scatter(
        [times[peak_idx]],
        [depths[peak_idx]],
        color="red",
        s=150,
        zorder=5,
        marker="*",
        label="Peak",
    )

    ax1.set_ylim(0, max_depth * 2.5)
    ax1.tick_params(axis="x", labelsize=FONTSIZE - 2, rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_rain = "#000080"
    ax2.set_ylabel("Rainfall (mm)", color=color_rain, fontsize=FONTSIZE)

    if len(precip_times) > 0:
        ax2.fill_between(
            precip_times,
            0,
            precip_values,
            step="pre",
            color=color_rain,
            alpha=0.7,
            label="Rainfall",
        )

    ax2.set_ylim(0, max_precip * 2.5)
    ax2.tick_params(axis="y", labelcolor=color_rain, labelsize=FONTSIZE - 2)
    ax2.invert_yaxis()

    plt.title(
        f"{slug} - Segment {segment_id} | Duration: {duration_minutes:.0f}min | Peak: {peak_value:.1f}mm",
        fontsize=FONTSIZE + 2,
    )
    fig.autofmt_xdate()
    plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def visualize_segments_by_sensor(
    df_valid: pd.DataFrame,
    df_precip: pd.DataFrame,
    output_dir: str = "visualizations/segments",
    min_duration_minutes: float = 30.0,
    lookback_hours: float = 1.0,
    max_plots: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100,
    n_workers: Optional[int] = None,
) -> int:
    """Visualize flood segments organized by sensor folders."""

    if df_valid is None or len(df_valid) == 0:
        raise ValueError("No valid segment data provided.")
    if df_precip is None or len(df_precip) == 0:
        raise ValueError("No precipitation data provided.")

    df_valid = df_valid.copy()
    df_valid["timestamp"] = pd.to_datetime(df_valid["timestamp"], utc=True)
    df_precip = df_precip.copy()
    df_precip["virtual_timestamp"] = pd.to_datetime(
        df_precip["virtual_timestamp"], utc=True
    )

    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or max(1, cpu_count() // 2)

    print(f"[1/3] Analyzing segments (min: {min_duration_minutes:.0f} min)...")
    seg_info = (
        df_valid.groupby(["slug", "segment_id"])
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            peak_value=("depth_proc_mm", "max"),
        )
        .reset_index()
    )
    seg_info["duration_minutes"] = (
        seg_info["end_time"] - seg_info["start_time"]
    ).dt.total_seconds() / 60
    seg_info = seg_info[seg_info["duration_minutes"] >= min_duration_minutes]

    if len(seg_info) == 0:
        print(f"      No segments >= {min_duration_minutes:.0f} min")
        return 0

    if max_plots is not None:
        seg_info = seg_info.head(max_plots)

    print(f"      {len(seg_info):,} segments to visualize")

    # Pre-group data by slug (O(1) lookup in workers)
    valid_dict = dict(iter(df_valid.groupby("slug")))
    precip_dict = dict(iter(df_precip.groupby("slug")))

    print(f"[2/3] Preparing {len(seg_info):,} tasks...")
    sensors = seg_info["slug"].unique()
    for slug in sensors:
        safe_slug = slug.replace("/", "_").replace("\\", "_")
        os.makedirs(os.path.join(output_dir, safe_slug), exist_ok=True)

    tasks = []
    for _, row in seg_info.iterrows():
        safe_slug = row["slug"].replace("/", "_").replace("\\", "_")
        output_path = os.path.join(
            output_dir, safe_slug, f"segment_{row['segment_id']:04d}.png"
        )
        tasks.append(
            (
                row["slug"],
                int(row["segment_id"]),
                row["start_time"],
                row["end_time"],
                float(row["duration_minutes"]),
                float(row["peak_value"]),
                output_path,
                figsize,
                dpi,
            )
        )

    # Single-stage parallel processing
    print(f"[3/3] Generating {len(tasks)} plots with {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_segment_worker,
        initargs=(valid_dict, precip_dict, lookback_hours),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_plot_segment, tasks),
                total=len(tasks),
                desc="      Plotting",
            )
        )

    results = [r for r in results if r]

    for slug in sensors:
        safe_slug = slug.replace("/", "_").replace("\\", "_")
        sensor_dir = os.path.join(output_dir, safe_slug)
        sensor_segs = seg_info[seg_info["slug"] == slug].copy()
        sensor_segs["plot_file"] = [
            f"segment_{sid:04d}.png" for sid in sensor_segs["segment_id"]
        ]
        sensor_segs.to_csv(os.path.join(sensor_dir, "summary.csv"), index=False)

    seg_info["folder"] = seg_info["slug"].str.replace("/", "_").str.replace("\\", "_")
    seg_info.to_csv(os.path.join(output_dir, "overall_summary.csv"), index=False)

    print()
    print("=" * 60)
    print(f"Segment visualization complete: {len(results)} plots")
    print(f"Sensors: {len(sensors)}, Output: {output_dir}")
    print("=" * 60)

    return len(results)


# Shared data for removed segment visualization workers
_removed_shared = {}


def _init_removed_worker(removed_dict):

    global _removed_shared
    _removed_shared = {"removed": removed_dict}


def _plot_removed_sensor(args: Tuple) -> Optional[str]:
    """Plot all removed segments for a single sensor (concatenated by time)."""
    slug, rank, max_depth, num_segments, output_path, figsize, dpi = args

    # Get removed segment data from pre-grouped dict (O(1) lookup)
    sensor_df = _removed_shared["removed"].get(slug)
    if sensor_df is None or len(sensor_df) == 0:
        return None

    sensor_df = sensor_df.sort_values("timestamp")
    times = sensor_df["timestamp"].values
    depths = sensor_df["depth_proc_mm"].values

    # Import plotting libraries inside worker
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FONTSIZE = 20
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE + 2,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE - 2,
            "ytick.labelsize": FONTSIZE - 2,
            "legend.fontsize": FONTSIZE - 2,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    color_depth = "#8E44AD"
    ax.set_xlabel("Time", fontsize=FONTSIZE)
    ax.set_ylabel("Flood Depth (mm)", color=color_depth, fontsize=FONTSIZE)

    max_y = depths.max() if len(depths) > 0 and depths.max() > 0 else 100

    ax.plot(times, depths, color=color_depth, linewidth=1.5, label="Flood Depth")
    ax.tick_params(axis="y", labelcolor=color_depth, labelsize=FONTSIZE - 2)
    ax.set_ylim(0, max_y * 1.3)
    ax.tick_params(axis="x", labelsize=FONTSIZE - 2, rotation=45)
    ax.grid(True, alpha=0.3)

    plt.title(
        f"{slug} - Removed Segments ({num_segments} segments) | Max: {max_depth:.1f}mm",
        fontsize=FONTSIZE + 2,
    )
    fig.autofmt_xdate()
    plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def visualize_removed_segments(
    df_segments: pd.DataFrame,
    df_filtered: pd.DataFrame,
    output_dir: str = "visualizations/removed_segments",
    max_plots: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 8),
    dpi: int = 100,
    n_workers: Optional[int] = None,
) -> int:
    """Visualize segments that were removed during filtering."""

    if df_segments is None or len(df_segments) == 0:
        raise ValueError("No segment data provided.")
    if df_filtered is None or len(df_filtered) == 0:
        raise ValueError("No filtered segment data provided.")

    df_segments = df_segments.copy()
    df_segments["timestamp"] = pd.to_datetime(df_segments["timestamp"], utc=True)
    df_filtered = df_filtered.copy()
    df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], utc=True)

    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or max(1, cpu_count() // 2)

    # Find removed segments using time-range matching (segment_id is renumbered after filtering)
    print("[1/3] Identifying removed segments (time-based matching)...")

    seg_starts = (
        df_segments.groupby(["slug", "segment_id"])["timestamp"].min().reset_index()
    )
    seg_starts["_timekey"] = (
        seg_starts["slug"] + "_" + seg_starts["timestamp"].astype(str)
    )

    filtered_starts = (
        df_filtered.groupby(["slug", "segment_id"])["timestamp"].min().reset_index()
    )
    filtered_timekeys = set(
        filtered_starts["slug"] + "_" + filtered_starts["timestamp"].astype(str)
    )

    removed_mask = ~seg_starts["_timekey"].isin(filtered_timekeys)
    removed_seg_keys = set(
        zip(
            seg_starts.loc[removed_mask, "slug"],
            seg_starts.loc[removed_mask, "segment_id"],
        )
    )

    print(f"      Total segments: {len(seg_starts):,}")
    print(f"      Filtered segments: {len(filtered_starts):,}")
    print(f"      Removed segments: {len(removed_seg_keys):,}")

    if len(removed_seg_keys) == 0:
        print("      No removed segments to visualize.")
        return 0

    df_segments["_key"] = list(zip(df_segments["slug"], df_segments["segment_id"]))
    df_removed = df_segments[df_segments["_key"].isin(removed_seg_keys)].drop(
        columns=["_key"]
    )

    print(f"      Removed records: {len(df_removed):,}")

    print("[2/3] Preparing data...")
    sensor_stats = (
        df_removed.groupby("slug")
        .agg(
            max_depth=("depth_proc_mm", "max"),
            num_segments=("segment_id", "nunique"),
            num_records=("timestamp", "count"),
        )
        .reset_index()
    )

    sensor_stats = sensor_stats.sort_values("max_depth", ascending=False).reset_index(
        drop=True
    )
    sensor_stats["rank"] = range(1, len(sensor_stats) + 1)

    if max_plots is not None:
        sensor_stats = sensor_stats.head(max_plots)

    print(f"      {len(sensor_stats)} sensors with removed segments")

    removed_dict = dict(iter(df_removed.groupby("slug")))

    tasks = []
    for _, row in sensor_stats.iterrows():
        safe_slug = row["slug"].replace("/", "_").replace("\\", "_")
        output_path = os.path.join(
            output_dir, f"{row['rank']:03d}_{safe_slug}_{row['max_depth']:.1f}mm.png"
        )
        tasks.append(
            (
                row["slug"],
                int(row["rank"]),
                float(row["max_depth"]),
                int(row["num_segments"]),
                output_path,
                figsize,
                dpi,
            )
        )

    # Parallel processing
    print(f"[3/3] Generating {len(tasks)} plots with {n_workers} workers...")

    with Pool(
        n_workers, initializer=_init_removed_worker, initargs=(removed_dict,)
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_plot_removed_sensor, tasks),
                total=len(tasks),
                desc="      Plotting",
            )
        )

    results = [r for r in results if r is not None]

    summary_path = os.path.join(output_dir, "removed_segments_summary.csv")
    sensor_stats["filename"] = [
        f"{r:03d}_{s.replace('/', '_').replace(chr(92), '_')}_{d:.1f}mm.png"
        for r, s, d in zip(
            sensor_stats["rank"], sensor_stats["slug"], sensor_stats["max_depth"]
        )
    ]
    sensor_stats.to_csv(summary_path, index=False)

    print()
    print("=" * 60)
    print(f"Removed segment visualization complete: {len(results)} plots")
    print(f"Output: {output_dir}")
    print("=" * 60)

    return len(results)


# Shared data for interpolation visualization workers
_interpolation_shared = {}


def _init_interpolation_worker(before_dict, after_dict):

    global _interpolation_shared
    _interpolation_shared = {"before": before_dict, "after": after_dict}


def _plot_interpolation_segment(args: Tuple) -> Optional[str]:
    """Plot a single segment comparing before/after interpolation (2x1 subplot)."""
    (slug, new_segment_id, interpolated_count, rank, output_path, figsize, dpi) = args

    # Get segment data from pre-grouped dicts (O(1) lookup)
    before_df = _interpolation_shared["before"].get(slug)
    after_df = _interpolation_shared["after"].get(slug)

    if before_df is None or after_df is None:
        return None

    seg_before = before_df[before_df["new_segment_id"] == new_segment_id].sort_values(
        "timestamp"
    )
    seg_after = after_df[after_df["new_segment_id"] == new_segment_id].sort_values(
        "timestamp"
    )

    if len(seg_before) == 0 or len(seg_after) == 0:
        return None

    times = seg_before["timestamp"].values
    depths_before = seg_before["depth_proc_mm"].values
    depths_after = seg_after["depth_proc_mm"].values

    # Find zero points and interpolated points
    zero_mask = depths_before == 0
    interp_mask = zero_mask & (depths_after > 0)

    # Import plotting libraries inside worker
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    FONTSIZE = 20
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE + 2,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE - 2,
            "ytick.labelsize": FONTSIZE - 2,
            "legend.fontsize": FONTSIZE - 2,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: Before interpolation
    ax1 = axes[0]
    color_line = "#3498DB"
    color_zero = "#E74C3C"

    ax1.plot(times, depths_before, color=color_line, linewidth=1.5, label="Depth")

    zero_times = times[zero_mask]
    zero_depths = depths_before[zero_mask]
    if len(zero_times) > 0:
        ax1.scatter(
            zero_times,
            zero_depths,
            color=color_zero,
            s=40,
            zorder=5,
            label=f"Zero values (n={len(zero_times)})",
        )

    ax1.set_ylabel("Depth (mm)", fontsize=FONTSIZE)
    ax1.set_title(
        f"BEFORE Interpolation - {slug}", fontsize=FONTSIZE + 2, fontweight="bold"
    )
    ax1.legend(loc="upper right", fontsize=FONTSIZE - 4)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis="y", labelsize=FONTSIZE - 2)

    # Bottom: After interpolation
    ax2 = axes[1]
    color_line_after = "#27AE60"
    color_interp = "#F39C12"

    ax2.plot(
        times,
        depths_after,
        color=color_line_after,
        linewidth=1.5,
        label="Depth (interpolated)",
    )

    # Highlight interpolated points with orange dots
    interp_times = times[interp_mask]
    interp_depths = depths_after[interp_mask]
    if len(interp_times) > 0:
        ax2.scatter(
            interp_times,
            interp_depths,
            color=color_interp,
            s=40,
            zorder=5,
            label=f"Interpolated (n={len(interp_times)})",
        )

    ax2.set_ylabel("Depth (mm)", fontsize=FONTSIZE)
    ax2.set_xlabel("Time", fontsize=FONTSIZE)
    ax2.set_title(f"AFTER Interpolation", fontsize=FONTSIZE + 2, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=FONTSIZE - 4)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="y", labelsize=FONTSIZE - 2)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.tick_params(axis="x", labelsize=FONTSIZE - 2, rotation=45)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def visualize_interpolation_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_dir: str = "visualizations/interpolation",
    top_n: int = 5,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150,
    n_workers: Optional[int] = None,
) -> int:
    """Visualize before/after interpolation comparison for top N segments."""

    if df_before is None or len(df_before) == 0:
        raise ValueError("No 'before' data provided.")
    if df_after is None or len(df_after) == 0:
        raise ValueError("No 'after' data provided.")

    required_cols = {"slug", "new_segment_id", "timestamp", "depth_proc_mm"}
    for name, df in [("df_before", df_before), ("df_after", df_after)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    df_before = df_before.copy()
    df_after = df_after.copy()
    df_before["timestamp"] = pd.to_datetime(df_before["timestamp"])
    df_after["timestamp"] = pd.to_datetime(df_after["timestamp"])

    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or max(1, cpu_count() // 2)

    print(f"[1/3] Calculating interpolated counts per segment...")

    df_before_aligned = df_before.reset_index(drop=True)
    df_after_aligned = df_after.reset_index(drop=True)

    was_zero = df_before_aligned["depth_proc_mm"] == 0
    now_positive = df_after_aligned["depth_proc_mm"] > 0
    interpolated_mask = was_zero & now_positive

    df_before_aligned["_was_interpolated"] = interpolated_mask

    # Count interpolated values per segment
    segment_interp_counts = (
        df_before_aligned.groupby(["slug", "new_segment_id"])["_was_interpolated"]
        .sum()
        .reset_index(name="interpolated_count")
    )

    # Select top N segments by interpolated count

    if top_n is None:
        top_n = len(segment_interp_counts)
    top_segments = segment_interp_counts.nlargest(top_n, "interpolated_count")
    top_segments["rank"] = range(1, len(top_segments) + 1)

    print(f"      Top {len(top_segments)} segments selected")
    print(f"      Interpolated counts: {top_segments['interpolated_count'].tolist()}")

    if len(top_segments) == 0:
        print("      No segments to visualize.")
        return 0

    # Pre-group data by slug (O(1) lookup in workers)
    print(f"[2/3] Preparing data...")
    before_dict = dict(iter(df_before.groupby("slug")))
    after_dict = dict(iter(df_after.groupby("slug")))

    tasks = []
    for _, row in top_segments.iterrows():
        safe_slug = row["slug"].replace("/", "_").replace("\\", "_")
        output_path = os.path.join(
            output_dir,
            f"{row['rank']:02d}_{safe_slug}_seg{row['new_segment_id']}_interp{row['interpolated_count']}.png",
        )
        tasks.append(
            (
                row["slug"],
                int(row["new_segment_id"]),
                int(row["interpolated_count"]),
                int(row["rank"]),
                output_path,
                figsize,
                dpi,
            )
        )

    # Parallel processing
    print(f"[3/3] Generating {len(tasks)} plots with {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_interpolation_worker,
        initargs=(before_dict, after_dict),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(_plot_interpolation_segment, tasks),
                total=len(tasks),
                desc="      Plotting",
            )
        )

    results = [r for r in results if r is not None]

    summary_path = os.path.join(output_dir, "interpolation_summary.csv")
    top_segments["filename"] = [
        f"{r:02d}_{s.replace('/', '_').replace(chr(92), '_')}_seg{seg}_interp{c}.png"
        for r, s, seg, c in zip(
            top_segments["rank"],
            top_segments["slug"],
            top_segments["new_segment_id"],
            top_segments["interpolated_count"],
        )
    ]
    top_segments.to_csv(summary_path, index=False)

    print()
    print("=" * 60)
    print(f"Interpolation comparison visualization complete: {len(results)} plots")
    print(f"Output: {output_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 60)

    return len(results)
