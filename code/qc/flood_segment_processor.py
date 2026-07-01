"""Filter flood segments, detect peaks, and validate events against rainfall."""

import json
import os
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# Shared read-only state initialized once per validation worker.
_validate_worker_data = {}


def _init_validate_worker(precip_sorted: pd.DataFrame, lookback: pd.Timedelta):
    """Worker initializer for _validate_sensor_peaks."""
    global _validate_worker_data
    _validate_worker_data["precip_sorted"] = precip_sorted
    _validate_worker_data["lookback"] = lookback


def _validate_sensor_peaks(args: Tuple) -> List[Dict]:
    """Validate peaks for a single sensor (worker function for parallel processing)."""
    slug, slug_peaks_records = args

    precip_sorted = _validate_worker_data["precip_sorted"]
    lookback = _validate_worker_data["lookback"]

    slug_precip = precip_sorted[precip_sorted["slug"] == slug]

    results = []

    if len(slug_precip) == 0:
        for peak in slug_peaks_records:
            results.append(
                {**peak, "is_valid_flood_event": False, "precip_1h_before": 0.0}
            )
        return results

    for peak in slug_peaks_records:
        peak_time = peak["peak_time"]
        window_start = peak_time - lookback

        mask = (slug_precip["virtual_timestamp"] >= window_start) & (
            slug_precip["virtual_timestamp"] <= peak_time
        )
        precip_window = slug_precip[mask]

        is_valid = (
            (precip_window["precip"] > 0).any() if len(precip_window) > 0 else False
        )
        precip_sum = precip_window["precip"].sum() if len(precip_window) > 0 else 0.0

        results.append(
            {
                **peak,
                "is_valid_flood_event": bool(is_valid),
                "precip_1h_before": float(precip_sum),
            }
        )

    return results


@dataclass
class ProcessingConfig:
    """Configuration for flood segment processing."""

    sensor_limit_mm: float = 3000.0

    filter_method: str = "laplace"

    change_quantile: float = 0.99

    mad_sigma: float = 3.5

    sigma_n: float = 3.5

    # A dedicated precipitation file takes priority over the integrated dataset.
    rainfall_lookback_hours: float = 3.0
    # The legacy file uses virtual_timestamp; the integrated file uses timestamp.
    precip_path: Optional[str] = None
    integrated_path: Optional[str] = None

    output_dir: str = "data/segments"
    filtered_segments_filename: str = "flood_segments_filtered.csv"
    peak_info_filename: str = "peak_info.csv"
    stats_filename: str = "processing_stats.json"


class FloodSegmentProcessor:
    """Processes extracted flood segments with filtering, peak detection, and rainfall validation."""

    def __init__(self, config: Optional[ProcessingConfig] = None):

        self.config = config or ProcessingConfig()

        self.df_segments: Optional[pd.DataFrame] = None
        self.df_filtered: Optional[pd.DataFrame] = None
        self.df_peaks: Optional[pd.DataFrame] = None
        self.df_precip: Optional[pd.DataFrame] = None

        self.stats: Dict = {}
        self.change_threshold: Optional[float] = None

    def load_segments(self, path: str) -> "FloodSegmentProcessor":
        """Load flood segments CSV file."""
        print(f"Loading segments from {path}...")
        self.df_segments = pd.read_csv(
            path, parse_dates=["timestamp"], engine="pyarrow"
        )
        print(f"      Loaded {len(self.df_segments):,} records")
        self.stats["original_records"] = len(self.df_segments)
        return self

    def _load_precipitation(self) -> None:
        """Load precipitation data for rainfall validation."""
        if self.df_precip is not None:
            return

        if self.config.precip_path is not None:
            if not os.path.exists(self.config.precip_path):
                raise FileNotFoundError(
                    f"Precipitation file not found: {self.config.precip_path}"
                )
            self.df_precip = pd.read_csv(
                self.config.precip_path,
                parse_dates=["virtual_timestamp"],
                engine="pyarrow",
            )
            return

        src = self.config.integrated_path
        if src is None:
            raise ValueError(
                "No precipitation source. Set ProcessingConfig.precip_path (a dedicated precip file "
                "with columns slug, virtual_timestamp, precip) OR ProcessingConfig.integrated_path "
                "(the cleaned_integrated file with a 'precip' column)."
            )
        if not os.path.exists(src):
            raise FileNotFoundError(f"integrated_path not found: {src}")

        cols = ["slug", "timestamp", "precip"]
        if src.endswith(".parquet"):
            df = pd.read_parquet(src, columns=cols, filters=[("precip", ">", 0)])
        else:
            df = pd.read_csv(src, usecols=cols, engine="pyarrow")
            df = df[df["precip"] > 0]

        self.df_precip = df.rename(columns={"timestamp": "virtual_timestamp"})

    def calculate_change_threshold(self) -> float:
        """Calculate the change threshold using one of four methods."""
        print("[1/4] Calculating change threshold...")

        if self.df_segments is None:
            raise ValueError("No segments loaded. Call load_segments() first.")

        df_sorted = self.df_segments.sort_values(["slug", "timestamp"])
        depth_change = df_sorted.groupby("slug")["depth_proc_mm"].diff()

        nonzero_mask = (depth_change != 0) & depth_change.notna()
        nonzero_signed = depth_change[nonzero_mask].to_numpy(dtype="float64")
        nonzero_abs = np.abs(nonzero_signed)

        method = self.config.filter_method

        if method == "mad":
            median_val = float(np.median(nonzero_abs))
            mad = float(np.median(np.abs(nonzero_abs - median_val)))

            self.change_threshold = median_val + (self.config.mad_sigma * mad * 1.4826)
            print(f"      Method: MAD (sigma={self.config.mad_sigma})")
            print(f"      Median: {median_val:.2f} mm, MAD: {mad:.2f} mm")
            self.stats["filter_method"] = "mad"
            self.stats["mad_sigma"] = self.config.mad_sigma
            self.stats["median_change_mm"] = median_val
            self.stats["mad_mm"] = mad

        elif method == "gaussian":
            sigma = float(np.std(nonzero_signed, ddof=1))
            self.change_threshold = self.config.sigma_n * sigma
            print(f"      Method: Gaussian (n={self.config.sigma_n}·σ)")
            print(f"      σ: {sigma:.2f} mm")
            self.stats["filter_method"] = "gaussian"
            self.stats["sigma_n"] = float(self.config.sigma_n)
            self.stats["gaussian_sigma_mm"] = sigma

        elif method == "laplace":
            median_signed = float(np.median(nonzero_signed))
            b = float(np.mean(np.abs(nonzero_signed - median_signed)))
            sigma_eq = b * np.sqrt(2)
            self.change_threshold = self.config.sigma_n * sigma_eq
            print(f"      Method: Laplace (n={self.config.sigma_n}·σ_eq)")
            print(f"      b: {b:.2f} mm, σ_eq = b·√2 = {sigma_eq:.2f} mm")
            self.stats["filter_method"] = "laplace"
            self.stats["sigma_n"] = float(self.config.sigma_n)
            self.stats["laplace_b_mm"] = b
            self.stats["laplace_sigma_eq_mm"] = sigma_eq

        elif method == "quantile":
            quantile_pct = self.config.change_quantile * 100
            self.change_threshold = float(np.percentile(nonzero_abs, quantile_pct))
            print(f"      Method: Quantile ({self.config.change_quantile:.0%})")
            self.stats["filter_method"] = "quantile"
            self.stats["change_quantile"] = self.config.change_quantile

        else:
            raise ValueError(
                f"Unknown filter_method: {method!r}. "
                f"Expected one of: 'quantile', 'mad', 'gaussian', 'laplace'."
            )

        print(f"      Non-zero changes: {len(nonzero_abs):,}")
        print(f"      Threshold: {self.change_threshold:.2f} mm")

        self.stats["change_threshold_mm"] = float(self.change_threshold)
        self.stats["nonzero_changes_count"] = int(len(nonzero_abs))

        return self.change_threshold

    def apply_filters(self) -> "FloodSegmentProcessor":
        """Apply threshold-based filters to segments."""
        print("[2/4] Applying filters (segment-level)...")

        if self.df_segments is None:
            raise ValueError("No segments loaded. Call load_segments() first.")
        if self.change_threshold is None:
            raise ValueError(
                "No threshold calculated. Call calculate_change_threshold() first."
            )

        n_original_records = len(self.df_segments)
        n_original_segments = self.df_segments.groupby(["slug", "segment_id"]).ngroups

        df_sorted = self.df_segments.sort_values(
            ["slug", "segment_id", "timestamp"]
        ).copy()

        sensor_violation = df_sorted["depth_proc_mm"] > self.config.sensor_limit_mm
        segments_with_sensor_violation = df_sorted.loc[
            sensor_violation, ["slug", "segment_id"]
        ].drop_duplicates()
        n_segments_sensor_violation = len(segments_with_sensor_violation)
        print(
            f"      Sensor limit filter (>{self.config.sensor_limit_mm}mm): {n_segments_sensor_violation:,} segments removed"
        )

        df_sorted["depth_change"] = df_sorted.groupby(["slug", "segment_id"])[
            "depth_proc_mm"
        ].diff()
        change_violation = df_sorted["depth_change"].abs() > self.change_threshold
        segments_with_change_violation = df_sorted.loc[
            change_violation, ["slug", "segment_id"]
        ].drop_duplicates()
        n_segments_change_violation = len(segments_with_change_violation)
        print(
            f"      Change filter (|change|>{self.change_threshold:.2f}mm): {n_segments_change_violation:,} segments removed"
        )

        all_violating_segments = pd.concat(
            [segments_with_sensor_violation, segments_with_change_violation]
        ).drop_duplicates()
        n_total_violating = len(all_violating_segments)

        df_sorted["_segment_key"] = (
            df_sorted["slug"].astype(str) + "_" + df_sorted["segment_id"].astype(str)
        )
        all_violating_segments["_segment_key"] = (
            all_violating_segments["slug"].astype(str)
            + "_"
            + all_violating_segments["segment_id"].astype(str)
        )
        violating_keys = set(all_violating_segments["_segment_key"])

        mask_valid = ~df_sorted["_segment_key"].isin(violating_keys)
        df_filtered = (
            df_sorted[mask_valid].drop(columns=["depth_change", "_segment_key"]).copy()
        )

        # Renumber segments (vectorized)
        df_filtered = self._renumber_segments(df_filtered)

        n_remaining_records = len(df_filtered)
        n_remaining_segments = df_filtered.groupby(["slug", "segment_id"]).ngroups
        pct_records_remaining = (
            100 * n_remaining_records / n_original_records
            if n_original_records > 0
            else 0
        )
        pct_segments_remaining = (
            100 * n_remaining_segments / n_original_segments
            if n_original_segments > 0
            else 0
        )

        print(
            f"      Segments: {n_remaining_segments:,} / {n_original_segments:,} ({pct_segments_remaining:.1f}%) remaining"
        )
        print(
            f"      Records: {n_remaining_records:,} / {n_original_records:,} ({pct_records_remaining:.1f}%) remaining"
        )

        self.df_filtered = df_filtered
        self.stats["original_segments"] = int(n_original_segments)
        self.stats["segments_sensor_violation"] = int(n_segments_sensor_violation)
        self.stats["segments_change_violation"] = int(n_segments_change_violation)
        self.stats["total_segments_removed"] = int(n_total_violating)
        self.stats["filtered_segments"] = int(n_remaining_segments)
        self.stats["filtered_records"] = int(n_remaining_records)

        return self

    def _renumber_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renumber segments to be consecutive after filtering."""

        original_segments = df.groupby("slug")["segment_id"].nunique().sum()

        def renumber_group(group):
            group = group.copy()
            new_ids, _ = pd.factorize(group["segment_id"], sort=True)
            group["segment_id"] = new_ids + 1
            return group

        df_renumbered = df.groupby("slug", group_keys=False).apply(
            renumber_group, include_groups=False
        )
        # Restore slug column after apply with include_groups=False
        df_renumbered["slug"] = df["slug"].values

        new_segments = df_renumbered.groupby("slug")["segment_id"].nunique().sum()

        if original_segments != new_segments:
            print(
                f"      Segments renumbered: {original_segments:,} -> {new_segments:,}"
            )

        return df_renumbered

    def detect_peaks(self) -> "FloodSegmentProcessor":
        """Detect peaks for each segment."""
        print("[3/4] Detecting peaks...")

        if self.df_filtered is None:
            raise ValueError("No filtered data. Call apply_filters() first.")

        df = self.df_filtered

        idx_max = df.groupby(["slug", "segment_id"])["depth_proc_mm"].idxmax()

        peaks = df.loc[
            idx_max, ["slug", "segment_id", "timestamp", "depth_proc_mm"]
        ].copy()
        peaks.columns = ["slug", "segment_id", "peak_time", "peak_value"]

        time_bounds = df.groupby(["slug", "segment_id"])["timestamp"].agg(
            ["min", "max"]
        )
        time_bounds.columns = ["start_time", "end_time"]
        time_bounds = time_bounds.reset_index()

        peaks = peaks.merge(time_bounds, on=["slug", "segment_id"])

        peaks["duration_minutes"] = (
            peaks["end_time"] - peaks["start_time"]
        ).dt.total_seconds() / 60

        self.df_peaks = peaks.reset_index(drop=True)

        print(f"      Detected {len(self.df_peaks):,} peaks")
        self.stats["total_peaks"] = len(self.df_peaks)

        return self

    def validate_events(
        self, n_workers: Optional[int] = None
    ) -> "FloodSegmentProcessor":
        """Validate flood events by checking for rainfall before peaks."""
        print("[4/4] Validating flood events...")

        if self.df_peaks is None:
            raise ValueError("No peaks detected. Call detect_peaks() first.")

        if n_workers is None:
            n_workers = max(1, cpu_count() // 2)

        self._load_precipitation()

        precip_sorted = self.df_precip.sort_values(["slug", "virtual_timestamp"])
        peaks_sorted = self.df_peaks.sort_values(["slug", "peak_time"])

        lookback = pd.Timedelta(hours=self.config.rainfall_lookback_hours)

        unique_slugs = peaks_sorted["slug"].unique()

        tasks = []
        for slug in unique_slugs:
            slug_peaks = peaks_sorted[peaks_sorted["slug"] == slug]

            peak_records = slug_peaks.to_dict("records")
            tasks.append((slug, peak_records))

        print(
            f"      Processing {len(unique_slugs):,} sensors with {n_workers} workers..."
        )

        with Pool(
            processes=n_workers,
            initializer=_init_validate_worker,
            initargs=(precip_sorted, lookback),
        ) as pool:
            all_results = list(
                tqdm(
                    pool.imap(_validate_sensor_peaks, tasks),
                    total=len(tasks),
                    desc="      Validating",
                )
            )

        results = []
        for sensor_results in all_results:
            results.extend(sensor_results)

        self.df_peaks = pd.DataFrame(results)

        n_valid = self.df_peaks["is_valid_flood_event"].sum()
        n_total = len(self.df_peaks)

        print(
            f"      Valid flood events: {n_valid:,} / {n_total:,} ({100 * n_valid / n_total:.1f}%)"
        )

        self.stats["valid_flood_events"] = int(n_valid)
        self.stats["invalid_flood_events"] = int(n_total - n_valid)

        return self

    def process(self) -> "FloodSegmentProcessor":
        """Run the complete processing pipeline."""
        self.calculate_change_threshold()
        self.apply_filters()
        self.detect_peaks()
        self.validate_events()
        return self

    def save_outputs(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save processed outputs to files."""
        if output_dir is None:
            output_dir = self.config.output_dir

        os.makedirs(output_dir, exist_ok=True)

        output_paths = {}

        if self.df_filtered is not None:
            segments_path = os.path.join(
                output_dir, self.config.filtered_segments_filename
            )
            self.df_filtered.to_csv(segments_path, index=False)
            output_paths["filtered_segments"] = segments_path

        if self.df_peaks is not None:
            peaks_path = os.path.join(output_dir, self.config.peak_info_filename)
            self.df_peaks.to_csv(peaks_path, index=False)
            output_paths["peak_info"] = peaks_path

        if self.df_filtered is not None and self.df_peaks is not None:
            valid_segments_path = os.path.join(output_dir, "flood_segments_valid.csv")
            df_valid_segments = self.get_valid_flood_segments()
            df_valid_segments.to_csv(valid_segments_path, index=False)
            output_paths["valid_segments"] = valid_segments_path

        stats_path = os.path.join(output_dir, self.config.stats_filename)
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        output_paths["stats"] = stats_path

        print()
        print("=" * 50)
        print("Processing complete:")
        print(
            f"  - Input: {self.stats.get('original_records', 0):,} records, {self.stats.get('original_segments', 0):,} segments"
        )
        print(
            f"  - Filtered out: {self.stats.get('total_segments_removed', 0):,} segments (sensor: {self.stats.get('segments_sensor_violation', 0):,}, change: {self.stats.get('segments_change_violation', 0):,})"
        )
        print(
            f"  - Output: {self.stats.get('filtered_records', 0):,} records, {self.stats.get('filtered_segments', 0):,} segments"
        )
        print(f"  - Total peaks: {self.stats.get('total_peaks', 0):,}")
        print(f"  - Valid flood events: {self.stats.get('valid_flood_events', 0):,}")
        print(f"  - Output files:")
        for name, path in output_paths.items():
            print(f"    - {path}")
        print("=" * 50)

        return output_paths

    def get_valid_flood_peaks(self) -> pd.DataFrame:
        """Get only the validated flood events."""
        if self.df_peaks is None:
            raise ValueError(
                "No peaks detected. Call detect_peaks() or process() first."
            )

        return self.df_peaks[self.df_peaks["is_valid_flood_event"]].copy()

    def get_valid_flood_segments(self) -> pd.DataFrame:
        """Get segment data for valid flood events only."""
        if self.df_filtered is None or self.df_peaks is None:
            raise ValueError("No processed data. Call process() first.")

        valid_peaks = self.df_peaks[self.df_peaks["is_valid_flood_event"]]
        valid_keys = valid_peaks[["slug", "segment_id"]].drop_duplicates()

        df_valid = self.df_filtered.merge(
            valid_keys, on=["slug", "segment_id"], how="inner"
        )

        return df_valid

    def get_filtered_segments_for_peak(
        self, slug: str, segment_id: int
    ) -> pd.DataFrame:
        """Get filtered segment data for a specific peak."""
        if self.df_filtered is None:
            raise ValueError(
                "No filtered data. Call apply_filters() or process() first."
            )

        mask = (self.df_filtered["slug"] == slug) & (
            self.df_filtered["segment_id"] == segment_id
        )
        return self.df_filtered[mask].copy()
