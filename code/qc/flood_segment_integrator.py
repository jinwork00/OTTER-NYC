"""Join temporally continuous flood segments and interpolate internal zero values."""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class IntegratorConfig:
    """Configuration for FloodSegmentIntegrator."""

    time_threshold_minutes: float = 2.0
    expected_interval_minutes: float = 1.0

    interpolation_method: str = "time"
    preserve_endpoints: bool = True

    include_all_segments: bool = True

    output_dir: str = "data/segments"
    interpolated_filename: str = "flood_segments_interpolated.csv"
    stats_filename: str = "integration_stats.json"


class FloodSegmentIntegrator:
    """Integrates flood segments based on temporal continuity and interpolates zero values."""

    def __init__(self, config: Optional[IntegratorConfig] = None):

        self.config = config or IntegratorConfig()

        self.df_segments: Optional[pd.DataFrame] = None
        self.df_continuous: Optional[pd.DataFrame] = None
        self.df_interpolated: Optional[pd.DataFrame] = None

        self._continuous_pairs: Optional[pd.DataFrame] = None
        self._zero_mask: Optional[pd.Series] = None

        self.stats: Dict = {}

    def load_segments(self, path: str) -> "FloodSegmentIntegrator":
        """Load segment data from CSV file."""
        print("[1/5] Loading segments...")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Segments file not found: {path}")

        self.df_segments = pd.read_csv(path, parse_dates=["timestamp"])

        required_cols = {"segment_id", "timestamp", "depth_proc_mm", "slug"}
        missing_cols = required_cols - set(self.df_segments.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.df_segments = self.df_segments.sort_values(
            ["slug", "segment_id", "timestamp"]
        ).reset_index(drop=True)

        self.stats["original_records"] = len(self.df_segments)
        self.stats["original_segments"] = self.df_segments["segment_id"].nunique()
        self.stats["sensors"] = self.df_segments["slug"].nunique()

        print(
            f"      Loaded {len(self.df_segments):,} records, "
            f"{self.stats['original_segments']:,} segments, "
            f"{self.stats['sensors']} sensors"
        )

        return self

    def detect_continuous_pairs(self) -> "FloodSegmentIntegrator":
        """Detect temporally continuous segment pairs."""
        print("[2/5] Detecting continuous segment pairs...")

        if self.df_segments is None:
            raise ValueError("No segments loaded. Call load_segments() first.")

        segment_times = (
            self.df_segments.groupby(["slug", "segment_id"])
            .agg(start_time=("timestamp", "min"), end_time=("timestamp", "max"))
            .reset_index()
        )

        segment_times = segment_times.sort_values(["slug", "start_time"]).reset_index(
            drop=True
        )

        segment_times["prev_end_time"] = segment_times.groupby("slug")[
            "end_time"
        ].shift(1)
        segment_times["is_continuous"] = (
            segment_times["start_time"] == segment_times["prev_end_time"]
        )

        self._continuous_pairs = segment_times[segment_times["is_continuous"]].copy()
        self.stats["continuous_pairs"] = len(self._continuous_pairs)

        print(
            f"      Found {self.stats['continuous_pairs']:,} continuous segment pairs"
        )

        return self

    def filter_continuous_segments(self) -> "FloodSegmentIntegrator":
        """Filter data to include continuous segments or all segments."""
        print("[3/5] Filtering continuous segments...")

        if self._continuous_pairs is None:
            raise ValueError(
                "No continuous pairs detected. Call detect_continuous_pairs() first."
            )

        if self.config.include_all_segments:
            self.df_continuous = self.df_segments.copy()
            self.stats["continuous_records"] = len(self.df_continuous)
            self.stats["continuous_segment_ids"] = self.df_segments[
                "segment_id"
            ].nunique()
            self.stats["filtered_mode"] = "all_segments"
            print(
                f"      Including ALL {self.stats['continuous_records']:,} records, "
                f"{self.stats['continuous_segment_ids']:,} segment IDs "
                f"(will re-segment by time breaks)"
            )
        else:
            continuous_ids = self._continuous_pairs[["slug", "segment_id"]].copy()
            continuous_ids["prev_segment_id"] = continuous_ids["segment_id"] - 1

            filter_df = pd.concat(
                [
                    continuous_ids[["slug", "prev_segment_id"]].rename(
                        columns={"prev_segment_id": "segment_id"}
                    ),
                    continuous_ids[["slug", "segment_id"]],
                ]
            ).drop_duplicates()

            self.df_continuous = (
                self.df_segments.merge(filter_df, on=["slug", "segment_id"])
                .sort_values(["slug", "segment_id", "timestamp"])
                .reset_index(drop=True)
            )

            self.stats["continuous_records"] = len(self.df_continuous)
            self.stats["continuous_segment_ids"] = len(filter_df)
            self.stats["filtered_mode"] = "continuous_only"
            print(
                f"      Filtered to {self.stats['continuous_records']:,} records, "
                f"{self.stats['continuous_segment_ids']:,} segment IDs (continuous only)"
            )

        return self

    def resegment_by_time_breaks(self) -> "FloodSegmentIntegrator":
        """Re-segment data based on time breaks."""
        print("[4/5] Re-segmenting by time breaks...")

        if self.df_continuous is None:
            raise ValueError(
                "No continuous segments. Call filter_continuous_segments() first."
            )

        threshold = self.config.time_threshold_minutes

        self.df_continuous["time_diff"] = (
            self.df_continuous.groupby("slug")["timestamp"].diff().dt.total_seconds()
            / 60
        )

        is_break = (self.df_continuous["time_diff"] >= threshold) | (
            self.df_continuous["time_diff"].isna()
        )

        self.df_continuous["new_segment_id"] = is_break.groupby(
            self.df_continuous["slug"]
        ).cumsum()

        self.stats["time_breaks"] = int(is_break.sum())
        self.stats["new_segments"] = self.df_continuous.groupby(
            ["slug", "new_segment_id"]
        ).ngroups

        print(
            f"      Created {self.stats['new_segments']:,} new segments "
            f"from {self.stats['time_breaks']:,} time breaks"
        )

        return self

    def apply_interpolation(self) -> "FloodSegmentIntegrator":
        """Apply time-weighted interpolation to fill zero values."""
        print("[5/5] Applying time-weighted interpolation...")

        if (
            self.df_continuous is None
            or "new_segment_id" not in self.df_continuous.columns
        ):
            raise ValueError(
                "Segments not re-segmented. Call resegment_by_time_breaks() first."
            )

        # Store zero mask before interpolation
        self._zero_mask = self.df_continuous["depth_proc_mm"] == 0
        self.stats["zeros_before"] = int(self._zero_mask.sum())

        self.df_interpolated = self.df_continuous.copy()

        method = self.config.interpolation_method
        preserve_endpoints = self.config.preserve_endpoints

        group_cols = ["slug", "new_segment_id"]

        def interpolate_group(group):
            """Interpolate a single segment group."""
            group = group.copy()
            group = group.set_index("timestamp")

            first_val = group["depth_proc_mm"].iloc[0]
            last_val = group["depth_proc_mm"].iloc[-1]

            group["depth_proc_mm"] = group["depth_proc_mm"].replace(0, np.nan)

            group["depth_proc_mm"] = group["depth_proc_mm"].interpolate(method=method)

            if preserve_endpoints:
                group.iloc[0, group.columns.get_loc("depth_proc_mm")] = first_val
                group.iloc[-1, group.columns.get_loc("depth_proc_mm")] = last_val

            group["depth_proc_mm"] = group["depth_proc_mm"].fillna(0)

            return group.reset_index()

        # Apply interpolation to each segment (include_groups=False to avoid FutureWarning)

        result_list = []
        for (slug, new_seg_id), group in self.df_interpolated.groupby(group_cols):
            interpolated = interpolate_group(group)
            interpolated["slug"] = slug
            interpolated["new_segment_id"] = new_seg_id
            result_list.append(interpolated)

        self.df_interpolated = pd.concat(result_list, ignore_index=True)

        self.stats["zeros_after"] = int(
            (self.df_interpolated["depth_proc_mm"] == 0).sum()
        )
        self.stats["interpolated_count"] = (
            self.stats["zeros_before"] - self.stats["zeros_after"]
        )

        if self.stats["zeros_before"] > 0:
            interp_rate = (
                self.stats["interpolated_count"] / self.stats["zeros_before"] * 100
            )
        else:
            interp_rate = 0.0

        print(
            f"      Interpolated {self.stats['interpolated_count']:,} values "
            f"({interp_rate:.1f}% of zeros)"
        )

        return self

    def process(self) -> "FloodSegmentIntegrator":
        """Execute the full processing pipeline."""
        if self.df_segments is None:
            raise ValueError("No segments loaded. Call load_segments() first.")

        return (
            self.detect_continuous_pairs()
            .filter_continuous_segments()
            .resegment_by_time_breaks()
            .apply_interpolation()
        )

    def save_outputs(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save processed outputs to files."""
        if self.df_interpolated is None:
            raise ValueError("No interpolated data. Call process() first.")

        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_paths = {}

        # Save interpolated segments
        interp_path = os.path.join(output_dir, self.config.interpolated_filename)
        self.df_interpolated.to_csv(interp_path, index=False)
        output_paths["interpolated_segments"] = interp_path

        stats_path = os.path.join(output_dir, self.config.stats_filename)
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        output_paths["stats"] = stats_path

        print("\n" + "=" * 60)
        print("Output Files Saved")
        print("=" * 60)
        for output_type, path in output_paths.items():
            print(f"  {output_type}: {path}")

        return output_paths

    def get_interpolated_segments(self) -> pd.DataFrame:
        """Get the interpolated segment data."""
        if self.df_interpolated is None:
            raise ValueError("No interpolated data. Call process() first.")
        return self.df_interpolated.copy()

    def get_continuous_segments(self) -> pd.DataFrame:
        """Get the continuous segment data before interpolation."""
        if self.df_continuous is None:
            raise ValueError(
                "No continuous segments. Call filter_continuous_segments() first."
            )
        return self.df_continuous.copy()

    def get_validation_report(self) -> Dict:
        """Generate a validation report comparing before and after interpolation."""
        if self.df_interpolated is None or self.df_continuous is None:
            raise ValueError("No data to validate. Call process() first.")

        before = self.df_continuous["depth_proc_mm"]
        after = self.df_interpolated["depth_proc_mm"]

        report = {
            "total_records": len(self.df_continuous),
            "zeros_before": self.stats["zeros_before"],
            "zeros_after": self.stats["zeros_after"],
            "zero_ratio_before": self.stats["zeros_before"] / len(before) * 100,
            "zero_ratio_after": self.stats["zeros_after"] / len(after) * 100,
            "interpolated_count": self.stats["interpolated_count"],
            "interpolation_rate": (
                self.stats["interpolated_count"] / self.stats["zeros_before"] * 100
                if self.stats["zeros_before"] > 0
                else 0.0
            ),
            "mean_before": float(before.mean()),
            "mean_after": float(after.mean()),
            "mean_change_pct": (after.mean() / before.mean() - 1) * 100
            if before.mean() > 0
            else 0.0,
            "std_before": float(before.std()),
            "std_after": float(after.std()),
            "new_segments": self.stats.get("new_segments", 0),
        }

        return report

    def print_summary(self) -> None:
        """Print a formatted summary of the processing results."""
        if self.df_interpolated is None:
            print("No data processed yet.")
            return

        report = self.get_validation_report()

        print("\n" + "=" * 60)
        print("FloodSegmentIntegrator Processing Summary")
        print("=" * 60)

        print("\n[Data Size]")
        print(f"  Total records: {report['total_records']:,}")
        print(f"  New segments:  {report['new_segments']:,}")

        print("\n[Zero Value Processing]")
        print(
            f"  Before:        {report['zeros_before']:,} ({report['zero_ratio_before']:.2f}%)"
        )
        print(
            f"  After:         {report['zeros_after']:,} ({report['zero_ratio_after']:.2f}%)"
        )
        print(f"  Interpolated:  {report['interpolated_count']:,}")
        print(f"  Interp. Rate:  {report['interpolation_rate']:.1f}%")

        print("\n[Depth Statistics]")
        print(f"  Mean Before:   {report['mean_before']:.2f} mm")
        print(f"  Mean After:    {report['mean_after']:.2f} mm")
        print(f"  Change:        {report['mean_change_pct']:+.1f}%")

        print("\n" + "=" * 60)
