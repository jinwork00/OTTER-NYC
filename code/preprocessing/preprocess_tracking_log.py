"""Build aligned FloodNet, ASOS precipitation, and terrain products with audit logging."""

import argparse
import glob
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional, TextIO, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from tqdm import tqdm

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import contextily as ctx
except Exception:
    ctx = None


CRS = "EPSG:2263"
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = str(REPO_ROOT / "data" / "01_raw")
FLOODNET_DIR = os.path.join(ROOT_DIR, "floodnet")
PRECIP_DIR = os.path.join(ROOT_DIR, "asos")
TERRAIN_DIR = os.path.join(ROOT_DIR, "dem")
RASTER_FILES = {
    "dem": os.path.join(TERRAIN_DIR, "elevation.tif"),
    "slope": os.path.join(TERRAIN_DIR, "slope.tif"),
    "aspect": os.path.join(TERRAIN_DIR, "aspect.tif"),
}
MAP_DIR = os.path.join("data", "map")
DEFAULT_BOROUGH_SHP_GLOB = os.path.join(MAP_DIR, "*.shp")
PRIORITY_EVENT_DATE = "2023-09-29"


def read_table(path: str, **kwargs) -> pd.DataFrame:

    if path.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def existing_table_path(*candidates: str) -> str:

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No candidate file exists: " + ", ".join(candidates))


class TeeStream:
    def __init__(self, *streams: TextIO):

        self.streams = streams

    def write(self, data: str) -> int:

        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:

        for stream in self.streams:
            stream.flush()


def setup_dual_logging(log_file_path: str) -> TextIO:

    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_stream = open(log_file_path, "a", encoding="utf-8")

    sys.stdout = TeeStream(sys.stdout, file_stream)
    sys.stderr = TeeStream(sys.stderr, file_stream)
    return file_stream


def log(msg: str) -> None:

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def section(title: str) -> None:

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_flood(df: pd.DataFrame, label: str) -> None:

    if df.empty:
        log(f"{label}: empty DataFrame")
        return

    total_rows = len(df)
    sensors = df["slug"].nunique() if "slug" in df.columns else 0
    depth_col = "depth_proc_mm" if "depth_proc_mm" in df.columns else None
    time_col = (
        "time"
        if "time" in df.columns
        else "timestamp"
        if "timestamp" in df.columns
        else None
    )

    log(f"{label} | rows={total_rows:,}, sensors={sensors:,}")
    if depth_col:
        nan_count = df[depth_col].isna().sum()
        zero_count = (df[depth_col] == 0).sum()
        positive_count = (df[depth_col] > 0).sum()
        log(
            f"{label} | depth NaN={nan_count:,} ({nan_count / total_rows:.2%}), "
            f"zero={zero_count:,} ({zero_count / total_rows:.2%}), "
            f"positive={positive_count:,} ({positive_count / total_rows:.2%})"
        )

    if time_col:
        tmin = pd.to_datetime(df[time_col], errors="coerce").min()
        tmax = pd.to_datetime(df[time_col], errors="coerce").max()
        log(f"{label} | time range: {tmin} -> {tmax}")


def summarize_integrated(df: pd.DataFrame, label: str) -> None:

    if df.empty:
        log(f"{label}: empty DataFrame")
        return

    total_rows = len(df)
    sensors = df["slug"].nunique() if "slug" in df.columns else 0
    timestamps = df["timestamp"].nunique() if "timestamp" in df.columns else 0

    log(
        f"{label} | rows={total_rows:,}, sensors={sensors:,}, timestamps={timestamps:,}"
    )
    if "depth_proc_mm" in df.columns:
        depth_nan = df["depth_proc_mm"].isna().sum()
        log(f"{label} | depth NaN={depth_nan:,} ({depth_nan / total_rows:.2%})")
    if "precip" in df.columns:
        precip_nan = df["precip"].isna().sum()
        precip_zero = (df["precip"] == 0).sum()
        log(
            f"{label} | precip NaN={precip_nan:,} ({precip_nan / total_rows:.2%}), "
            f"precip zero={precip_zero:,} ({precip_zero / total_rows:.2%})"
        )


def summarize_terrain(df: pd.DataFrame, id_col: str = "slug") -> None:

    if df.empty:
        log("Terrain summary: empty DataFrame")
        return

    feature_cols = [c for c in df.columns if c != id_col]
    log(f"Terrain summary | rows={len(df):,}, features={len(feature_cols):,}")
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        total = len(series)
        nan_count = int(series.isna().sum())
        valid = series.dropna()
        if valid.empty:
            log_reduction_reason(
                "Terrain feature",
                f"{col}: dtype={df[col].dtype}, valid=0/{total:,}, NaN={nan_count:,} (all missing)",
            )
            continue

        sample = (
            valid if len(valid) <= 100000 else valid.sample(100000, random_state=42)
        )
        approx_unique = int(sample.round(4).nunique())
        log_reduction_reason(
            "Terrain feature",
            (
                f"{col}: dtype={df[col].dtype}, valid={len(valid):,}/{total:,}, NaN={nan_count:,}, "
                f"min={valid.min():.6f}, max={valid.max():.6f}, mean={valid.mean():.6f}, "
                f"approx_unique(rounded4)={approx_unique:,}"
            ),
        )


def summarize_transition(before: pd.DataFrame, after: pd.DataFrame, label: str) -> None:

    before_rows, after_rows = len(before), len(after)
    removed = before_rows - after_rows

    ratio = 0.0 if before_rows == 0 else removed / before_rows
    log(
        f"{label} | rows {before_rows:,} -> {after_rows:,} (removed {removed:,}, {ratio:.2%})"
    )


def log_reduction_reason(label: str, reason: str) -> None:

    log(f"{label} | reason: {reason}")


def load_valid_sensor_df() -> pd.DataFrame:

    floodnet_files = glob.glob(f"{FLOODNET_DIR}/*/*.parquet") + glob.glob(
        f"{FLOODNET_DIR}/*/*.csv"
    )
    if len(floodnet_files) == 0:
        raise ValueError("No FloodNet sensor files found in FloodNet directory.")

    raw_data_slugs = {os.path.basename(os.path.dirname(f)) for f in floodnet_files}
    deployment_path = existing_table_path(
        os.path.join(FLOODNET_DIR, "deployment-info.parquet"),
        os.path.join(FLOODNET_DIR, "deployment-info.csv"),
    )
    deployment_info_df = read_table(deployment_path)
    deployment_info_df = deployment_info_df.dropna(subset=["latitude", "longitude"])
    valid_slugs = set(deployment_info_df["slug"]).intersection(raw_data_slugs)
    return deployment_info_df[deployment_info_df["slug"].isin(valid_slugs)]


def create_location_df(
    df: pd.DataFrame,
    crs_to: str,
    id_col: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:

    out = df.copy()

    transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

    out["x"], out["y"] = transformer.transform(
        out[lon_col].to_numpy(), out[lat_col].to_numpy()
    )
    return out[[id_col, lat_col, lon_col, "x", "y"]]


def create_terrain_df(
    location_df: pd.DataFrame,
    raster_file_path_dict: dict,
    id_col: str,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:

    section("Stage C-1: Terrain sampling")

    coords = location_df[[x_col, y_col]].values
    ids = location_df[id_col].values
    terrain_data = {id_col: ids}

    for raster_name, raster_path in raster_file_path_dict.items():
        if not os.path.exists(raster_path):
            log_reduction_reason(
                "Terrain sampling", f"{raster_name}: file missing -> {raster_path}"
            )
            terrain_data[raster_name] = [np.nan] * len(ids)
            continue

        with rasterio.open(raster_path) as src:
            sampled_values = list(src.sample(coords))
            values = []
            out_of_bounds = 0
            nodata_count = 0
            nan_count = 0
            empty_sample = 0

            for i, sample_result in enumerate(sampled_values):
                x, y = coords[i]
                in_bounds = (
                    src.bounds.left <= x <= src.bounds.right
                    and src.bounds.bottom <= y <= src.bounds.top
                )
                if not in_bounds:
                    out_of_bounds += 1
                    values.append(np.nan)
                    continue

                if len(sample_result) == 0:
                    empty_sample += 1
                    values.append(np.nan)
                    continue

                val = sample_result[0]
                if src.nodata is not None and val == src.nodata:
                    nodata_count += 1
                    values.append(np.nan)
                    continue
                if np.isnan(val):
                    nan_count += 1
                    values.append(np.nan)
                    continue

                values.append(float(val))

            terrain_data[raster_name] = values
            valid_count = len(values) - int(np.isnan(values).sum())
            log_reduction_reason(
                "Terrain sampling",
                (
                    f"{raster_name}: total_points={len(values):,}, valid={valid_count:,}, "
                    f"out_of_bounds={out_of_bounds:,}, nodata={nodata_count:,}, "
                    f"nan={nan_count:,}, empty_sample={empty_sample:,}"
                ),
            )

    terrain_df = pd.DataFrame(terrain_data)
    summarize_terrain(terrain_df, id_col=id_col)
    return terrain_df


def load_floodnet_df_by_sensor(filepath: str, sensor_id: str) -> pd.DataFrame:

    df = read_table(filepath)
    raw_rows = len(df)
    if "time" not in df.columns or "depth_proc_mm" not in df.columns:
        out = pd.DataFrame(columns=["slug", "time", "depth_proc_mm"])
        out.attrs["raw_rows"] = raw_rows
        out.attrs["kept_rows"] = 0
        out.attrs["dropped_rows"] = raw_rows
        out.attrs["missing_time_rows"] = raw_rows if "time" not in df.columns else 0
        out.attrs["missing_depth_rows"] = (
            raw_rows if "depth_proc_mm" not in df.columns else 0
        )
        out.attrs["missing_both_rows"] = raw_rows
        out.attrs["missing_required_columns"] = 1
        return out

    missing_time_mask = df["time"].isna()
    missing_depth_mask = df["depth_proc_mm"].isna()
    missing_either_mask = missing_time_mask | missing_depth_mask

    df = df.dropna(subset=["time", "depth_proc_mm"])
    df["slug"] = sensor_id
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    out = df[["slug", "time", "depth_proc_mm"]]
    out.attrs["raw_rows"] = raw_rows
    out.attrs["kept_rows"] = len(out)
    out.attrs["dropped_rows"] = int(missing_either_mask.sum())
    out.attrs["missing_time_rows"] = int(missing_time_mask.sum())
    out.attrs["missing_depth_rows"] = int(missing_depth_mask.sum())
    out.attrs["missing_both_rows"] = int((missing_time_mask & missing_depth_mask).sum())
    out.attrs["missing_required_columns"] = 0
    return out


def filter_sensors_by_data_quality(
    floodnet_df: pd.DataFrame,
    min_positive_ratio: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    sensor_stats = floodnet_df.groupby("slug")["depth_proc_mm"].agg(
        total_count="count",
        positive_count=lambda x: (x > 0).sum(),
        total_sum="sum",
    )

    positive_ratio = sensor_stats["positive_count"] / sensor_stats["total_count"].clip(
        lower=1
    )
    valid_mask = sensor_stats["total_sum"] > 0
    if min_positive_ratio > 0:
        valid_mask = valid_mask & (positive_ratio >= min_positive_ratio)

    valid_sensors = sensor_stats[valid_mask].index.tolist()
    log(
        f"Sensor filtering | kept {len(valid_sensors):,} / {floodnet_df['slug'].nunique():,} sensors"
    )

    failed_total_sum = sensor_stats["total_sum"] <= 0
    failed_pos_ratio = (
        positive_ratio < min_positive_ratio
        if min_positive_ratio > 0
        else pd.Series(False, index=sensor_stats.index)
    )
    removed_mask = ~valid_mask

    reason_total_sum_only = (removed_mask & failed_total_sum & ~failed_pos_ratio).sum()
    reason_pos_ratio_only = (removed_mask & ~failed_total_sum & failed_pos_ratio).sum()
    reason_both = (removed_mask & failed_total_sum & failed_pos_ratio).sum()

    log_reduction_reason(
        "Sensor filtering",
        (
            f"removed sensors={removed_mask.sum():,}; "
            f"total_sum<=0 only={reason_total_sum_only:,}, "
            f"positive_ratio<{min_positive_ratio:.2f} only={reason_pos_ratio_only:,}, "
            f"both={reason_both:,}"
        ),
    )

    if removed_mask.any():
        removed_ids = sensor_stats.index[removed_mask].tolist()
        preview = ", ".join(map(str, removed_ids[:10]))
        suffix = " ..." if len(removed_ids) > 10 else ""
        log_reduction_reason(
            "Sensor filtering", f"removed sensor IDs sample: {preview}{suffix}"
        )

    return floodnet_df[floodnet_df["slug"].isin(valid_sensors)], sensor_stats


def resample_to_1minute(floodnet_df: pd.DataFrame) -> pd.DataFrame:

    floodnet_df = floodnet_df.set_index("time").sort_index()
    resampled = (
        floodnet_df.groupby("slug")["depth_proc_mm"]
        .resample("1min")
        .mean()
        .reset_index()
    )
    return resampled[["slug", "time", "depth_proc_mm"]]


def load_floodnet_df(floodnet_dir: str, max_processes: int = 12) -> pd.DataFrame:

    sensor_df = load_valid_sensor_df()
    sensor_ids = sensor_df["slug"].unique().tolist()

    file_sensor_pairs = []
    sensor_file_counts = {}
    for sensor_id in sensor_ids:
        raw_files = sorted(
            glob.glob(os.path.join(floodnet_dir, sensor_id, "*.parquet"))
            + glob.glob(os.path.join(floodnet_dir, sensor_id, "*.csv"))
        )
        sensor_file_counts[sensor_id] = len(raw_files)
        file_sensor_pairs.extend((raw_file, sensor_id) for raw_file in raw_files)

    section("Stage A: Load raw FloodNet")
    log(f"Loading {len(file_sensor_pairs):,} FloodNet files")
    if sensor_file_counts:
        non_zero_counts = [c for c in sensor_file_counts.values() if c > 0]
        if non_zero_counts:
            log_reduction_reason(
                "Flood raw file discovery",
                (
                    f"sensors={len(sensor_file_counts):,}, sensors_with_files={len(non_zero_counts):,}, "
                    f"min_files_per_sensor={min(non_zero_counts):,}, "
                    f"max_files_per_sensor={max(non_zero_counts):,}, "
                    f"avg_files_per_sensor={np.mean(non_zero_counts):.2f}"
                ),
            )
            preview_items = list(sensor_file_counts.items())[:10]
            preview_text = ", ".join([f"{sid}:{cnt}" for sid, cnt in preview_items])
            log_reduction_reason(
                "Flood raw file discovery",
                f"sensor:file_count sample -> {preview_text}",
            )

    n_processes = min(mp.cpu_count(), len(file_sensor_pairs), max_processes)
    if n_processes == 0:
        raise ValueError("No FloodNet sensor files found to load.")
    log(f"FloodNet loading workers={n_processes}")

    with mp.Pool(processes=n_processes) as pool:
        dataframes = list(
            tqdm(
                pool.starmap(load_floodnet_df_by_sensor, file_sensor_pairs),
                total=len(file_sensor_pairs),
                desc="Loading FloodNet files",
            )
        )

    total_raw_rows = sum(int(df.attrs.get("raw_rows", len(df))) for df in dataframes)
    total_kept_rows = sum(int(df.attrs.get("kept_rows", len(df))) for df in dataframes)
    total_dropped_rows = sum(int(df.attrs.get("dropped_rows", 0)) for df in dataframes)
    total_missing_time_rows = sum(
        int(df.attrs.get("missing_time_rows", 0)) for df in dataframes
    )
    total_missing_depth_rows = sum(
        int(df.attrs.get("missing_depth_rows", 0)) for df in dataframes
    )
    total_missing_both_rows = sum(
        int(df.attrs.get("missing_both_rows", 0)) for df in dataframes
    )
    total_missing_required_columns_files = sum(
        int(df.attrs.get("missing_required_columns", 0)) for df in dataframes
    )

    log_reduction_reason(
        "Flood raw loader",
        (
            f"rows dropped before merge due to missing time/depth: raw={total_raw_rows:,}, "
            f"kept={total_kept_rows:,}, dropped={total_dropped_rows:,}; "
            f"missing_time_rows={total_missing_time_rows:,}, "
            f"missing_depth_rows={total_missing_depth_rows:,}, both_missing={total_missing_both_rows:,}; "
            f"files_missing_required_columns={total_missing_required_columns_files:,}"
        ),
    )

    non_empty_dfs = [df for df in dataframes if not df.empty]
    raw_df = pd.concat(non_empty_dfs, ignore_index=True)
    summarize_flood(raw_df, "Flood raw")
    return raw_df


def calculate_distance(
    target_coords: np.ndarray, source_coords: np.ndarray
) -> np.ndarray:

    target_rad = np.radians(target_coords)
    source_rad = np.radians(source_coords)

    lat1, lon1 = target_rad[:, 0][:, np.newaxis], target_rad[:, 1][:, np.newaxis]
    lat2, lon2 = source_rad[:, 0][np.newaxis, :], source_rad[:, 1][np.newaxis, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def idw_interpolation(
    target_coords: np.ndarray,
    source_coords: np.ndarray,
    values: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:

    distances = np.maximum(calculate_distance(target_coords, source_coords), 1e-10)

    weights = 1 / (distances**power)

    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


def load_iem_asos_df_by_station(filepath: str) -> pd.DataFrame:

    try:
        if filepath.endswith(".parquet"):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        log(f"Error loading {filepath}: {exc}")
        out = pd.DataFrame()
        out.attrs["file_read_error"] = 1
        out.attrs["missing_required_columns"] = 0
        out.attrs["raw_rows"] = 0
        out.attrs["dropped_timestamp_na_rows"] = 0
        out.attrs["dropped_duplicate_timestamp_rows"] = 0
        out.attrs["kept_rows"] = 0
        return out

    required_cols = ["station", "station_name", "lat", "lon", "valid(UTC)", "precip"]
    if not all(col in df.columns for col in required_cols):
        log(f"Missing required columns in {filepath}")
        out = pd.DataFrame()
        out.attrs["file_read_error"] = 0
        out.attrs["missing_required_columns"] = 1
        out.attrs["raw_rows"] = len(df)
        out.attrs["dropped_timestamp_na_rows"] = 0
        out.attrs["dropped_duplicate_timestamp_rows"] = 0
        out.attrs["kept_rows"] = 0
        return out

    raw_rows = len(df)
    df = df[required_cols].copy()
    df.columns = ["station", "station_name", "lat", "lon", "timestamp", "precip"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    dropped_timestamp_na_rows = int(df["timestamp"].isna().sum())
    df["precip"] = df["precip"].replace("M", np.nan)
    df["precip"] = pd.to_numeric(df["precip"], errors="coerce")
    df["precip"] = df["precip"].fillna(0) * 25.4
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    before_dedup_rows = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    dropped_duplicate_timestamp_rows = before_dedup_rows - len(df)

    df.attrs["file_read_error"] = 0
    df.attrs["missing_required_columns"] = 0
    df.attrs["raw_rows"] = raw_rows
    df.attrs["dropped_timestamp_na_rows"] = dropped_timestamp_na_rows
    df.attrs["dropped_duplicate_timestamp_rows"] = dropped_duplicate_timestamp_rows
    df.attrs["kept_rows"] = len(df)
    return df


def create_iem_asos_df(precip_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    station_names = ["NYC", "JFK", "LGA", "EWR", "HPN"]
    station_files = []
    for name in station_names:
        for candidate in (
            os.path.join(precip_dir, f"{name}.parquet"),
            os.path.join(precip_dir, f"IEM_ASOS_{name}.parquet"),
            os.path.join(precip_dir, f"IEM_ASOS_{name}.csv"),
            os.path.join(precip_dir, f"{name}.csv"),
        ):
            if os.path.exists(candidate):
                station_files.append(candidate)
                break
    if not station_files:
        raise ValueError(f"No IEM ASOS files found in {precip_dir}")

    section("Stage C: Load precipitation station data")
    log(f"Loading {len(station_files):,} station files")

    with mp.Pool(processes=min(len(station_files), 5)) as pool:
        station_dfs = pool.map(load_iem_asos_df_by_station, station_files)

    file_read_error_count = sum(
        int(df.attrs.get("file_read_error", 0)) for df in station_dfs
    )
    missing_required_columns_count = sum(
        int(df.attrs.get("missing_required_columns", 0)) for df in station_dfs
    )
    total_raw_rows = sum(int(df.attrs.get("raw_rows", len(df))) for df in station_dfs)
    total_dropped_timestamp_na_rows = sum(
        int(df.attrs.get("dropped_timestamp_na_rows", 0)) for df in station_dfs
    )
    total_dropped_duplicate_timestamp_rows = sum(
        int(df.attrs.get("dropped_duplicate_timestamp_rows", 0)) for df in station_dfs
    )
    total_kept_rows = sum(int(df.attrs.get("kept_rows", len(df))) for df in station_dfs)

    log_reduction_reason(
        "Precip station loading",
        (
            f"files total={len(station_files):,}, valid={len(station_files) - file_read_error_count - missing_required_columns_count:,}, "
            f"dropped_read_error={file_read_error_count:,}, dropped_missing_columns={missing_required_columns_count:,}; "
            f"rows raw={total_raw_rows:,}, dropped_timestamp_na={total_dropped_timestamp_na_rows:,}, "
            f"dropped_duplicate_timestamp={total_dropped_duplicate_timestamp_rows:,}, kept={total_kept_rows:,}"
        ),
    )

    valid_dfs = [df for df in station_dfs if not df.empty]
    if not valid_dfs:
        raise ValueError("No valid station data found")

    station_info = {}
    all_station_data = []
    for df in valid_dfs:
        first_row = df.iloc[0]
        station = first_row["station"]
        station_name = first_row["station_name"]
        station_rows = len(df)
        unique_ts = df["timestamp"].nunique()
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        dropped_ts_na = int(df.attrs.get("dropped_timestamp_na_rows", 0))
        dropped_dup_ts = int(df.attrs.get("dropped_duplicate_timestamp_rows", 0))

        log_reduction_reason(
            "Precip station merge source",
            (
                f"station={station} ({station_name}), rows={station_rows:,}, unique_timestamps={unique_ts:,}, "
                f"range={ts_min} -> {ts_max}, dropped_timestamp_na={dropped_ts_na:,}, "
                f"dropped_duplicate_timestamp={dropped_dup_ts:,}"
            ),
        )

        station_info[first_row["station"]] = {
            "station_name": first_row["station_name"],
            "lat": first_row["lat"],
            "lon": first_row["lon"],
        }
        all_station_data.append(df[["station", "timestamp", "precip"]])

    static_df = pd.DataFrame.from_dict(station_info, orient="index")
    static_df = static_df.sort_values("station_name").reset_index(names="station")

    combined_df = pd.concat(all_station_data, ignore_index=True)
    dynamic_df = combined_df.pivot_table(
        index="timestamp",
        columns="station",
        values="precip",
        aggfunc="mean",
    ).sort_index()

    log(f"Station static rows={len(static_df):,}, dynamic rows={len(dynamic_df):,}")
    log_reduction_reason(
        "Precip station merge result",
        f"merged stations={len(all_station_data):,}; static metadata rows={len(static_df):,}; dynamic timestamp rows={len(dynamic_df):,}",
    )
    return static_df, dynamic_df


def select_visualization_timestamp(
    iem_dynamic_df: pd.DataFrame,
    prefer_timestamp: Optional[str] = None,
) -> pd.Timestamp:

    if iem_dynamic_df.empty:
        raise ValueError(
            "iem_dynamic_df is empty; cannot select timestamp for visualization"
        )

    if prefer_timestamp is not None:
        requested_ts = pd.to_datetime(prefer_timestamp, utc=True, errors="coerce")
        if pd.isna(requested_ts):
            log_reduction_reason(
                "Visualization timestamp",
                f"requested timestamp parse failed -> {prefer_timestamp}; falling back to automatic selection",
            )
        else:
            if requested_ts in iem_dynamic_df.index:
                log_reduction_reason(
                    "Visualization timestamp",
                    f"using user-provided timestamp exactly: {requested_ts}",
                )
                return requested_ts

            nearest_pos = iem_dynamic_df.index.get_indexer(
                [requested_ts], method="nearest"
            )[0]
            nearest_ts = iem_dynamic_df.index[nearest_pos]
            log_reduction_reason(
                "Visualization timestamp",
                (
                    f"requested timestamp {requested_ts} not found; "
                    f"using nearest available timestamp {nearest_ts}"
                ),
            )
            return nearest_ts

    candidate = iem_dynamic_df.copy()
    candidate["_std"] = candidate.std(axis=1, skipna=True)
    candidate["_sum"] = candidate.sum(axis=1, skipna=True)
    candidate["_range"] = candidate.max(axis=1, skipna=True) - candidate.min(
        axis=1, skipna=True
    )
    candidate = candidate[(candidate["_sum"] > 0) & candidate["_std"].notna()]
    if candidate.empty:
        fallback_ts = iem_dynamic_df.index[-1]
        log_reduction_reason(
            "Visualization timestamp",
            (
                "no timestamp met variability criteria(sum>0 and std available); "
                f"fallback to last timestamp {fallback_ts}"
            ),
        )
        return fallback_ts

    best_ts = candidate.sort_values(["_std", "_range", "_sum"], ascending=False).index[
        0
    ]
    best_row = candidate.loc[best_ts]
    log_reduction_reason(
        "Visualization timestamp",
        (
            f"auto-selected timestamp {best_ts} with std={best_row['_std']:.4f}, "
            f"range={best_row['_range']:.4f}, sum={best_row['_sum']:.4f}"
        ),
    )
    return best_ts


def select_visualization_timestamps(
    iem_dynamic_df: pd.DataFrame,
    count: int,
    prefer_timestamp: Optional[str] = None,
) -> list[pd.Timestamp]:

    if iem_dynamic_df.empty:
        return []

    count = max(1, int(count))
    selected: list[pd.Timestamp] = []

    if prefer_timestamp is not None:
        selected.append(
            select_visualization_timestamp(
                iem_dynamic_df, prefer_timestamp=prefer_timestamp
            )
        )

    candidate = iem_dynamic_df.copy()
    candidate["_std"] = candidate.std(axis=1, skipna=True)
    candidate["_sum"] = candidate.sum(axis=1, skipna=True)
    candidate["_range"] = candidate.max(axis=1, skipna=True) - candidate.min(
        axis=1, skipna=True
    )
    candidate["_max"] = candidate.max(axis=1, skipna=True)
    candidate["_active_stations"] = candidate.notna().sum(axis=1)
    candidate["_nonzero_stations"] = (candidate.fillna(0) > 0).sum(axis=1)
    candidate = candidate[(candidate["_sum"] > 0) & candidate["_std"].notna()]

    if not candidate.empty:
        sum_thr = float(candidate["_sum"].quantile(0.65))
        max_thr = float(candidate["_max"].quantile(0.65))
        std_thr = float(candidate["_std"].quantile(0.65))
        validated = candidate[
            (candidate["_active_stations"] >= 3)
            & (candidate["_nonzero_stations"] >= 2)
            & (candidate["_sum"] >= sum_thr)
            & (candidate["_max"] >= max_thr)
            & (candidate["_std"] >= std_thr)
        ]
        if not validated.empty:
            candidate = validated

    priority_day = pd.Timestamp(PRIORITY_EVENT_DATE, tz="UTC").date()
    priority_candidate = candidate[candidate.index.date == priority_day]
    if not priority_candidate.empty:
        priority_ts = priority_candidate.sort_values(
            ["_sum", "_max", "_std", "_range"], ascending=False
        ).index[0]
        if priority_ts not in selected:
            selected.append(priority_ts)

    if not candidate.empty:
        ranked = candidate.sort_values(
            ["_sum", "_max", "_std", "_range"], ascending=False
        ).index.tolist()
        for ts in ranked:
            if ts not in selected:
                selected.append(ts)
            if len(selected) >= count:
                break

    if len(selected) < count:
        for ts in iem_dynamic_df.index.tolist():
            if ts not in selected:
                selected.append(ts)
            if len(selected) >= count:
                break

    return selected[:count]


def add_tile_basemap(ax: plt.Axes, alpha: float = 0.8) -> bool:

    if ctx is None:
        log_reduction_reason(
            "Visualization basemap", "contextily unavailable; tile basemap skipped"
        )
        return False

    providers = [
        ("CartoDB.Positron", ctx.providers.CartoDB.Positron),
        ("OpenStreetMap.Mapnik", ctx.providers.OpenStreetMap.Mapnik),
    ]
    for provider_name, provider in providers:
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=provider,
                alpha=alpha,
                reset_extent=False,
                zorder=0,
            )
            log_reduction_reason(
                "Visualization basemap", f"tile basemap loaded from {provider_name}"
            )
            return True
        except Exception as exc:
            log_reduction_reason(
                "Visualization basemap", f"{provider_name} failed: {exc}"
            )

    log_reduction_reason(
        "Visualization basemap",
        "all tile providers failed; plotting without web basemap",
    )
    return False


def find_borough_shapefile(prefer_path: Optional[str] = None) -> Optional[str]:

    if prefer_path and os.path.exists(prefer_path):
        return prefer_path

    shp_paths = glob.glob(DEFAULT_BOROUGH_SHP_GLOB)
    if not shp_paths:
        return None

    priority_keywords = [
        "borough",
        "boro",
        "nybb",
        "boundary",
        "boundaries",
        "geo_export",
    ]
    for shp_path in shp_paths:
        lower = os.path.basename(shp_path).lower()
        if any(k in lower for k in priority_keywords):
            return shp_path

    return shp_paths[0]


def plot_station_sensor_idw_figure(
    location_df: pd.DataFrame,
    iem_static_df: pd.DataFrame,
    iem_dynamic_df: pd.DataFrame,
    precip_df: pd.DataFrame,
    output_path: str,
    borough_shapefile: Optional[str] = None,
    target_timestamp: Optional[pd.Timestamp] = None,
) -> None:

    section("Stage G: Visualization (Station/Sensor + IDW snapshot)")

    if location_df.empty:
        log_reduction_reason("Visualization", "skipped because location_df is empty")
        return
    if iem_static_df.empty or iem_dynamic_df.empty:
        log_reduction_reason("Visualization", "skipped because station data is empty")
        return

    ts = (
        target_timestamp
        if target_timestamp is not None
        else select_visualization_timestamp(iem_dynamic_df)
    )
    ts = pd.to_datetime(ts, utc=True)
    if ts not in iem_dynamic_df.index:
        nearest_station_pos = iem_dynamic_df.index.get_indexer([ts], method="nearest")[
            0
        ]
        station_ts = iem_dynamic_df.index[nearest_station_pos]
        log_reduction_reason(
            "Visualization",
            f"requested timestamp {ts} not found in station series; using nearest {station_ts}",
        )
    else:
        station_ts = ts

    station_row = iem_dynamic_df.loc[station_ts]
    station_snap = iem_static_df.copy()
    station_snap["precip"] = station_snap["station"].map(station_row.to_dict())
    station_snap["precip"] = pd.to_numeric(station_snap["precip"], errors="coerce")

    valid_station = station_snap.dropna(subset=["lat", "lon", "precip"])
    if valid_station.empty:
        log_reduction_reason(
            "Visualization",
            f"skipped because station precip is all NaN at {station_ts}",
        )
        return

    sensor_merged = location_df[["slug", "latitude", "longitude"]].copy()
    sensor_coords = sensor_merged[["latitude", "longitude"]].to_numpy()
    station_coords = valid_station[["lat", "lon"]].to_numpy()
    station_values = valid_station["precip"].to_numpy()
    sensor_merged["precip"] = idw_interpolation(
        sensor_coords, station_coords, station_values, power=2.0
    )

    bounds_lon_min = min(location_df["longitude"].min(), station_snap["lon"].min())
    bounds_lon_max = max(location_df["longitude"].max(), station_snap["lon"].max())
    bounds_lat_min = min(location_df["latitude"].min(), station_snap["lat"].min())
    bounds_lat_max = max(location_df["latitude"].max(), station_snap["lat"].max())

    lon_pad = max((bounds_lon_max - bounds_lon_min) * 0.08, 0.03)
    lat_pad = max((bounds_lat_max - bounds_lat_min) * 0.08, 0.02)
    extent = (
        bounds_lon_min - lon_pad,
        bounds_lon_max + lon_pad,
        bounds_lat_min - lat_pad,
        bounds_lat_max + lat_pad,
    )

    borough_gdf = None
    resolved_shp = find_borough_shapefile(prefer_path=borough_shapefile)
    if resolved_shp is None:
        log_reduction_reason(
            "Visualization basemap",
            "borough shapefile not found under data/map/*.shp; plotting sensors/stations only",
        )
    elif gpd is None:
        log_reduction_reason(
            "Visualization basemap",
            f"geopandas unavailable; cannot render shapefile ({resolved_shp})",
        )
    else:
        try:
            borough_gdf = gpd.read_file(resolved_shp)
            if borough_gdf.crs is not None:
                borough_gdf = borough_gdf.to_crs("EPSG:4326")
            log_reduction_reason(
                "Visualization basemap", f"loaded shapefile: {resolved_shp}"
            )
        except Exception as exc:
            log_reduction_reason(
                "Visualization basemap",
                f"failed to load shapefile {resolved_shp}: {exc}",
            )
            borough_gdf = None

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    panel_a, panel_b = axes
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    panel_a.set_facecolor("#f7f8fa")
    panel_b.set_facecolor("#f7f8fa")

    if borough_gdf is not None and not borough_gdf.empty:
        borough_gdf.boundary.plot(
            ax=panel_a, color="#555555", linewidth=0.95, alpha=0.85, zorder=2
        )

    panel_a.scatter(
        location_df["longitude"],
        location_df["latitude"],
        s=22,
        c="#2f78b7",
        alpha=0.8,
        edgecolors="#ffffff",
        linewidths=0.3,
        label=f"FloodNet sensors (n={location_df['slug'].nunique():,})",
    )
    panel_a.scatter(
        station_snap["lon"],
        station_snap["lat"],
        s=205,
        c="#ff6b4a",
        marker="^",
        edgecolors="#1f1f1f",
        linewidths=1.1,
        label="ASOS stations (5)",
    )
    station_text_offsets = {
        "NYC": (-14, 6),
        "LGA": (10, 6),
        "JFK": (-12, 4),
        "EWR": (8, 4),
        "HPN": (-10, 4),
    }
    station_text_align = {
        "NYC": "right",
        "JFK": "right",
        "HPN": "right",
    }
    for _, row in station_snap.iterrows():
        station_name = str(row["station"])
        offset = station_text_offsets.get(station_name, (8, 4))
        ha = station_text_align.get(station_name, "left")
        ann = panel_a.annotate(
            row["station"],
            (row["lon"], row["lat"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha=ha,
            zorder=8,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )
        ann.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

    panel_a.set_xlim(extent[0], extent[1])
    panel_a.set_ylim(extent[2], extent[3])
    panel_a.set_xlabel("Longitude")
    panel_a.set_ylabel("Latitude")
    panel_a.set_title("Station + Sensor Distribution", pad=10)
    panel_a.legend(loc="upper left", frameon=True, framealpha=0.92, borderpad=0.5)
    panel_a.grid(alpha=0.2, linewidth=0.7)
    panel_a.text(
        0.015,
        0.985,
        "(a)",
        transform=panel_a.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    for spine in panel_a.spines.values():
        spine.set_linewidth(1.1)
    map_a_ok = add_tile_basemap(panel_a, alpha=0.72)
    if not map_a_ok and borough_gdf is None:
        panel_a.text(
            0.02,
            0.98,
            "No basemap (install contextily or check network)",
            transform=panel_a.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#555555",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "none",
            },
        )

    sensor_merged = sensor_merged.dropna(subset=["latitude", "longitude", "precip"])
    if sensor_merged.empty:
        log_reduction_reason(
            "Visualization",
            f"skipped because merged sensor snapshot is empty at {station_ts}",
        )
        plt.close(fig)
        return

    lon_grid = np.linspace(extent[0], extent[1], 240)
    lat_grid = np.linspace(extent[2], extent[3], 240)
    grid_lon_2d, grid_lat_2d = np.meshgrid(lon_grid, lat_grid)
    grid_targets = np.column_stack([grid_lat_2d.ravel(), grid_lon_2d.ravel()])
    grid_precip = idw_interpolation(
        grid_targets, station_coords, station_values, power=2.0
    ).reshape(grid_lon_2d.shape)

    vmin = float(
        np.nanmin(
            [grid_precip.min(), sensor_merged["precip"].min(), station_values.min()]
        )
    )
    vmax = float(
        np.nanmax(
            [grid_precip.max(), sensor_merged["precip"].max(), station_values.max()]
        )
    )
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    mesh = panel_b.pcolormesh(
        grid_lon_2d,
        grid_lat_2d,
        grid_precip,
        cmap="cividis",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        alpha=0.35,
        zorder=1,
    )
    panel_b.contour(
        grid_lon_2d,
        grid_lat_2d,
        grid_precip,
        levels=10,
        colors="#3b3b3b",
        linewidths=0.35,
        alpha=0.35,
        zorder=2,
    )

    sc = panel_b.scatter(
        sensor_merged["longitude"],
        sensor_merged["latitude"],
        c=sensor_merged["precip"],
        s=24,
        cmap="cividis",
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
        edgecolors="#ffffff",
        linewidths=0.2,
        label="IDW at sensors",
        zorder=3,
    )
    station_draw = panel_b.scatter(
        station_snap["lon"],
        station_snap["lat"],
        c=station_snap["precip"],
        s=220,
        cmap="cividis",
        vmin=vmin,
        vmax=vmax,
        marker="*",
        edgecolors="#111111",
        linewidths=1.1,
        label="Station observation",
        zorder=4,
    )
    for _, row in station_snap.iterrows():
        station_name = str(row["station"])
        offset = station_text_offsets.get(station_name, (8, 4))
        ha = station_text_align.get(station_name, "left")
        value_text = "NaN" if pd.isna(row["precip"]) else f"{row['precip']:.2f}"
        ann = panel_b.annotate(
            f"{row['station']} ({value_text})",
            (row["lon"], row["lat"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            ha=ha,
            zorder=8,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )
        ann.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

    if borough_gdf is not None and not borough_gdf.empty:
        borough_gdf.boundary.plot(
            ax=panel_b, color="#4d4d4d", linewidth=0.95, alpha=0.85, zorder=2
        )

    panel_b.set_xlim(extent[0], extent[1])
    panel_b.set_ylim(extent[2], extent[3])
    panel_b.set_xlabel("Longitude")
    panel_b.set_ylabel("Latitude")
    panel_b.set_title(
        f"IDW Snapshot at {pd.to_datetime(station_ts).strftime('%Y-%m-%d %H:%M UTC')}",
        pad=10,
    )
    panel_b.grid(alpha=0.2, linewidth=0.7)
    panel_b.legend(
        handles=[station_draw],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        borderpad=0.5,
    )
    panel_b.text(
        0.015,
        0.985,
        "(b)",
        transform=panel_b.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    for spine in panel_b.spines.values():
        spine.set_linewidth(1.1)
    map_b_ok = add_tile_basemap(panel_b, alpha=0.62)
    if not map_b_ok and borough_gdf is None:
        panel_b.text(
            0.02,
            0.98,
            "No basemap (install contextily or check network)",
            transform=panel_b.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#555555",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "none",
            },
        )

    cbar = fig.colorbar(mesh, ax=panel_b, fraction=0.046, pad=0.04)
    cbar.set_label("Precipitation (mm/min)")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log_reduction_reason(
        "Visualization output",
        (
            f"saved figure to {output_path}; panel(a)=stations+sensors, "
            f"panel(b)=full-grid IDW + sensor dots + station observations at {station_ts}"
        ),
    )


def build_figure_output_paths(base_path: str, count: int) -> list[str]:

    count = max(1, int(count))
    if count == 1:
        return [base_path]

    root, ext = os.path.splitext(base_path)
    if ext == "":
        ext = ".png"
    return [f"{root}_ex{i + 1}{ext}" for i in range(count)]


def create_precip_df(
    location_df: pd.DataFrame,
    precip_dir: str,
    id_col: str = "slug",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    power: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    iem_static_df, iem_dynamic_df = create_iem_asos_df(precip_dir)
    floodnet_coords = location_df[[lat_col, lon_col]].values
    iem_coords = iem_static_df[["lat", "lon"]].values
    sensor_ids = location_df[id_col].values

    n_sensors = len(sensor_ids)
    timestamps_list = []
    sensor_ids_list = []
    precip_values_list = []

    section("Stage D: IDW precipitation interpolation")
    zero_sum_timestamps = 0
    all_nan_timestamps = 0
    for timestamp, row in tqdm(
        iem_dynamic_df.iterrows(), total=len(iem_dynamic_df), desc="IDW"
    ):
        precip_values = row.values
        if np.nansum(precip_values) == 0:
            zero_sum_timestamps += 1
            continue

        valid_mask = ~np.isnan(precip_values)
        if np.any(valid_mask):
            interpolated = idw_interpolation(
                floodnet_coords,
                iem_coords[valid_mask],
                precip_values[valid_mask],
                power,
            )
            timestamps_list.extend([timestamp] * n_sensors)
            sensor_ids_list.extend(sensor_ids)
            precip_values_list.extend(interpolated)
        else:
            all_nan_timestamps += 1

    precip_df = pd.DataFrame(
        {
            id_col: sensor_ids_list,
            "timestamp": timestamps_list,
            "precip": precip_values_list,
        }
    )
    theoretical_rows = n_sensors * len(iem_dynamic_df)
    generated_rows = len(precip_df)
    missing_rows = theoretical_rows - generated_rows
    log(f"Interpolated precip rows={len(precip_df):,}")
    log(
        "Interpolation stats | "
        f"total_timestamps={len(iem_dynamic_df):,}, "
        f"zero_sum_skipped={zero_sum_timestamps:,}, "
        f"all_nan_skipped={all_nan_timestamps:,}, "
        f"used={len(iem_dynamic_df) - zero_sum_timestamps - all_nan_timestamps:,}"
    )
    log_reduction_reason(
        "IDW interpolation",
        (
            f"theoretical_rows={theoretical_rows:,}, generated_rows={generated_rows:,}, missing_rows={missing_rows:,}; "
            "missing rows occur because timestamps with zero total precip or all-NaN station values are skipped"
        ),
    )
    return precip_df, iem_static_df, iem_dynamic_df


def create_integrated_df(
    floodnet_df: pd.DataFrame, precip_df: pd.DataFrame
) -> pd.DataFrame:

    section("Stage E: Integrate flood + precip")

    floodnet_df = floodnet_df.rename(columns={"time": "timestamp"})
    flood_idx = floodnet_df.set_index(["slug", "timestamp"])
    precip_idx = precip_df.set_index(["slug", "timestamp"])

    out = flood_idx.join(precip_idx, how="outer", sort=False).reset_index()
    out = out.sort_values(["slug", "timestamp"], ignore_index=True)
    summarize_integrated(out, "Integrated")
    return out


def notebook_clean_steps(integrated_df: pd.DataFrame) -> pd.DataFrame:

    section("Stage F: Notebook cleaning steps (tracking only)")

    before = integrated_df

    after_step1 = before[~(before["depth_proc_mm"].isna() & (before["precip"] == 0))]
    summarize_transition(before, after_step1, "Step F1 remove(depth NaN & precip==0)")
    removed_step1 = len(before) - len(after_step1)
    log_reduction_reason(
        "Step F1 remove(depth NaN & precip==0)",
        f"removed rows where depth_proc_mm is NaN AND precip == 0 | removed={removed_step1:,}",
    )
    summarize_integrated(after_step1, "After F1")

    before_step2 = after_step1
    after_step2 = before_step2.groupby("timestamp").filter(
        lambda x: not x["depth_proc_mm"].isna().all()
    )
    summarize_transition(
        before_step2, after_step2, "Step F2 remove timestamps(all depth NaN)"
    )
    removed_ts = (
        before_step2.groupby("timestamp")["depth_proc_mm"]
        .apply(lambda x: x.isna().all())
        .sum()
    )
    log_reduction_reason(
        "Step F2 remove timestamps(all depth NaN)",
        f"removed groups where all sensors at a timestamp had depth_proc_mm NaN | removed_timestamps={removed_ts:,}",
    )
    summarize_integrated(after_step2, "After F2 (final tracked)")

    return after_step2


def run_tracking_pipeline(
    min_positive_ratio: float,
    max_processes: int,
    idw_power: float,
    figure_path: Optional[str],
    borough_shapefile: Optional[str],
    selected_timestamp: Optional[str],
    figure_count: int,
    skip_figure: bool,
    output_dir: str,
    output_format: str,
) -> None:
    """Run preprocessing, optional visualization, and output serialization."""
    id_col = "slug"
    pipeline_start = perf_counter()

    raw_flood = load_floodnet_df(FLOODNET_DIR, max_processes=max_processes)

    section("Stage B: Filter + resample FloodNet")
    filtered_flood, sensor_stats = filter_sensors_by_data_quality(
        raw_flood, min_positive_ratio=min_positive_ratio
    )
    summarize_transition(raw_flood, filtered_flood, "Sensor-quality filter")
    removed_sensors = set(sensor_stats.index) - set(filtered_flood["slug"].unique())
    if removed_sensors:
        removed_rows = raw_flood["slug"].isin(removed_sensors).sum()
        log_reduction_reason(
            "Sensor-quality filter",
            f"rows removed because their slug belonged to filtered-out sensors | removed_rows={removed_rows:,}",
        )
    summarize_flood(filtered_flood, "Flood filtered")

    resampled_flood = resample_to_1minute(filtered_flood)
    summarize_transition(filtered_flood, resampled_flood, "Resample to 1min")
    if len(resampled_flood) < len(filtered_flood):
        log_reduction_reason(
            "Resample to 1min",
            "rows decreased because irregular/duplicate timestamps are aggregated into 1-minute bins by mean",
        )
    elif len(resampled_flood) > len(filtered_flood):
        log_reduction_reason(
            "Resample to 1min",
            "rows increased because resample creates full 1-minute grids between observed min/max timestamps per sensor",
        )
    else:
        log_reduction_reason(
            "Resample to 1min", "row count unchanged after 1-minute aggregation"
        )
    summarize_flood(resampled_flood, "Flood resampled(1min)")

    valid_sensor_ids = resampled_flood["slug"].unique()
    sensor_df = load_valid_sensor_df()
    filtered_sensor_df = sensor_df[sensor_df["slug"].isin(valid_sensor_ids)]
    removed_location_sensors = sensor_df[~sensor_df["slug"].isin(valid_sensor_ids)][
        "slug"
    ].nunique()
    log_reduction_reason(
        "Location sensor alignment",
        (
            f"input deployment sensors={sensor_df['slug'].nunique():,}, valid flood sensors={len(valid_sensor_ids):,}, "
            f"removed_from_location={removed_location_sensors:,}"
        ),
    )

    location_df = create_location_df(filtered_sensor_df, crs_to=CRS, id_col=id_col)
    log(f"Location rows for valid sensors={len(location_df):,}")

    terrain_df = create_terrain_df(location_df, RASTER_FILES, id_col=id_col)
    log(f"Terrain rows for valid sensors={len(terrain_df):,}")

    precip_df, iem_static_df, iem_dynamic_df = create_precip_df(
        location_df,
        PRECIP_DIR,
        id_col=id_col,
        power=idw_power,
    )

    if skip_figure:
        log_reduction_reason("Visualization", "skipped by --skip-figure")
    else:
        default_figure_path = os.path.join(
            "logs",
            f"preprocess_tracking_station_sensor_idw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )
        resolved_figure_path = figure_path or default_figure_path
        timestamps = select_visualization_timestamps(
            iem_dynamic_df,
            count=figure_count,
            prefer_timestamp=selected_timestamp,
        )
        output_paths = build_figure_output_paths(resolved_figure_path, len(timestamps))
        log_reduction_reason(
            "Visualization",
            f"creating {len(output_paths):,} figure(s): {', '.join(output_paths)}",
        )
        for ts, out_path in zip(timestamps, output_paths):
            plot_station_sensor_idw_figure(
                location_df=location_df,
                iem_static_df=iem_static_df,
                iem_dynamic_df=iem_dynamic_df,
                precip_df=precip_df,
                output_path=out_path,
                borough_shapefile=borough_shapefile,
                target_timestamp=ts,
            )

    integrated_df = create_integrated_df(resampled_flood, precip_df)

    final_df = notebook_clean_steps(integrated_df)

    section(f"Stage H: Save processed outputs ({output_format})")
    os.makedirs(output_dir, exist_ok=True)

    def save_df(
        df: pd.DataFrame, name: str, label: str, keep_index: bool = False
    ) -> str:
        extension = "parquet" if output_format == "parquet" else "csv"
        path = os.path.join(output_dir, f"{name}.{extension}")
        if output_format == "parquet":
            df.to_parquet(
                path, compression="zstd", compression_level=9, index=keep_index
            )
        else:
            df.to_csv(path, index=keep_index)
        size_mb = os.path.getsize(path) / 1e6
        log(f"Saved {label:<22s} -> {path} ({size_mb:.1f} MB, {len(df):,} rows)")
        return path

    save_df(final_df, name="cleaned_integrated", label="final dataset")

    save_df(location_df, name="flood_sensor_locations", label="sensor locations")

    save_df(terrain_df, name="flood_sensor_terrain", label="sensor terrain")

    save_df(resampled_flood, name="floodnet", label="flood")

    save_df(precip_df, name="precip", label="precip")

    section("Tracking finished")
    log(f"Outputs written under: {output_dir}")
    summarize_integrated(final_df, "Final tracked dataset")
    log(f"Total runtime={perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track preprocessing transformations without saving dataset files."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path. Default: logs/preprocess_tracking_<timestamp>.log",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable file logging and print logs to console only.",
    )
    parser.add_argument(
        "--min-positive-ratio",
        type=float,
        default=0.0,
        help="Minimum ratio of depth_proc_mm > 0 per sensor (e.g., 0.1).",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=12,
        help="Maximum worker processes for raw FloodNet file loading.",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="Power parameter used in IDW interpolation.",
    )
    parser.add_argument(
        "--figure-path",
        type=str,
        default=None,
        help="Output path for the 2-panel station/sensor + IDW figure.",
    )
    parser.add_argument(
        "--borough-shapefile",
        type=str,
        default=None,
        help="Optional NYC borough boundary shapefile path (.shp).",
    )
    parser.add_argument(
        "--selected-timestamp",
        type=str,
        default=None,
        help="Optional UTC timestamp for panel (b), e.g. 2023-09-29 12:00:00+00:00.",
    )
    parser.add_argument(
        "--figure-count",
        type=int,
        default=5,
        help="Number of example figures to generate (default: 5).",
    )
    parser.add_argument(
        "--skip-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "시각화 PNG 생성을 건너뛸지 여부 (기본: True = 건너뜀). "
            "시각화를 만들고 싶으면 --no-skip-figure 를 붙이세요."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "preprocessing"),
        help="저장 결과 디렉터리 (기본: outputs/preprocessing)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Stage H 산출물 파일 형식 (기본: parquet)",
    )
    args = parser.parse_args()

    log_stream: Optional[TextIO] = None
    if not args.no_log_file:
        default_name = (
            f"preprocess_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_path = args.log_file or os.path.join("logs", default_name)
        log_stream = setup_dual_logging(log_path)
        print(f"[logging] console + file enabled: {log_path}")
    else:
        print("[logging] console-only mode enabled (--no-log-file)")

    section("Preprocess tracking run")
    log(
        "Config | "
        f"log_file={'disabled' if args.no_log_file else (args.log_file or log_path)}, "
        f"min_positive_ratio={args.min_positive_ratio}, "
        f"max_processes={args.max_processes}, "
        f"idw_power={args.idw_power}, "
        f"figure_path={args.figure_path}, "
        f"borough_shapefile={args.borough_shapefile}, "
        f"selected_timestamp={args.selected_timestamp}, "
        f"figure_count={args.figure_count}, "
        f"skip_figure={args.skip_figure}, "
        f"output_dir={args.output_dir}, "
        f"output_format={args.output_format}"
    )
    run_tracking_pipeline(
        min_positive_ratio=args.min_positive_ratio,
        max_processes=args.max_processes,
        idw_power=args.idw_power,
        figure_path=args.figure_path,
        borough_shapefile=args.borough_shapefile,
        selected_timestamp=args.selected_timestamp,
        figure_count=args.figure_count,
        skip_figure=args.skip_figure,
        output_dir=args.output_dir,
        output_format=args.output_format,
    )

    if log_stream is not None:
        log_stream.close()
