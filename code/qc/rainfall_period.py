"""Build global rainfall periods and assign timestamps to them."""

import numpy as np
import pandas as pd


def compute_rainfall_periods(integrated_path: str) -> pd.DataFrame:
    """Build the period table from the global set of rain-observed (precip>0) timestamps."""

    if integrated_path.endswith(".parquet"):
        df = pd.read_parquet(
            integrated_path,
            columns=["timestamp", "precip"],
            filters=[("precip", ">", 0)],
        )
    else:
        df = pd.read_csv(
            integrated_path, usecols=["timestamp", "precip"], engine="pyarrow"
        )
        df = df[df["precip"] > 0]

    ts = pd.DatetimeIndex(pd.unique(df["timestamp"])).sort_values()

    gap = ts.to_series().diff().dt.total_seconds()
    breaks = gap > 3600
    breaks.iloc[0] = False
    period_id = breaks.cumsum().to_numpy()

    per = pd.DataFrame({"timestamp": ts, "period_id": period_id})
    table = (
        per.groupby("period_id")["timestamp"].agg(start="min", end="max").reset_index()
    )
    return table


def assign_period(timestamps, periods: pd.DataFrame):
    """For each timestamp, find which period's [start, end] span it falls in (None if none)."""
    starts = pd.DatetimeIndex(periods["start"])
    ends = periods["end"].to_numpy()  # datetime64 (UTC)
    ids = periods["period_id"].to_numpy()
    ts = pd.DatetimeIndex(timestamps)

    idx = starts.searchsorted(ts, side="right") - 1
    safe = np.clip(idx, 0, len(periods) - 1)

    valid = (idx >= 0) & (ts.to_numpy() <= ends[safe])

    out = pd.array(np.where(valid, ids[safe], -1), dtype="Int32")
    out[~valid] = pd.NA
    return out
