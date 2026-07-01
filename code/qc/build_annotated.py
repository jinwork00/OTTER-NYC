"""Build the annotated training grid from validated flood segments and precipitation data."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_pipeline
from rainfall_period import compute_rainfall_periods, assign_period

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLOODNET = REPO_ROOT / "data" / "02_preprocessed" / "floodnet.parquet"
DEFAULT_CLEANED_INTEGRATED = (
    REPO_ROOT / "data" / "02_preprocessed" / "cleaned_integrated.parquet"
)
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "qc"
DEFAULT_ANNOTATED_GRID = REPO_ROOT / "data" / "03_qc_events" / "annotated_grid.parquet"


def ensure_valid_segments(floodnet_path, integrated_parquet, outdir, force=False):
    """Reuse flood_segments_valid.csv if present; else (or with --force) run the pipeline (~10 min)."""
    valid_path = os.path.join(outdir, "flood_segments_valid.csv")
    if os.path.exists(valid_path) and not force:
        print(f"[1/3] valid segments 재사용 / reuse: {valid_path}")
        return valid_path

    print("[1/3] run_pipeline 실행 / running extractor → integrator → processor ...")
    paths = run_pipeline.run(
        input_path=floodnet_path, outdir=outdir, precip_source=integrated_parquet
    )
    return paths.get("valid_segments", valid_path)


def annotate(
    integrated_parquet,
    valid_segments_csv,
    out_path,
    write_csv=False,
    depth_fill_policy="nan",
):

    if depth_fill_policy not in {"nan", "zero"}:
        raise ValueError("depth_fill_policy must be 'nan' or 'zero'")
    integrated_parquet = str(integrated_parquet)
    valid_segments_csv = str(valid_segments_csv)
    out_path = str(out_path)

    print("[2/3] cleaned_integrated 로드 / loading base (slug, timestamp, precip) ...")
    base = pd.read_parquet(integrated_parquet, columns=["slug", "timestamp", "precip"])
    base["slug"] = base["slug"].astype("category")
    base["precip"] = base["precip"].astype("float32")
    print(f"      base rows={len(base):,}")

    # load valid segments: bring both segment_id and the interpolated depth
    valid = pd.read_csv(
        valid_segments_csv,
        usecols=["slug", "timestamp", "segment_id", "depth_proc_mm"],
        dtype={"segment_id": "Int32", "depth_proc_mm": "float32"},
    )
    valid["timestamp"] = pd.to_datetime(valid["timestamp"], utc=True).astype(
        base["timestamp"].dtype
    )
    valid["slug"] = valid["slug"].astype(base["slug"].dtype)
    assert valid["slug"].notna().all(), (
        "valid CSV 에 base 에 없는 slug 존재 / unknown slug found"
    )
    valid = valid.drop_duplicates(["slug", "timestamp"])
    print(f"      valid segment rows = {len(valid):,}")

    # keep depth NaN outside by default, or fill it with 0 mm if requested.
    base = base.merge(valid, on=["slug", "timestamp"], how="left")
    base["segment_id"] = base["segment_id"].astype("Int32")
    n_outside = int(base["depth_proc_mm"].isna().sum())
    if depth_fill_policy == "zero":
        base["depth_proc_mm"] = base["depth_proc_mm"].fillna(0.0)
        print(
            f"      구간 밖 수심 0 채움 / zero-filled outside-segment depth rows = {n_outside:,}"
        )
    else:
        print(
            f"      구간 밖 수심 NaN 유지 / kept outside-segment depth as NaN rows = {n_outside:,}"
        )
    base["depth_proc_mm"] = base["depth_proc_mm"].astype("float32")
    n_seg = int(base["segment_id"].notna().sum())
    n_dep = int(base["depth_proc_mm"].notna().sum())
    print(
        f"      segment_id 부여 / labeled rows = {n_seg:,}  (수심 보유 / with depth = {n_dep:,})"
    )
    print(
        f"      (valid 행 중 base 격자에 없는 시각은 빠짐 / valid rows missing from the "
        f"base grid are dropped: {len(valid) - n_seg:,})"
    )

    print("[3/3] rainfall_period ...")
    periods = compute_rainfall_periods(integrated_parquet)
    print(f"      rainfall periods = {len(periods):,}")
    base["rainfall_period"] = assign_period(base["timestamp"], periods)
    print(
        f"      rainfall_period 부여 / labeled rows = {int(base['rainfall_period'].notna().sum()):,}"
    )

    base = base[
        [
            "slug",
            "timestamp",
            "depth_proc_mm",
            "precip",
            "segment_id",
            "rainfall_period",
        ]
    ]
    base.to_parquet(out_path, index=False)
    if write_csv:
        base.to_csv(out_path.replace(".parquet", ".csv"), index=False)
    print(
        f"[done] 저장 / saved: {out_path}  (rows={len(base):,}, cols={list(base.columns)})"
    )
    return base


def parse_args():
    p = argparse.ArgumentParser(
        description="cleaned_integrated 를 구간 안 보간 수심 + 라벨 형식으로 주석/변환"
    )
    p.add_argument(
        "--floodnet",
        default=str(DEFAULT_FLOODNET),
        help="extractor 입력: floodnet.csv 또는 .parquet (slug, time, depth_proc_mm) "
        "— parquet 권장(빠름) / extractor input (floodnet file, parquet recommended)",
    )
    p.add_argument(
        "--integrated-parquet",
        default=str(DEFAULT_CLEANED_INTEGRATED),
        help="base 로드 + 강수원 (cleaned_integrated.parquet) / base load + precip source",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="파이프라인 산출 폴더 / pipeline outputs",
    )
    p.add_argument(
        "--out", default=str(DEFAULT_ANNOTATED_GRID), help="최종 주석 파일 경로"
    )
    p.add_argument(
        "--csv", action="store_true", help="CSV 도 함께 저장 / also write a .csv"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="outdir 산출물이 있어도 run_pipeline 을 처음부터 다시 실행 / rebuild segments",
    )
    p.add_argument(
        "--depth-fill-policy",
        choices=["nan", "zero"],
        default="nan",
        help="valid segment 밖 depth_proc_mm 처리 방식. 기본 nan=NaN 유지, zero=0 mm 로 채움 "
        "/ outside-segment depth policy: nan keeps NaN (default), zero fills 0 mm",
    )
    p.add_argument(
        "--zero-fill-depth",
        action="store_true",
        help="--depth-fill-policy zero 의 짧은 별칭 / shortcut for --depth-fill-policy zero",
    )
    return p.parse_args()


def main():
    a = parse_args()
    depth_fill_policy = "zero" if a.zero_fill_depth else a.depth_fill_policy
    valid_csv = ensure_valid_segments(
        a.floodnet, a.integrated_parquet, a.outdir, force=a.force
    )
    annotate(
        a.integrated_parquet,
        valid_csv,
        a.out,
        write_csv=a.csv,
        depth_fill_policy=depth_fill_policy,
    )


if __name__ == "__main__":
    main()
