"""Run flood-segment extraction, integration, filtering, and rainfall validation."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flood_segment_extractor import FloodSegmentExtractor, file_columns
from flood_segment_integrator import FloodSegmentIntegrator, IntegratorConfig
from flood_segment_processor import FloodSegmentProcessor, ProcessingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLOODNET = REPO_ROOT / "data" / "02_preprocessed" / "floodnet.parquet"
DEFAULT_CLEANED_INTEGRATED = (
    REPO_ROOT / "data" / "02_preprocessed" / "cleaned_integrated.parquet"
)
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "qc"


def parse_args():
    p = argparse.ArgumentParser(
        description="유효 홍수 이벤트 추출 파이프라인 / flood valid-event pipeline"
    )
    p.add_argument(
        "--input",
        default=str(DEFAULT_FLOODNET),
        help="extractor 입력 (.csv/.parquet): floodnet.* (slug, time, depth_proc_mm) "
        "또는 cleaned_integrated.* — parquet 권장(빠름) "
        "/ extractor input; floodnet or cleaned_integrated, parquet recommended",
    )
    p.add_argument(
        "--outdir", default=str(DEFAULT_OUTDIR), help="출력 폴더 / output directory"
    )

    p.add_argument(
        "--filter-method",
        default="laplace",
        choices=["quantile", "mad", "gaussian", "laplace"],
    )
    p.add_argument(
        "--sigma-n",
        type=float,
        default=3.5,
        help="laplace/gaussian 임계 배수 (임계값 = sigma_n · σ_eq) / threshold multiplier",
    )
    p.add_argument(
        "--sensor-limit-mm",
        type=float,
        default=3000.0,
        help="이 값 초과 수위를 포함한 구간 제거 / drop segments with depth above this",
    )
    p.add_argument(
        "--lookback-hours",
        type=float,
        default=3.0,
        help="피크 전 강수를 확인할 시간 구간 / hours before a peak to check for rain",
    )

    p.add_argument(
        "--precip-source",
        default=str(DEFAULT_CLEANED_INTEGRATED),
        help="강수를 읽을 cleaned_integrated 파일(.csv/.parquet). 미지정 시 --input 사용. "
        "floodnet 입력에는 precip 이 없으므로 그때는 필수 "
        "/ precip file; defaults to --input — REQUIRED when --input is a floodnet file",
    )
    p.add_argument(
        "--precip-path",
        default=None,
        help="(선택) 별도 precip 파일(slug, virtual_timestamp, precip). 주면 우선 사용 "
        "/ optional dedicated precip file; takes priority if given",
    )
    return p.parse_args()


def run(
    input_path,
    outdir=str(DEFAULT_OUTDIR),
    filter_method="laplace",
    sigma_n=3.5,
    sensor_limit_mm=3000.0,
    lookback_hours=3.0,
    precip_source=None,
    precip_path=None,
):
    """Run the 3 stages and return the dict of output paths (incl. 'valid_segments')."""
    os.makedirs(outdir, exist_ok=True)

    precip_source = precip_source or input_path
    if precip_path is None and "precip" not in file_columns(precip_source):
        raise SystemExit(
            f"ERROR: '{precip_source}' 에 precip 컬럼이 없습니다. floodnet 입력을 쓸 때는 "
            "--precip-source 로 cleaned_integrated.parquet 을 지정하세요. "
            "/ no 'precip' column; pass --precip-source cleaned_integrated.parquet"
        )

    seg_csv = os.path.join(outdir, "flood_segments.csv")
    interp_csv = os.path.join(outdir, "flood_segments_interpolated.csv")

    print("\n=== [1/3] EXTRACTOR : floodnet 수심에서 연속 비-0 구간 추출 ===")
    FloodSegmentExtractor(
        input_path=input_path, output_path=seg_csv
    ).extract_all_segments()

    print("\n=== [2/3] INTEGRATOR : 구간 잇기 + 0 보간 + 재세그먼트 ===")
    integ = FloodSegmentIntegrator(
        IntegratorConfig(
            output_dir=outdir,
            interpolated_filename="flood_segments_interpolated_raw.csv",
        )
    )
    integ.load_segments(seg_csv).process()
    integ.save_outputs(outdir)

    di = integ.get_interpolated_segments()
    di = di.drop(columns=["segment_id", "time_diff"], errors="ignore").rename(
        columns={"new_segment_id": "segment_id"}
    )
    di[["slug", "segment_id", "timestamp", "depth_proc_mm"]].to_csv(
        interp_csv, index=False
    )
    n_seg = di.groupby(["slug", "segment_id"]).ngroups
    print(f"  bridge → {interp_csv}  (재세그먼트/segments = {n_seg:,})")

    print("\n=== [3/3] PROCESSOR : laplace 필터 + 피크 + 강수 검증 ===")
    cfg = ProcessingConfig(
        sensor_limit_mm=sensor_limit_mm,
        filter_method=filter_method,
        sigma_n=sigma_n,
        rainfall_lookback_hours=lookback_hours,
        precip_path=precip_path,  # takes priority
        integrated_path=(None if precip_path else precip_source),
        output_dir=outdir,
    )
    proc = FloodSegmentProcessor(cfg)
    proc.load_segments(interp_csv).process()
    paths = proc.save_outputs(outdir)

    print("\n=== DONE ===")
    print(
        f"  filter={cfg.filter_method}  sigma_n={cfg.sigma_n}  "
        f"sensor_limit={cfg.sensor_limit_mm}mm  lookback={cfg.rainfall_lookback_hours}h"
    )
    peak_csv = os.path.join(outdir, cfg.peak_info_filename)
    if os.path.exists(peak_csv):
        pk = pd.read_csv(peak_csv)
        n_valid = int(pk["is_valid_flood_event"].sum()) if len(pk) else 0
        pct = (100 * n_valid / len(pk)) if len(pk) else 0.0
        print(
            f"  peaks = {len(pk):,}  |  valid flood events = {n_valid:,} ({pct:.1f}%)"
        )
    print(f"  outputs → {outdir}")
    print(
        "  files: flood_segments.csv, flood_segments_interpolated.csv, "
        "flood_segments_filtered.csv, peak_info.csv, flood_segments_valid.csv, *_stats.json"
    )
    return paths


def main():
    a = parse_args()
    run(
        a.input,
        a.outdir,
        a.filter_method,
        a.sigma_n,
        a.sensor_limit_mm,
        a.lookback_hours,
        a.precip_source,
        a.precip_path,
    )


if __name__ == "__main__":
    main()
