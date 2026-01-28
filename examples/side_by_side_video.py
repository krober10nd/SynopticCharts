"""Side-by-side GFS vs ECMWF video example.

This example generates a scan-friendly side-by-side comparison video for the
same model initialization time (UTC) and lead-time sequence.

Workflow:
  1) Generate frames for GFS
  2) Generate frames for ECMWF
  3) Stitch each matching lead time into a single side-by-side PNG
  4) Encode stitched frames into an MP4 (requires ffmpeg)

Output:
  - output/compare_gfs_frames/           (individual GFS frames)
  - output/compare_ecmwf_frames/         (individual ECMWF frames)
  - output/compare_side_by_side_frames/  (stitched frames)
  - output/gfs_vs_ecmwf_side_by_side.mp4

Notes:
  - ECMWF does not provide categorical precip-type masks like GFS. In this
    codebase, ECMWF precipitation is rendered as total precip using the rain
    colormap (a styling choice).
  - For stability on macOS, this script forces the Matplotlib backend to Agg.

Prerequisites:
  - ffmpeg installed (macOS: `brew install ffmpeg`)

"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# IMPORTANT (macOS): avoid GUI backends when running in parallel.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

from synoptic_charts import BatchChartGenerator, Config
from synoptic_charts.api import get_available_lead_times
from synoptic_charts.video import VideoGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRun:
    name: str
    forecast_cycle: datetime
    output_dir: Path


def parse_cycle_yyyymmddhh(value: str) -> datetime:
    """Parse a UTC cycle string like '2026012700' into a datetime."""
    value = (value or "").strip()
    if not re.fullmatch(r"\d{10}", value):
        raise ValueError("cycle must be in format YYYYMMDDHH (e.g., 2026012700)")
    year = int(value[0:4])
    month = int(value[4:6])
    day = int(value[6:8])
    hour = int(value[8:10])
    return datetime(year, month, day, hour)


def validate_cycle_for_model(model: str, cycle: datetime) -> None:
    """Validate the init time is a plausible cycle hour for the model."""
    model_u = (model or "").upper()
    if model_u == "ECMWF" and cycle.hour not in (0, 12):
        raise ValueError(
            "ECMWF cycles are typically 00Z/12Z only. "
            f"Got hour={cycle.hour:02d}Z. Use 00 or 12."
        )
    if model_u == "GFS" and cycle.hour not in (0, 6, 12, 18):
        raise ValueError(
            "GFS cycles are typically 00Z/06Z/12Z/18Z. "
            f"Got hour={cycle.hour:02d}Z. Use 00, 06, 12, or 18."
        )


def common_lead_times(
    *,
    start_hour: int,
    end_hour: int,
    interval: int,
    models: Iterable[str],
) -> List[int]:
    """Return lead times that exist for all listed models."""
    requested = list(range(start_hour, end_hour + 1, interval))
    if not requested:
        return []

    common: Optional[set[int]] = None
    for model in models:
        available = set(get_available_lead_times(model))
        these = set(requested) & available
        common = these if common is None else (common & these)

    return sorted(common or set())


def _lead_time_from_path(path: str | Path) -> Optional[int]:
    """Extract lead time from a BatchChartGenerator frame filename."""
    try:
        stem = Path(path).stem  # frame_024
        return int(stem.split("_")[-1])
    except Exception:
        return None


def generate_frames_for_model(
    *,
    model: str,
    forecast_cycle: datetime,
    region: str,
    config: Config,
    lead_times: List[int],
    output_dir: Path,
    max_workers: int = 4,
) -> Dict[int, Path]:
    """Generate frames and return a mapping lead_time -> frame_path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = BatchChartGenerator(
        model=model,
        forecast_cycle=forecast_cycle,
        region=region,
        config=config,
        output_dir=output_dir,
    )

    result = batch.generate_frames(
        lead_times=lead_times,
        parallel=True,
        parallel_backend="process",
        max_workers=max_workers,
    )

    mapping: Dict[int, Path] = {}
    for p in result.get("successful_frames", []):
        lt = _lead_time_from_path(p)
        if lt is not None:
            mapping[lt] = Path(p)

    return mapping


def stitch_side_by_side(
    *,
    left_path: Path,
    right_path: Path,
    output_path: Path,
    header_left: str,
    header_right: str,
    header_center: str,
) -> None:
    """Create a side-by-side PNG with a simple header bar."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise RuntimeError(
            "Pillow is required for side-by-side stitching. Install with: pip install pillow"
        ) from e

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    # Normalize heights if they differ.
    if left.height != right.height:
        target_h = max(left.height, right.height)
        left = left.resize((int(left.width * target_h / left.height), target_h), resample=Image.BICUBIC)
        right = right.resize((int(right.width * target_h / right.height), target_h), resample=Image.BICUBIC)

    header_h = max(48, left.height // 18)
    gap_px = max(6, (left.width + right.width) // 400)
    bg = (12, 15, 20)
    out = Image.new(
        "RGB",
        (left.width + gap_px + right.width, left.height + header_h),
        color=bg,
    )

    out.paste(left, (0, header_h))
    out.paste(right, (left.width + gap_px, header_h))

    draw = ImageDraw.Draw(out)

    # Font: use default bitmap font to avoid platform-specific font dependencies.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad_x = 12
    pad_y = 10
    text_color = (230, 237, 243)

    # Left / center / right header strings.
    draw.text((pad_x, pad_y), header_left, fill=text_color, font=font)

    # Center align (best-effort).
    if header_center:
        try:
            w = draw.textlength(header_center, font=font)
        except Exception:
            w = len(header_center) * 6
        draw.text(((out.width - w) / 2, pad_y), header_center, fill=text_color, font=font)

    if header_right:
        try:
            w_r = draw.textlength(header_right, font=font)
        except Exception:
            w_r = len(header_right) * 6
        draw.text((out.width - w_r - pad_x, pad_y), header_right, fill=text_color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path, format="PNG", optimize=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a side-by-side GFS vs ECMWF comparison MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--cycle",
        default="2026012300",
        help="Initialization time (UTC) as YYYYMMDDHH (e.g., 2026012300)",
    )
    parser.add_argument(
        "--region",
        default="CONUS",
        help="Built-in region name (see docs/configuration.md)",
    )
    parser.add_argument("--start-hour", type=int, default=0, help="First lead time (hours)")
    parser.add_argument("--end-hour", type=int, default=24 * 5, help="Last lead time (hours)")
    parser.add_argument("--interval", type=int, default=6, help="Lead time spacing (hours)")

    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("high_quality.yaml")),
        help="Path to a SynopticCharts YAML config",
    )
    parser.add_argument(
        "--precip-mode",
        choices=["rate", "accumulated"],
        default="rate",
        help="Precipitation rendering mode",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel worker processes")

    parser.add_argument("--fps", type=int, default=5, help="Video frames per second")
    parser.add_argument(
        "--video-path",
        default="output/gfs_vs_ecmwf_side_by_side.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output frame directories before regenerating",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _build_arg_parser().parse_args()

    forecast_cycle = parse_cycle_yyyymmddhh(args.cycle)
    region = args.region

    start_hour = args.start_hour
    end_hour = args.end_hour
    interval = args.interval

    config_path = Path(args.config)
    config = Config.load_from_file(config_path)
    config.precip_mode = args.precip_mode

    max_workers = args.max_workers

    fps = args.fps

    # Keep rate vs accumulated outputs separate by default.
    default_video = Path("output/gfs_vs_ecmwf_side_by_side.mp4")
    requested_video = Path(args.video_path)
    if requested_video == default_video and str(args.precip_mode).lower() == "accumulated":
        video_path = Path("output/gfs_vs_ecmwf_side_by_side_accum.mp4")
    else:
        video_path = requested_video

    validate_cycle_for_model("GFS", forecast_cycle)
    validate_cycle_for_model("ECMWF", forecast_cycle)

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

    leads = common_lead_times(
        start_hour=start_hour,
        end_hour=end_hour,
        interval=interval,
        models=["GFS", "ECMWF"],
    )
    if not leads:
        raise RuntimeError(
            "No common lead times found for the requested range. "
            "Try a different end_hour/interval."
        )

    print("Side-by-side model comparison")
    print("=" * 72)
    print(f"Init time (UTC): {forecast_cycle:%Y-%m-%d %HZ}")
    print(f"Region: {region}")
    print(f"Precip mode: {config.precip_mode}")
    print(f"Lead times: F{leads[0]:03d} .. F{leads[-1]:03d} every {interval}h ({len(leads)} frames)")
    print(f"Output video: {video_path}")
    print()

    run_tag = f"{region.lower()}_{str(config.precip_mode).lower()}_{forecast_cycle:%Y%m%d%H}"

    gfs_run = ModelRun(
        name="GFS",
        forecast_cycle=forecast_cycle,
        output_dir=Path(f"output/compare_{run_tag}_gfs_frames"),
    )
    ecmwf_run = ModelRun(
        name="ECMWF",
        forecast_cycle=forecast_cycle,
        output_dir=Path(f"output/compare_{run_tag}_ecmwf_frames"),
    )

    stitched_dir = Path(f"output/compare_{run_tag}_side_by_side_frames")

    if args.clean:
        for d in (gfs_run.output_dir, ecmwf_run.output_dir, stitched_dir):
            shutil.rmtree(d, ignore_errors=True)

    print("Step 1: Generating GFS frames")
    gfs_frames = generate_frames_for_model(
        model=gfs_run.name,
        forecast_cycle=gfs_run.forecast_cycle,
        region=region,
        config=config,
        lead_times=leads,
        output_dir=gfs_run.output_dir,
        max_workers=max_workers,
    )
    print(f"  GFS frames: {len(gfs_frames)}/{len(leads)}")

    print("Step 2: Generating ECMWF frames")
    ecmwf_frames = generate_frames_for_model(
        model=ecmwf_run.name,
        forecast_cycle=ecmwf_run.forecast_cycle,
        region=region,
        config=config,
        lead_times=leads,
        output_dir=ecmwf_run.output_dir,
        max_workers=max_workers,
    )
    print(f"  ECMWF frames: {len(ecmwf_frames)}/{len(leads)}")

    print("Step 3: Stitching side-by-side frames")
    stitched_dir.mkdir(parents=True, exist_ok=True)

    stitched_paths: List[Path] = []
    skipped: List[int] = []

    for lt in leads:
        left = gfs_frames.get(lt)
        right = ecmwf_frames.get(lt)
        if left is None or right is None:
            skipped.append(lt)
            continue

        valid_time = forecast_cycle + timedelta(hours=lt)
        header_center = f"Init {forecast_cycle:%Y-%m-%d %HZ}  |  Valid {valid_time:%Y-%m-%d %HZ}  |  F{lt:03d}"

        out_path = stitched_dir / f"frame_{lt:03d}.png"
        stitch_side_by_side(
            left_path=left,
            right_path=right,
            output_path=out_path,
            header_left="GFS",
            header_right="ECMWF",
            header_center=header_center,
        )
        stitched_paths.append(out_path)

    print(f"  Stitched frames: {len(stitched_paths)}")
    if skipped:
        print(f"  Skipped lead times (missing one model): {skipped}")

    if not stitched_paths:
        raise RuntimeError("No stitched frames were created; cannot build video.")

    print("Step 4: Encoding MP4")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    stitched_paths_sorted = sorted(stitched_paths, key=lambda p: _lead_time_from_path(p) or 0)

    VideoGenerator.create_video_from_frames(
        frame_paths=stitched_paths_sorted,
        output_path=video_path,
        fps=fps,
        codec="libx264",
        crf=23,
        preset="medium",
    )

    print("\nDone")
    print(f"  Video: {video_path}")
    print(f"  Frames: {stitched_dir}")


if __name__ == "__main__":
    main()
