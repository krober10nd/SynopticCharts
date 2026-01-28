"""Side-by-side GFS vs ECMWF video generation.

This module powers the CLI subcommand that generates scan-friendly side-by-side
comparison videos between two models (typically GFS vs ECMWF).

It intentionally avoids importing from the examples directory so it can be used
as a stable library feature.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from .api import get_available_lead_times
from .batch import BatchChartGenerator
from .config import Config
from .exceptions import InvalidParameterError, VideoCreationError
from .video import VideoGenerator

logger = logging.getLogger(__name__)

CycleAlign = Literal["strict", "previous", "next", "nearest"]


def _format_cycle(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H")


def align_cycle_to_model(model: str, cycle: datetime, strategy: CycleAlign) -> datetime:
    """Align an init cycle time to a model's valid cycle hours.

    Args:
        model: "GFS" or "ECMWF"
        cycle: desired init time (naive datetime treated as UTC)
        strategy: "strict" raises if invalid; otherwise adjusts within +/- 24h.
    """
    model_u = (model or "").upper()
    if model_u == "GFS":
        valid_hours = (0, 6, 12, 18)
    elif model_u == "ECMWF":
        valid_hours = (0, 12)
    else:
        raise InvalidParameterError(f"Unknown model '{model}'.")

    if cycle.hour in valid_hours:
        return cycle

    if strategy == "strict":
        raise InvalidParameterError(
            f"{model_u} cycle hour must be one of {valid_hours}. Got {cycle:%Y-%m-%d %HZ}."
        )

    # Build candidate datetimes around the requested day (previous, same, next).
    candidates: List[datetime] = []
    for day_offset in (-1, 0, 1):
        base = datetime(cycle.year, cycle.month, cycle.day) + timedelta(days=day_offset)
        for h in valid_hours:
            candidates.append(base + timedelta(hours=h))

    if strategy == "previous":
        eligible = [c for c in candidates if c <= cycle]
        if not eligible:
            raise InvalidParameterError(f"Unable to align {model_u} cycle {cycle} using 'previous'.")
        return max(eligible)

    if strategy == "next":
        eligible = [c for c in candidates if c >= cycle]
        if not eligible:
            raise InvalidParameterError(f"Unable to align {model_u} cycle {cycle} using 'next'.")
        return min(eligible)

    if strategy == "nearest":
        # Tie-break toward earlier cycles for deterministic behavior.
        candidates_sorted = sorted(candidates, key=lambda c: (abs((c - cycle).total_seconds()), c))
        return candidates_sorted[0]

    raise InvalidParameterError(f"Unknown alignment strategy '{strategy}'.")


def common_lead_times(
    *,
    start_hour: int,
    end_hour: Optional[int],
    interval: int,
    models: Iterable[str],
) -> List[int]:
    """Return lead times (hours) that exist for all models.

    If end_hour is None, uses the maximum common lead time across models.
    """
    models_list = [m for m in models]
    if not models_list:
        return []

    if interval <= 0:
        raise InvalidParameterError("interval must be > 0")
    if start_hour < 0:
        raise InvalidParameterError("start_hour must be >= 0")

    max_common = min(max(get_available_lead_times(m)) for m in models_list)
    effective_end = max_common if end_hour is None else min(end_hour, max_common)
    if effective_end < start_hour:
        return []

    requested = set(range(start_hour, effective_end + 1, interval))
    common: set[int] = set(requested)
    for model in models_list:
        common &= set(get_available_lead_times(model))

    return sorted(common)


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
    parallel: bool,
    max_workers: Optional[int],
    show_progress: bool,
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
        parallel=parallel,
        parallel_backend="process" if parallel else "thread",
        max_workers=max_workers,
        show_progress=show_progress,
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
    gap_px: int = 8,
) -> None:
    """Create a side-by-side PNG with a simple header bar."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise InvalidParameterError(
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
    gap = max(0, int(gap_px))
    bg = (12, 15, 20)
    out = Image.new(
        "RGB",
        (left.width + gap + right.width, left.height + header_h),
        color=bg,
    )

    out.paste(left, (0, header_h))
    out.paste(right, (left.width + gap, header_h))

    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad_x = 12
    pad_y = 10
    text_color = (230, 237, 243)

    draw.text((pad_x, pad_y), header_left, fill=text_color, font=font)

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


def create_side_by_side_video(
    *,
    gfs_cycle: datetime,
    ecmwf_cycle: datetime,
    region: str,
    config: Config,
    precip_mode: Literal["rate", "accumulated"],
    start_hour: int,
    end_hour: Optional[int],
    interval: int,
    fps: int,
    codec: str,
    crf: int,
    preset: str,
    parallel: bool,
    max_workers: Optional[int],
    clean: bool,
    show_progress: bool = True,
    output_video_path: Optional[Path] = None,
    output_root: Path = Path("output"),
    gap_px: int = 8,
) -> Tuple[Path, Path]:
    """Generate a side-by-side comparison MP4 and return (video_path, stitched_frames_dir)."""

    if not shutil.which("ffmpeg"):
        raise VideoCreationError("ffmpeg not found. Install with: brew install ffmpeg")

    leads = common_lead_times(
        start_hour=start_hour,
        end_hour=end_hour,
        interval=interval,
        models=["GFS", "ECMWF"],
    )
    if not leads:
        raise InvalidParameterError("No common lead times found for the requested range.")

    cfg = replace(config, precip_mode=str(precip_mode).lower())

    tag = f"{region.lower()}_{precip_mode}_{_format_cycle(gfs_cycle)}_gfs_{_format_cycle(ecmwf_cycle)}_ecmwf"
    gfs_dir = output_root / f"compare_{tag}_gfs_frames"
    ecmwf_dir = output_root / f"compare_{tag}_ecmwf_frames"
    stitched_dir = output_root / f"compare_{tag}_side_by_side_frames"

    if clean:
        for d in (gfs_dir, ecmwf_dir, stitched_dir):
            if d.exists():
                shutil.rmtree(d)

    logger.info(
        "Generating side-by-side frames: region=%s mode=%s leads=%s..%s step=%sh",
        region,
        precip_mode,
        leads[0],
        leads[-1],
        interval,
    )

    gfs_frames = generate_frames_for_model(
        model="GFS",
        forecast_cycle=gfs_cycle,
        region=region,
        config=cfg,
        lead_times=leads,
        output_dir=gfs_dir,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
    )

    ecmwf_frames = generate_frames_for_model(
        model="ECMWF",
        forecast_cycle=ecmwf_cycle,
        region=region,
        config=cfg,
        lead_times=leads,
        output_dir=ecmwf_dir,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
    )

    stitched_dir.mkdir(parents=True, exist_ok=True)

    stitched_count = 0
    for lt in leads:
        left = gfs_frames.get(lt)
        right = ecmwf_frames.get(lt)
        if left is None or right is None:
            continue

        out = stitched_dir / f"frame_{lt:03d}.png"
        header_left = f"GFS {gfs_cycle:%Y-%m-%d %HZ}"
        header_right = f"ECMWF {ecmwf_cycle:%Y-%m-%d %HZ}"
        header_center = f"F{lt:03d}  ({precip_mode})"

        stitch_side_by_side(
            left_path=left,
            right_path=right,
            output_path=out,
            header_left=header_left,
            header_right=header_right,
            header_center=header_center,
            gap_px=gap_px,
        )
        stitched_count += 1

    if stitched_count == 0:
        raise VideoCreationError("No stitched frames were produced (missing model frames).")

    if output_video_path is None:
        output_video_path = output_root / f"gfs_vs_ecmwf_side_by_side_{tag}.mp4"

    video_gen = VideoGenerator(
        frame_dir=stitched_dir,
        output_path=Path(output_video_path),
        fps=fps,
        codec=codec,
        crf=crf,
        preset=preset,
    )

    video_gen.create_video(overwrite=True)
    return Path(output_video_path), stitched_dir
