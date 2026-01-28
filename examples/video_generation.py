"""Video Generation Example

Creates an animated forecast video for the 24-hour period AFTER the reference
time used by examples/basic_chart.py.

Reference (basic chart):
    - forecast_cycle: yesterday's 06Z cycle (hard-coded reference date)
    - lead_time: F018
    - reference valid time: forecast_cycle + 18h

This script generates frames from F018 through F042 (next 24 hours of valid
time) and then encodes them into an MP4 using ffmpeg.

Prerequisites:
    - ffmpeg installed on system (brew install ffmpeg on macOS)
    - Sufficient disk space for frames

Output:
    - PNG frames in output/frames_after_basic_24h/
    - MP4 video in output/after_basic_24h.mp4
"""

import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# IMPORTANT (macOS): when generating frames in parallel, ensure Matplotlib uses a
# non-GUI backend. Otherwise the default macOS backend may try to create windows
# from worker threads and crash with "NSWindow should only be instantiated on the main thread".
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

from synoptic_charts import BatchChartGenerator, create_video_from_batch, Config


def get_basic_chart_forecast_cycle() -> datetime:
    """Match the forecast cycle selection used by examples/basic_chart.py."""
    now = datetime(2026, 1, 25, 12, 0, 0) - timedelta(days=1)
    cycle_hour = (now.hour // 6) * 6
    cycle = datetime(now.year, now.month, now.day, cycle_hour)
    # Go back one cycle to ensure data availability
    return cycle - timedelta(hours=6)


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

try:
    import tqdm as _tqdm  # noqa: F401
except Exception:
    print("Note: 'tqdm' not installed. Progress bars will be disabled.")

# ============================================================================
# Configuration
# ============================================================================

def main() -> None:
    # Load the same high-quality YAML used by the basic chart example.
    config_path = Path(__file__).with_name("high_quality.yaml")
    config = Config.load_from_file(config_path)

    # Forecast parameters
    model = "GFS"
    forecast_cycle = get_basic_chart_forecast_cycle()
    region = "CONUS"

    # Frame generation settings
    reference_lead_time = 0  # Matches examples/basic_chart.py
    start_hour = reference_lead_time
    end_hour = reference_lead_time + 48 
    interval = 3  # 3-hour intervals (9 frames over a 24h window)

    # Video settings
    output_dir = Path("output/frames_after_basic_24h")
    video_path = Path("output/after_basic_24h.mp4")
    fps = 5  # Frames per second

    print("Video Generation Workflow")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Forecast cycle: {forecast_cycle.strftime('%Y-%m-%d %H:00 UTC')}")
    ref_valid = forecast_cycle + timedelta(hours=reference_lead_time)
    end_valid = forecast_cycle + timedelta(hours=end_hour)
    print(f"Reference (basic chart): F{reference_lead_time:03d} valid {ref_valid.strftime('%Y-%m-%d %H:00 UTC')}")
    print(f"Window: {ref_valid.strftime('%Y-%m-%d %H:00 UTC')} → {end_valid.strftime('%Y-%m-%d %H:00 UTC')}")
    print(f"Forecast range: F{start_hour:03d} - F{end_hour:03d} (every {interval}h)")
    print(f"Expected frames: {len(range(start_hour, end_hour + 1, interval))}")
    print(f"Output directory: {output_dir}")
    print(f"Video output: {video_path}")
    print()

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Check ffmpeg availability early
    if not shutil.which("ffmpeg"):
        print("Warning: ffmpeg not found. Install with: brew install ffmpeg")
        print("Video creation will fail without ffmpeg.")
        print()

    print("Step 1: Generating frames")
    print("-" * 60)

    batch = BatchChartGenerator(
        model=model,
        forecast_cycle=forecast_cycle,
        region=region,
        config=config,
        output_dir=output_dir,
    )

    try:
        # IMPORTANT: Matplotlib/Cartopy is not thread-safe; use process workers.
        result = batch.generate_forecast_sequence(
            start_hour=start_hour,
            end_hour=end_hour,
            interval=interval,
            parallel=True,
            parallel_backend="process",
            max_workers=4,
        )

        successful = len(result["successful_frames"])
        failed = len(result["failed_frames"])
        total = successful + failed
        success_rate = successful / total * 100 if total > 0 else 0

        print()
        print("Batch generation complete!")
        print(f"  Successful: {successful}/{total} frames ({success_rate:.1f}%)")
        print(f"  Total time: {result['total_time']:.1f} seconds")
        if successful > 0:
            print(f"  Average: {result['total_time']/successful:.1f} seconds per frame")

        if failed > 0:
            print(f"  Failed frames: {result['failed_frames']}")
            print("  Warning: Video will have gaps at failed frames")

        if successful == 0:
            print("\nError: No frames generated. Cannot create video.")
            raise SystemExit(1)

    except Exception as e:
        print(f"Error during frame generation: {e}")
        print("\nCommon issues:")
        print("  - Network connectivity required for data download")
        print("  - Insufficient disk space for frames")
        print("  - Model data not yet available")
        raise SystemExit(1)

    print("\nStep 2: Creating video")
    print("-" * 60)

    try:
        video_output = create_video_from_batch(
            batch_result=result,
            output_path=video_path,
            fps=fps,
            codec="libx264",
            crf=23,
            preset="medium",
        )

        video_size = video_path.stat().st_size / (1024 * 1024)
        print("\nVideo creation complete!")
        print(f"  Output: {video_output}")
        print(f"  Size: {video_size:.1f} MB")
        print(f"  Frames: {len(result['successful_frames'])}")
        print(f"  Frame rate: {fps} fps")
        print(f"  Duration: ~{len(result['successful_frames'])/fps:.1f} seconds")

    except Exception as e:
        print(f"Error during video creation: {e}")
        print("\nCommon issues:")
        print("  - ffmpeg not installed (install with: brew install ffmpeg)")
        print("  - Insufficient disk space")
        print("  - Codec not supported (try codec='mpeg4')")
        raise SystemExit(1)

    print("\nStep 3: Cleanup options")
    print("-" * 60)
    cleanup = False

    if cleanup:
        deleted = batch.cleanup_frames()
        print(f"Deleted {deleted} frame files to save disk space")
    else:
        print(f"Frames preserved in: {output_dir}")
        print("To delete frames later, run:")
        print(f"  rm -r {output_dir}")

    print("\n" + "=" * 60)
    print("Video generation complete!")
    print(f"View your animation: {video_path}\n")


if __name__ == "__main__":
    main()

# ============================================================================
# Tips for Customization
# ============================================================================

"""
Video Quality Tips:
-------------------
1. Higher DPI (200-300) for better quality, but larger files
2. Lower CRF (18-20) for higher quality, but larger files
3. Slower preset (slow/veryslow) for better compression

Performance Tips:
-----------------
1. Enable parallel processing for faster frame generation
2. Adjust max_workers based on CPU cores
3. Use smaller forecast range (0-24h) for quick tests

Animation Tips:
--------------
1. fps=10: Smooth, standard animation
2. fps=5: Slower, easier to see details
3. fps=15-20: Fast, good for long forecast periods

Troubleshooting:
----------------
- If frames fail: Check network connection and model data availability
- If video fails: Ensure ffmpeg is installed and frames exist
- If too slow: Reduce forecast range or disable parallel processing
"""
