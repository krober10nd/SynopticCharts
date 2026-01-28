"""
Command-line interface for SynopticCharts package.

Provides argparse-based CLI with subcommands for creating charts, batch
processing, video generation, and combined animation workflows.

Usage:
    synoptic-charts chart --model GFS --cycle 2024011500 --lead-time 24 --output chart.png
    synoptic-charts batch --model GFS --cycle 2024011500 --start-hour 0 --end-hour 120 --output-dir frames/
    synoptic-charts video --frame-dir frames/ --output forecast.mp4
    synoptic-charts animate --model GFS --cycle 2024011500 --output forecast.mp4
"""

import argparse
import io
import os
import sys
import warnings
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional

from dataclasses import replace

from .api import create_chart, get_available_lead_times
from .batch import BatchChartGenerator
from .video import VideoGenerator, create_video_from_batch
from .config import Config
from .constants import MODELS
from .logging_config import setup_logging
from .exceptions import SynopticChartsError
from .compare_video import align_cycle_to_model, create_side_by_side_video


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """
    Configure logging based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments with verbose, quiet, log_file
    """
    if hasattr(args, 'silent') and args.silent:
        verbosity = -2  # ERROR
    elif hasattr(args, 'quiet') and args.quiet:
        verbosity = -1  # WARNING
    elif hasattr(args, 'verbose') and args.verbose:
        verbosity = 1  # DEBUG
    else:
        verbosity = 0  # INFO
    
    log_file = getattr(args, 'log_file', None)
    setup_logging(verbosity=verbosity, log_file=log_file)


def parse_cycle(cycle_str: str) -> datetime:
    """
    Parse forecast cycle string to datetime.
    
    Supports formats:
        - YYYYMMDDHH (e.g., 2024011500)
        - YYYY-MM-DD-HH (e.g., 2024-01-15-00)
        - YYYYMMDD_HH (e.g., 20240115_00)
    
    Args:
        cycle_str: Cycle string to parse
        
    Returns:
        Parsed datetime
        
    Raises:
        argparse.ArgumentTypeError: If format is invalid
    """
    formats = [
        "%Y%m%d%H",         # 2024011500
        "%Y-%m-%d-%H",      # 2024-01-15-00
        "%Y%m%d_%H",        # 20240115_00
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(cycle_str, fmt)
        except ValueError:
            continue
    
    raise argparse.ArgumentTypeError(
        f"Invalid cycle format: {cycle_str}. "
        "Expected YYYYMMDDHH, YYYY-MM-DD-HH, or YYYYMMDD_HH"
    )


def validate_model(model_str: str) -> str:
    """
    Validate model name.
    
    Args:
        model_str: Model name to validate
        
    Returns:
        Validated model name
        
    Raises:
        argparse.ArgumentTypeError: If model is invalid
    """
    if model_str not in MODELS:
        available = ", ".join(MODELS.keys())
        raise argparse.ArgumentTypeError(
            f"Invalid model: {model_str}. Available models: {available}"
        )
    return model_str


def load_config(config_path: Optional[str]) -> Optional[Config]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file (YAML or JSON)
        
    Returns:
        Config object or None if no path provided
    """
    if config_path is None:
        return None
    
    try:
        return Config.load_from_file(config_path)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_print(args: argparse.Namespace, *values: object, **kwargs) -> None:
    """Print unless --silent was provided."""
    if getattr(args, "silent", False):
        return
    print(*values, **kwargs)


@contextmanager
def _silence_third_party_output(args: argparse.Namespace):
    """Capture noisy stdout/stderr in --silent mode.

    We still print explicit CLI outputs (final paths) outside this context.
    On exceptions, captured output is forwarded to stderr to aid debugging.
    """

    if not getattr(args, "silent", False):
        yield
        return

    warnings.filterwarnings("ignore")
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        try:
            yield
        except Exception:
            try:
                sys.stderr.write(buf_err.getvalue())
                sys.stderr.write(buf_out.getvalue())
            except Exception:
                pass
            raise


def cmd_chart(args: argparse.Namespace) -> int:
    """Handle 'chart' subcommand."""
    _cli_print(args, f"Creating chart: {args.model} F{args.lead_time:03d}")
    
    try:
        # Load config
        config = load_config(args.config)
        if config is None:
            config = Config()
        if getattr(args, "background_color", None):
            config.background_color = args.background_color
        if config is None:
            config = Config()
        
        # Override config values if specified
        if args.dpi:
            config.default_dpi = args.dpi
        if args.cache_dir:
            config.cache_dir = args.cache_dir
        if getattr(args, "background_color", None):
            config.background_color = args.background_color
        
        # Create chart
        with _silence_third_party_output(args):
            output_path = create_chart(
                model=args.model,
                forecast_cycle=args.cycle,
                lead_time=args.lead_time,
                region=args.region,
                output_path=args.output,
                config=config
            )
        
        if getattr(args, "silent", False):
            print(str(output_path))
        else:
            print(f"Success! Chart saved to: {output_path}")
        return 0
        
    except SynopticChartsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Handle 'batch' subcommand."""
    _cli_print(
        args,
        f"Generating frames: {args.model} "
        f"F{args.start_hour:03d}-F{args.end_hour:03d} "
        f"every {args.interval}h"
    )
    
    try:
        # Load config
        config = load_config(args.config)
        if config is None:
            config = Config()
        if getattr(args, "background_color", None):
            config.background_color = args.background_color
        
        # Create batch generator
        batch = BatchChartGenerator(
            model=args.model,
            forecast_cycle=args.cycle,
            region=args.region,
            config=config,
            output_dir=Path(args.output_dir)
        )
        
        # Generate frames
        with _silence_third_party_output(args):
            result = batch.generate_forecast_sequence(
                start_hour=args.start_hour,
                end_hour=args.end_hour,
                interval=args.interval,
                parallel=args.parallel,
                max_workers=args.workers,
                show_progress=not (getattr(args, "quiet", False) or getattr(args, "silent", False)),
            )
        
        # Print summary
        successful = len(result['successful_frames'])
        total = successful + len(result['failed_frames'])
        success_rate = successful / total * 100 if total > 0 else 0
        
        if getattr(args, "silent", False):
            print(str(args.output_dir))
        else:
            print(f"\nBatch generation complete!")
            print(f"  Successful: {successful}/{total} ({success_rate:.1f}%)")
            print(f"  Total time: {result['total_time']:.1f}s")
            print(f"  Output directory: {args.output_dir}")
        
        if result['failed_frames'] and not getattr(args, "silent", False):
            _cli_print(args, f"  Failed frames: {result['failed_frames']}")
        
        return 0 if successful > 0 else 1
        
    except SynopticChartsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_video(args: argparse.Namespace) -> int:
    """Handle 'video' subcommand."""
    _cli_print(args, f"Creating video from frames in: {args.frame_dir}")
    
    try:
        # Create video generator
        video_gen = VideoGenerator(
            frame_dir=Path(args.frame_dir),
            output_path=Path(args.output),
            fps=args.fps,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset
        )
        
        # Generate video
        with _silence_third_party_output(args):
            video_path = video_gen.create_video(
                frame_pattern=args.pattern,
                overwrite=args.overwrite
            )
        
        # Get file size
        file_size = Path(video_path).stat().st_size / (1024 * 1024)
        
        if getattr(args, "silent", False):
            print(str(video_path))
        else:
            _cli_print(args, "\nSuccess! Video created:")
            _cli_print(args, f"  Path: {video_path}")
            _cli_print(args, f"  Size: {file_size:.1f} MB")
            _cli_print(args, f"  FPS: {args.fps}")
        
        return 0
        
    except SynopticChartsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_animate(args: argparse.Namespace) -> int:
    """Handle 'animate' subcommand (combined batch + video)."""
    _cli_print(
        args,
        f"Creating animation: {args.model} "
        f"F{args.start_hour:03d}-F{args.end_hour:03d}"
    )
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Determine output directory
        output_dir = args.output_dir if args.output_dir else Path("frames_temp")
        
        # Create batch generator
        _cli_print(args, "\n[1/2] Generating frames...")
        batch = BatchChartGenerator(
            model=args.model,
            forecast_cycle=args.cycle,
            region=args.region,
            config=config,
            output_dir=output_dir
        )
        
        # Generate frames
        with _silence_third_party_output(args):
            result = batch.generate_forecast_sequence(
                start_hour=args.start_hour,
                end_hour=args.end_hour,
                interval=args.interval,
                parallel=args.parallel,
                max_workers=args.workers,
                show_progress=not (getattr(args, "quiet", False) or getattr(args, "silent", False)),
            )
        
        successful = len(result['successful_frames'])
        total = successful + len(result['failed_frames'])
        
        _cli_print(args, f"  Generated {successful}/{total} frames in {result['total_time']:.1f}s")
        
        if successful == 0:
            print("Error: No frames generated", file=sys.stderr)
            return 1
        
        # Create video
        _cli_print(args, "\n[2/2] Creating video...")
        with _silence_third_party_output(args):
            video_path = create_video_from_batch(
                batch_result=result,
                output_path=Path(args.output),
                fps=args.fps,
                codec=args.codec,
                crf=args.crf,
                preset=args.preset
            )
        
        # Get file size
        file_size = Path(video_path).stat().st_size / (1024 * 1024)
        
        if getattr(args, "silent", False):
            print(str(video_path))
        else:
            _cli_print(args, "\nSuccess! Animation created:")
            _cli_print(args, f"  Video: {video_path}")
            _cli_print(args, f"  Size: {file_size:.1f} MB")
            _cli_print(args, f"  Frames: {successful}")
            _cli_print(args, f"  FPS: {args.fps}")
        
        # Cleanup frames if requested
        if args.cleanup_frames:
            _cli_print(args, "\nCleaning up frames...")
            deleted = batch.cleanup_frames()
            _cli_print(args, f"  Deleted {deleted} frames")
        
        return 0
        
    except SynopticChartsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_compare_video(args: argparse.Namespace) -> int:
    """Handle 'compare-video' subcommand (GFS vs ECMWF side-by-side)."""
    try:
        config = load_config(args.config)
        if config is None:
            config = Config()
        if getattr(args, "background_color", None):
            config.background_color = args.background_color

        # Determine model cycles. If a single cycle is provided, align each model.
        base_cycle: datetime = args.cycle
        gfs_cycle = args.gfs_cycle if args.gfs_cycle is not None else align_cycle_to_model("GFS", base_cycle, args.align_gfs)
        ecmwf_cycle = args.ecmwf_cycle if args.ecmwf_cycle is not None else align_cycle_to_model("ECMWF", base_cycle, args.align_ecmwf)

        if gfs_cycle != base_cycle:
            _cli_print(args, f"Note: adjusted GFS cycle to {gfs_cycle:%Y-%m-%d %HZ}")
        if ecmwf_cycle != base_cycle:
            _cli_print(args, f"Note: adjusted ECMWF cycle to {ecmwf_cycle:%Y-%m-%d %HZ}")

        if args.modes == "both":
            modes = ["rate", "accumulated"]
        else:
            modes = [args.modes]

        for mode in modes:
            _cli_print(args, "\nCreating comparison video:")
            _cli_print(args, f"  Mode: {mode}")
            _cli_print(args, f"  GFS cycle:   {gfs_cycle:%Y-%m-%d %HZ}")
            _cli_print(args, f"  ECMWF cycle: {ecmwf_cycle:%Y-%m-%d %HZ}")
            _cli_print(args, f"  Region: {args.region}")
            _cli_print(args, f"  Interval: {args.interval}h")
            _cli_print(args, f"  Start hour: {args.start_hour}")
            _cli_print(args, f"  End hour: {args.end_hour if args.end_hour is not None else 'max common'}")

            with _silence_third_party_output(args):
                video_path, frames_dir = create_side_by_side_video(
                    gfs_cycle=gfs_cycle,
                    ecmwf_cycle=ecmwf_cycle,
                    region=args.region,
                    config=config,
                    precip_mode=mode,
                    start_hour=args.start_hour,
                    end_hour=args.end_hour,
                    interval=args.interval,
                    fps=args.fps,
                    codec=args.codec,
                    crf=args.crf,
                    preset=args.preset,
                    parallel=not args.no_parallel,
                    max_workers=args.workers,
                    clean=args.clean,
                    show_progress=not (getattr(args, "quiet", False) or getattr(args, "silent", False)),
                    output_root=Path(args.output_root),
                    gap_px=args.gap_px,
                    output_video_path=None,
                )

            if getattr(args, "silent", False):
                print(str(video_path))
            else:
                file_size = Path(video_path).stat().st_size / (1024 * 1024)
                _cli_print(args, "\nSuccess!")
                _cli_print(args, f"  Video: {video_path} ({file_size:.1f} MB)")
                _cli_print(args, f"  Frames: {frames_dir}")

        return 0

    except SynopticChartsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_forecast_video(args: argparse.Namespace) -> int:
    """Handle 'forecast-video' subcommand (single model or side-by-side)."""
    try:
        config = load_config(args.config)
        if config is None:
            config = Config()
        if getattr(args, "background_color", None):
            config.background_color = args.background_color

        if args.modes == "both":
            modes = ["rate", "accumulated"]
        else:
            modes = [args.modes]

        output_root = Path(args.output_root)

        model_u = str(args.model).upper()
        if model_u in ("GFS", "ECMWF"):
            cycle = align_cycle_to_model(model_u, args.cycle, args.align)
            if cycle != args.cycle:
                _cli_print(args, f"Note: adjusted {model_u} cycle to {cycle:%Y-%m-%d %HZ}")

            max_available = max(get_available_lead_times(model_u))
            requested_end = args.max_lead
            end_hour_cap = max_available if requested_end is None else min(int(requested_end), int(max_available))
            if requested_end is not None and int(requested_end) > int(max_available):
                _cli_print(args, f"Note: capped max lead to {max_available}h for {model_u}")

            available = set(get_available_lead_times(model_u))
            requested = list(range(int(args.start_hour), int(end_hour_cap) + 1, int(args.interval)))
            lead_times = [lt for lt in requested if lt in available]
            dropped = [lt for lt in requested if lt not in available]
            if not lead_times:
                raise SynopticChartsError(
                    f"No valid lead times for {model_u} in requested range. "
                    f"Try a different --interval or --max-lead."
                )
            actual_end = max(lead_times)
            if dropped:
                _cli_print(
                    args,
                    f"Note: dropping {len(dropped)} unavailable lead times (e.g., {dropped[:5]}...). "
                    f"Last included lead: {actual_end}h"
                )
            elif actual_end != end_hour_cap:
                _cli_print(args, f"Note: last included lead is {actual_end}h")

            for mode in modes:
                cfg = replace(config, precip_mode=mode)

                tag = f"{model_u.lower()}_{args.region.lower()}_{mode}_{cycle:%Y%m%d%H}_f{actual_end:03d}"
                frame_dir = output_root / f"frames_{tag}"
                video_path = output_root / f"{tag}.mp4"

                _cli_print(args, "\nCreating forecast video:")
                _cli_print(args, f"  Model: {model_u}")
                _cli_print(args, f"  Cycle: {cycle:%Y-%m-%d %HZ}")
                _cli_print(args, f"  Region: {args.region}")
                _cli_print(args, f"  Mode: {mode}")
                _cli_print(args, f"  Leads: F{lead_times[0]:03d}..F{actual_end:03d} every {args.interval}h (filtered)")
                _cli_print(args, f"  Output: {video_path}")

                batch = BatchChartGenerator(
                    model=model_u,
                    forecast_cycle=cycle,
                    region=args.region,
                    config=cfg,
                    output_dir=frame_dir,
                )

                with _silence_third_party_output(args):
                    result = batch.generate_frames(
                        lead_times=lead_times,
                        parallel=not args.no_parallel,
                        parallel_backend="process" if not args.no_parallel else "thread",
                        max_workers=args.workers,
                        show_progress=not (getattr(args, "quiet", False) or getattr(args, "silent", False)),
                    )

                successful = len(result.get("successful_frames", []))
                if successful == 0:
                    raise SynopticChartsError(f"No frames generated for {model_u} ({mode}).")

                with _silence_third_party_output(args):
                    create_video_from_batch(
                        batch_result=result,
                        output_path=video_path,
                        fps=args.fps,
                        codec=args.codec,
                        crf=args.crf,
                        preset=args.preset,
                    )

                if getattr(args, "silent", False):
                    print(str(video_path))
                else:
                    file_size = video_path.stat().st_size / (1024 * 1024)
                    _cli_print(args, "\nSuccess!")
                    _cli_print(args, f"  Video: {video_path} ({file_size:.1f} MB)")
                    _cli_print(args, f"  Frames: {frame_dir}")

            return 0

        if model_u in ("COMPARE", "SIDE-BY-SIDE", "SIDE_BY_SIDE"):
            base_cycle: datetime = args.cycle
            gfs_cycle = args.gfs_cycle if args.gfs_cycle is not None else align_cycle_to_model("GFS", base_cycle, args.align_gfs)
            ecmwf_cycle = args.ecmwf_cycle if args.ecmwf_cycle is not None else align_cycle_to_model("ECMWF", base_cycle, args.align_ecmwf)

            if gfs_cycle != base_cycle:
                _cli_print(args, f"Note: adjusted GFS cycle to {gfs_cycle:%Y-%m-%d %HZ}")
            if ecmwf_cycle != base_cycle:
                _cli_print(args, f"Note: adjusted ECMWF cycle to {ecmwf_cycle:%Y-%m-%d %HZ}")

            for mode in modes:
                _cli_print(args, "\nCreating comparison video:")
                _cli_print(args, f"  Mode: {mode}")
                _cli_print(args, f"  GFS cycle:   {gfs_cycle:%Y-%m-%d %HZ}")
                _cli_print(args, f"  ECMWF cycle: {ecmwf_cycle:%Y-%m-%d %HZ}")
                _cli_print(args, f"  Region: {args.region}")
                _cli_print(args, f"  Interval: {args.interval}h")
                _cli_print(args, f"  Start hour: {args.start_hour}")
                _cli_print(args, f"  Max lead: {args.max_lead if args.max_lead is not None else 'max common'}")

                with _silence_third_party_output(args):
                    video_path, frames_dir = create_side_by_side_video(
                        gfs_cycle=gfs_cycle,
                        ecmwf_cycle=ecmwf_cycle,
                        region=args.region,
                        config=config,
                        precip_mode=mode,
                        start_hour=args.start_hour,
                        end_hour=args.max_lead,
                        interval=args.interval,
                        fps=args.fps,
                        codec=args.codec,
                        crf=args.crf,
                        preset=args.preset,
                        parallel=not args.no_parallel,
                        max_workers=args.workers,
                        clean=args.clean,
                        output_root=output_root,
                        gap_px=args.gap_px,
                        show_progress=not (getattr(args, "quiet", False) or getattr(args, "silent", False)),
                        output_video_path=None,
                    )

                if getattr(args, "silent", False):
                    print(str(video_path))
                else:
                    file_size = Path(video_path).stat().st_size / (1024 * 1024)
                    _cli_print(args, "\nSuccess!")
                    _cli_print(args, f"  Video: {video_path} ({file_size:.1f} MB)")
                    _cli_print(args, f"  Frames: {frames_dir}")

            return 0

        raise SynopticChartsError(
            "Invalid --model. Use GFS, ECMWF, or compare."
        )

    except SynopticChartsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="synoptic-charts",
        description="Generate synoptic meteorological charts from model data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    def _add_common_globalish_args(p: argparse.ArgumentParser) -> None:
        """Add args that users reasonably expect to work after subcommands too.

        Argparse only treats options as "global" when they appear before the
        subcommand token. To make UX forgiving, we add these to subparsers as
        well.
        """

        p.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable DEBUG logging"
        )
        p.add_argument(
            "-q", "--quiet",
            action="store_true",
            help="Suppress INFO logging (WARNING+ only)"
        )
        p.add_argument(
            "--silent",
            action="store_true",
            help="Suppress most console output (prints only final output path(s))"
        )
        p.add_argument(
            "--background-color",
            type=str,
            default=None,
            help="Map/figure background color (Matplotlib color spec, e.g. '#1f2328' or 'black')",
        )
        p.add_argument(
            "--log-file",
            type=str,
            help="Write logs to file"
        )
    
    # Global arguments (still supported before subcommands)
    _add_common_globalish_args(parser)
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ========================================================================
    # chart subcommand
    # ========================================================================
    parser_chart = subparsers.add_parser(
        "chart",
        help="Create a single synoptic chart"
    )
    _add_common_globalish_args(parser_chart)
    parser_chart.add_argument(
        "--model",
        type=validate_model,
        required=True,
        help="Model name (GFS, ECMWF)"
    )
    parser_chart.add_argument(
        "--cycle",
        type=parse_cycle,
        required=True,
        help="Forecast cycle (YYYYMMDDHH)"
    )
    parser_chart.add_argument(
        "--lead-time",
        type=int,
        required=True,
        help="Forecast hour"
    )
    parser_chart.add_argument(
        "--region",
        type=str,
        default="CONUS",
        help="Region name (default: CONUS)"
    )
    parser_chart.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    parser_chart.add_argument(
        "--config",
        type=str,
        help="Config file path (YAML/JSON)"
    )
    parser_chart.add_argument(
        "--dpi",
        type=int,
        help="Override DPI setting"
    )
    parser_chart.add_argument(
        "--cache-dir",
        type=str,
        help="Override cache directory"
    )
    parser_chart.set_defaults(func=cmd_chart)
    
    # ========================================================================
    # batch subcommand
    # ========================================================================
    parser_batch = subparsers.add_parser(
        "batch",
        help="Generate multiple frames for a forecast sequence"
    )
    _add_common_globalish_args(parser_batch)
    parser_batch.add_argument(
        "--model",
        type=validate_model,
        required=True,
        help="Model name (GFS, ECMWF)"
    )
    parser_batch.add_argument(
        "--cycle",
        type=parse_cycle,
        required=True,
        help="Forecast cycle (YYYYMMDDHH)"
    )
    parser_batch.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="First forecast hour (default: 0)"
    )
    parser_batch.add_argument(
        "--end-hour",
        type=int,
        default=120,
        help="Last forecast hour (default: 120)"
    )
    parser_batch.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Hour interval between frames (default: 3)"
    )
    parser_batch.add_argument(
        "--region",
        type=str,
        default="CONUS",
        help="Region name (default: CONUS)"
    )
    parser_batch.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for frames"
    )
    parser_batch.add_argument(
        "--config",
        type=str,
        help="Config file path (YAML/JSON)"
    )
    parser_batch.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing"
    )
    parser_batch.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers"
    )
    parser_batch.set_defaults(func=cmd_batch)
    
    # ========================================================================
    # video subcommand
    # ========================================================================
    parser_video = subparsers.add_parser(
        "video",
        help="Create video from existing frames"
    )
    _add_common_globalish_args(parser_video)
    parser_video.add_argument(
        "--frame-dir",
        type=str,
        required=True,
        help="Directory containing frames"
    )
    parser_video.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video file path"
    )
    parser_video.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10)"
    )
    parser_video.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec (default: libx264)"
    )
    parser_video.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Quality setting, 0-51, lower=better (default: 23)"
    )
    parser_video.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="Encoding preset (default: medium)"
    )
    parser_video.add_argument(
        "--pattern",
        type=str,
        default="frame_%03d.png",
        help="Frame filename pattern (default: frame_%%03d.png)"
    )
    parser_video.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing video"
    )
    parser_video.set_defaults(func=cmd_video)
    
    # ========================================================================
    # animate subcommand (batch + video)
    # ========================================================================
    parser_animate = subparsers.add_parser(
        "animate",
        help="Generate frames and create video (all-in-one)"
    )
    _add_common_globalish_args(parser_animate)
    parser_animate.add_argument(
        "--model",
        type=validate_model,
        required=True,
        help="Model name (GFS, ECMWF)"
    )
    parser_animate.add_argument(
        "--cycle",
        type=parse_cycle,
        required=True,
        help="Forecast cycle (YYYYMMDDHH)"
    )
    parser_animate.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="First forecast hour (default: 0)"
    )
    parser_animate.add_argument(
        "--end-hour",
        type=int,
        default=120,
        help="Last forecast hour (default: 120)"
    )
    parser_animate.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Hour interval between frames (default: 3)"
    )
    parser_animate.add_argument(
        "--region",
        type=str,
        default="CONUS",
        help="Region name (default: CONUS)"
    )
    parser_animate.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video file path"
    )
    parser_animate.add_argument(
        "--output-dir",
        type=str,
        help="Frame directory (default: frames_temp)"
    )
    parser_animate.add_argument(
        "--config",
        type=str,
        help="Config file path (YAML/JSON)"
    )
    parser_animate.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing"
    )
    parser_animate.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers"
    )
    parser_animate.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10)"
    )
    parser_animate.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec (default: libx264)"
    )
    parser_animate.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Quality setting (default: 23)"
    )
    parser_animate.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="Encoding preset (default: medium)"
    )
    parser_animate.add_argument(
        "--cleanup-frames",
        action="store_true",
        help="Delete frames after video creation"
    )
    parser_animate.set_defaults(func=cmd_animate)

    # ========================================================================
    # compare-video subcommand (GFS vs ECMWF side-by-side)
    # ========================================================================
    parser_compare = subparsers.add_parser(
        "compare-video",
        help="Generate side-by-side GFS vs ECMWF comparison videos"
    )
    _add_common_globalish_args(parser_compare)
    parser_compare.add_argument(
        "--cycle",
        type=parse_cycle,
        required=True,
        help=(
            "Requested initialization time (YYYYMMDDHH). For side-by-side runs this is a base cycle that is "
            "aligned per-model unless --gfs-cycle/--ecmwf-cycle are provided."
        )
    )
    parser_compare.add_argument(
        "--gfs-cycle",
        type=parse_cycle,
        help="Override GFS init cycle (YYYYMMDDHH)"
    )
    parser_compare.add_argument(
        "--ecmwf-cycle",
        type=parse_cycle,
        help="Override ECMWF init cycle (YYYYMMDDHH)"
    )
    parser_compare.add_argument(
        "--align-gfs",
        choices=["strict", "previous", "next", "nearest"],
        default="strict",
        help="How to align base --cycle to a valid GFS cycle hour"
    )
    parser_compare.add_argument(
        "--align-ecmwf",
        choices=["strict", "previous", "next", "nearest"],
        default="previous",
        help="How to align base --cycle to a valid ECMWF cycle hour"
    )
    parser_compare.add_argument(
        "--region",
        type=str,
        default="CONUS",
        help=(
            "Region preset name (default: CONUS). See available presets in synoptic_charts/constants.py. "
            "To customize extents, edit/add a REGIONS entry (extent=[W,E,S,N])."
        )
    )
    parser_compare.add_argument(
        "--config",
        type=str,
        help="Config file path (YAML/JSON)"
    )
    parser_compare.add_argument(
        "--modes",
        choices=["rate", "accumulated", "both"],
        default="both",
        help="Which precipitation visualization(s) to render"
    )
    parser_compare.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="First lead time (hours)"
    )
    parser_compare.add_argument(
        "--end-hour",
        type=int,
        default=None,
        help="Last lead time (hours). If omitted, uses maximum common lead time."
    )
    parser_compare.add_argument(
        "--interval",
        type=int,
        default=6,
        help="Lead time spacing (hours)"
    )
    parser_compare.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes"
    )
    parser_compare.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing for frame generation"
    )
    parser_compare.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Video frames per second"
    )
    parser_compare.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec (default: libx264)"
    )
    parser_compare.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Quality setting, 0-51, lower=better (default: 23)"
    )
    parser_compare.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="Encoding preset (default: medium)"
    )
    parser_compare.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Root output directory for frames/videos"
    )
    parser_compare.add_argument(
        "--gap-px",
        type=int,
        default=8,
        help="Pixel gap between the two panels"
    )
    parser_compare.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing frame directories before regenerating"
    )
    parser_compare.set_defaults(func=cmd_compare_video)

    # ========================================================================
    # forecast-video subcommand (single model OR compare)
    # ========================================================================
    parser_forecast = subparsers.add_parser(
        "forecast-video",
        help="Generate forecast MP4 for a model or side-by-side comparison"
    )
    _add_common_globalish_args(parser_forecast)
    parser_forecast.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (GFS, ECMWF) or 'compare' for side-by-side"
    )
    parser_forecast.add_argument(
        "--cycle",
        type=parse_cycle,
        required=True,
        help=(
            "Requested initialization time (YYYYMMDDHH). If the model doesn't run that hour, the cycle is "
            "aligned using --align (e.g., ECMWF typically supports 00Z/12Z)."
        )
    )
    parser_forecast.add_argument(
        "--max-lead",
        type=int,
        default=None,
        help=(
            "Maximum forecast lead time (hours). If omitted, uses the model's max (or max common for compare). "
            "For single-model runs, values beyond the model's max are capped."
        )
    )
    parser_forecast.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="First lead time (hours)"
    )
    parser_forecast.add_argument(
        "--interval",
        type=int,
        default=6,
        help="Lead time spacing (hours)"
    )
    parser_forecast.add_argument(
        "--region",
        type=str,
        default="CONUS",
        help=(
            "Region preset name (default: CONUS). See available presets in synoptic_charts/constants.py. "
            "To customize extents, edit/add a REGIONS entry (extent=[W,E,S,N])."
        )
    )
    parser_forecast.add_argument(
        "--config",
        type=str,
        help="Config file path (YAML/JSON)"
    )
    parser_forecast.add_argument(
        "--modes",
        choices=["rate", "accumulated", "both"],
        default="both",
        help="Which precipitation visualization(s) to render"
    )
    parser_forecast.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes"
    )
    parser_forecast.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing for frame generation"
    )
    parser_forecast.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Video frames per second"
    )
    parser_forecast.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec (default: libx264)"
    )
    parser_forecast.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Quality setting, 0-51, lower=better (default: 23)"
    )
    parser_forecast.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="Encoding preset (default: medium)"
    )
    parser_forecast.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Root output directory for frames/videos"
    )
    parser_forecast.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output frame directories before regenerating (compare only)"
    )
    parser_forecast.add_argument(
        "--gap-px",
        type=int,
        default=8,
        help="Pixel gap between the two panels (compare only)"
    )
    # Alignment controls
    parser_forecast.add_argument(
        "--align",
        choices=["strict", "previous", "next", "nearest"],
        default="strict",
        help="How to align --cycle to a valid cycle hour for single-model runs"
    )
    parser_forecast.add_argument(
        "--gfs-cycle",
        type=parse_cycle,
        help="Override GFS init cycle (compare only)"
    )
    parser_forecast.add_argument(
        "--ecmwf-cycle",
        type=parse_cycle,
        help="Override ECMWF init cycle (compare only)"
    )
    parser_forecast.add_argument(
        "--align-gfs",
        choices=["strict", "previous", "next", "nearest"],
        default="strict",
        help="How to align base --cycle to a valid GFS cycle hour (compare only)"
    )
    parser_forecast.add_argument(
        "--align-ecmwf",
        choices=["strict", "previous", "next", "nearest"],
        default="previous",
        help="How to align base --cycle to a valid ECMWF cycle hour (compare only)"
    )
    parser_forecast.set_defaults(func=cmd_forecast_video)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if subcommand was provided
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging_from_args(args)
    
    # Execute subcommand
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
