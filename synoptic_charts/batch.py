"""
Batch processing module for generating multiple synoptic chart frames.

This module provides the BatchChartGenerator class for efficiently creating
sequences of charts across multiple forecast lead times, with support for
parallel processing. Designed primarily for video generation workflows.

Example:
    >>> from synoptic_charts import BatchChartGenerator
    >>> from datetime import datetime
    >>> 
    >>> # Create batch generator
    >>> batch = BatchChartGenerator(
    ...     model="GFS",
    ...     forecast_cycle=datetime(2024, 1, 15, 0),
    ...     region="CONUS"
    ... )
    >>> 
    >>> # Generate forecast sequence
    >>> result = batch.generate_forecast_sequence(
    ...     start_hour=0,
    ...     end_hour=120,
    ...     interval=3,
    ...     parallel=True
    ... )
    >>> print(f"Generated {len(result['successful_frames'])} frames")
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import xarray as xr

from .api import create_chart, create_chart_from_data, get_available_lead_times
from .config import Config
from .data import ModelDownloader
from .exceptions import InvalidParameterError

logger = logging.getLogger(__name__)


def _generate_frame_worker(
    *,
    model: str,
    forecast_cycle: datetime,
    region: str,
    config: Config,
    output_dir: str,
    lead_time: int,
) -> Optional[str]:
    """Process-safe worker for frame rendering."""

    # IMPORTANT: In process-based parallel runs, force a non-interactive backend.
    # This avoids macOS GUI backend issues and makes rendering deterministic.
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    frame_filename = f"frame_{lead_time:03d}.png"
    frame_path = Path(output_dir) / frame_filename
    try:
        create_chart(
            model=model,
            forecast_cycle=forecast_cycle,
            lead_time=lead_time,
            region=region,
            output_path=frame_path,
            config=config,
        )
        return str(frame_path)
    except Exception as e:
        logger.error(f"Failed to generate frame F{lead_time:03d}: {e}")
        return None

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.debug("tqdm not available, progress bars disabled")


class BatchChartGenerator:
    """
    Generate multiple synoptic chart frames for video creation.
    
    This class manages batch generation of charts across multiple forecast
    lead times, with support for parallel processing to speed up generation.
    
    Attributes:
        model: Model name (e.g., "GFS", "ECMWF")
        forecast_cycle: Model initialization datetime
        region: Region name from REGIONS dictionary
        config: Configuration object
        output_dir: Directory for generated frames
        
    Example:
        >>> batch = BatchChartGenerator(
        ...     model="GFS",
        ...     forecast_cycle=datetime(2024, 1, 15, 0),
        ...     region="CONUS"
        ... )
        >>> result = batch.generate_frames([0, 3, 6, 9, 12])
    """
    
    def __init__(
        self,
        model: str,
        forecast_cycle: datetime,
        region: str = "CONUS",
        config: Optional[Config] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize batch chart generator.
        
        Args:
            model: Model name (e.g., "GFS", "ECMWF")
            forecast_cycle: Model initialization datetime
            region: Region name (default: "CONUS")
            config: Optional Config object
            output_dir: Optional output directory (defaults to ./frames)
        """
        self.model = model
        self.forecast_cycle = forecast_cycle
        self.region = region
        self.config = config if config is not None else Config()
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path("frames")
        else:
            self.output_dir = Path(output_dir)
        
        logger.info(
            f"Initialized BatchChartGenerator: {model} "
            f"init={forecast_cycle.strftime('%Y%m%d %HZ')} region={region}"
        )
    
    def generate_frames(
        self,
        lead_times: List[int],
        parallel: bool = False,
        max_workers: Optional[int] = None,
        parallel_backend: str = "thread",
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate chart frames for specified lead times.
        
        Args:
            lead_times: List of forecast hours to generate
            parallel: Enable parallel processing (default: False)
            max_workers: Maximum parallel workers (default: CPU count)
            
        Returns:
            Dictionary with keys:
                - successful_frames: List of paths to generated frames
                - failed_frames: List of lead times that failed
                - total_time: Total generation time in seconds
                
        Example:
            >>> result = batch.generate_frames([0, 6, 12, 18, 24])
            >>> print(f"Success rate: {len(result['successful_frames'])/len(lead_times)*100}%")
        """
        logger.info(
            f"Generating {len(lead_times)} frames "
            f"(parallel={parallel}, max_workers={max_workers})"
        )

        precip_mode = str(getattr(self.config, "precip_mode", "rate") or "rate").strip().lower()
        if precip_mode == "accumulated":
            if parallel:
                logger.info(
                    "precip_mode='accumulated' requires cumulative init->lead sums; "
                    "running sequentially to keep accumulation consistent"
                )
            return self._generate_frames_accumulated_cumulative(
                lead_times=lead_times,
                show_progress=show_progress,
            )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        successful_frames = []
        failed_frames = []
        
        if parallel:
            # Parallel processing
            backend = (parallel_backend or "thread").strip().lower()
            if backend not in {"thread", "process"}:
                raise InvalidParameterError(
                    f"Invalid parallel_backend='{parallel_backend}'. Use 'thread' or 'process'."
                )

            logger.info(f"Using parallel processing (backend={backend})")

            Executor = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
            with Executor(max_workers=max_workers) as executor:
                # Submit all tasks
                if backend == "thread":
                    future_to_lead_time = {
                        executor.submit(self._generate_single_frame, lead_time): lead_time
                        for lead_time in lead_times
                    }
                else:
                    # Processes require a top-level worker function.
                    future_to_lead_time = {
                        executor.submit(
                            _generate_frame_worker,
                            model=self.model,
                            forecast_cycle=self.forecast_cycle,
                            region=self.region,
                            config=self.config,
                            output_dir=str(self.output_dir),
                            lead_time=lead_time,
                        ): lead_time
                        for lead_time in lead_times
                    }
                
                # Process results with optional progress bar
                if TQDM_AVAILABLE:
                    futures = tqdm(
                        as_completed(future_to_lead_time),
                        total=len(lead_times),
                        desc="Generating frames",
                        unit="frame",
                        disable=not show_progress,
                    )
                else:
                    futures = as_completed(future_to_lead_time)
                
                for future in futures:
                    lead_time = future_to_lead_time[future]
                    try:
                        frame_path = future.result()
                        if frame_path:
                            successful_frames.append(frame_path)
                            logger.debug(f"Frame F{lead_time:03d} completed: {frame_path}")
                        else:
                            failed_frames.append(lead_time)
                            logger.warning(f"Frame F{lead_time:03d} failed")
                    except Exception as e:
                        failed_frames.append(lead_time)
                        logger.error(f"Frame F{lead_time:03d} failed with error: {e}")
        
        else:
            # Sequential processing
            logger.info("Using sequential processing")
            
            # Optional progress bar
            if TQDM_AVAILABLE:
                lead_times_iter = tqdm(
                    lead_times,
                    desc="Generating frames",
                    unit="frame",
                    disable=not show_progress,
                )
            else:
                lead_times_iter = lead_times
            
            for lead_time in lead_times_iter:
                try:
                    frame_path = self._generate_single_frame(lead_time)
                    if frame_path:
                        successful_frames.append(frame_path)
                        logger.debug(f"Frame F{lead_time:03d} completed: {frame_path}")
                    else:
                        failed_frames.append(lead_time)
                        logger.warning(f"Frame F{lead_time:03d} failed")
                except Exception as e:
                    failed_frames.append(lead_time)
                    logger.error(f"Frame F{lead_time:03d} failed with error: {e}")
        
        # Calculate statistics
        total_time = time.time() - start_time
        success_count = len(successful_frames)
        total_count = len(lead_times)
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        avg_time = total_time / total_count if total_count > 0 else 0
        
        logger.info(
            f"Batch generation complete: {success_count}/{total_count} frames "
            f"({success_rate:.1f}%) in {total_time:.1f}s "
            f"(avg {avg_time:.1f}s/frame)"
        )
        
        if failed_frames:
            logger.warning(f"Failed frames: {failed_frames}")
        
        return {
            "successful_frames": sorted(successful_frames),
            "failed_frames": failed_frames,
            "total_time": total_time
        }

    def _generate_frames_accumulated_cumulative(
        self,
        *,
        lead_times: List[int],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Generate accumulated-precipitation frames with init->lead cumulative totals."""

        logger.info(f"Generating {len(lead_times)} accumulated-mode frames (cumulative)")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        successful_frames: List[str] = []
        failed_frames: List[int] = []

        ordered = sorted(set(int(x) for x in lead_times))

        if TQDM_AVAILABLE:
            lead_times_iter = tqdm(
                ordered,
                desc="Generating frames",
                unit="frame",
                disable=not show_progress,
            )
        else:
            lead_times_iter = ordered

        cumulative = None

        for lead_time in lead_times_iter:
            frame_filename = f"frame_{lead_time:03d}.png"
            frame_path = self.output_dir / frame_filename

            try:
                downloader = ModelDownloader(
                    model_name=self.model,
                    forecast_cycle=self.forecast_cycle,
                    config=self.config,
                    lead_time=int(lead_time),
                )

                # Fetch core fields (MSLP + heights) without precip.
                data_dict = downloader.fetch_all_data(use_cache=True, include_precip=False)
                if not data_dict:
                    raise RuntimeError("No core data returned")

                # Incremental precip (mm) and cumulative sum.
                step_ds = downloader.fetch_precipitation_accumulation_increment(use_cache=True)
                step_da = None
                if step_ds is not None and hasattr(step_ds, "data_vars"):
                    if "precip_step" in step_ds.data_vars:
                        step_da = step_ds["precip_step"]
                    else:
                        dvs = list(step_ds.data_vars)
                        step_da = step_ds[dvs[0]] if dvs else None

                if step_da is not None:
                    if cumulative is None:
                        cumulative = step_da
                    else:
                        try:
                            a, b = xr.align(cumulative, step_da, join="inner")
                            cumulative = a + b
                        except Exception:
                            cumulative = cumulative + step_da

                data_dict["precip_accum"] = (
                    xr.Dataset({"precip_accum": cumulative}) if cumulative is not None else None
                )

                valid_time = self.forecast_cycle + timedelta(hours=int(lead_time))
                create_chart_from_data(
                    data_dict=data_dict,
                    model_name=self.model,
                    init_time=self.forecast_cycle,
                    valid_time=valid_time,
                    lead_time=int(lead_time),
                    region=self.region,
                    output_path=frame_path,
                    config=self.config,
                )
                successful_frames.append(str(frame_path))
            except Exception as e:
                failed_frames.append(int(lead_time))
                logger.error(f"Failed to generate frame F{lead_time:03d}: {e}")

        total_time = time.time() - start_time
        return {
            "successful_frames": sorted(successful_frames),
            "failed_frames": failed_frames,
            "total_time": total_time,
        }
    
    def generate_forecast_sequence(
        self,
        start_hour: int = 0,
        end_hour: int = 120,
        interval: int = 3,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        parallel_backend: str = "thread",
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a sequence of frames across a forecast range.
        
        Args:
            start_hour: First forecast hour (default: 0)
            end_hour: Last forecast hour (default: 120)
            interval: Hour interval between frames (default: 3)
            parallel: Enable parallel processing (default: False)
            max_workers: Maximum parallel workers (default: CPU count)
            
        Returns:
            Dictionary with keys:
                - successful_frames: List of paths to generated frames
                - failed_frames: List of lead times that failed
                - total_time: Total generation time in seconds
                
        Raises:
            InvalidParameterError: If lead times are invalid for the model
            
        Example:
            >>> result = batch.generate_forecast_sequence(
            ...     start_hour=0,
            ...     end_hour=48,
            ...     interval=6,
            ...     parallel=True
            ... )
        """
        logger.info(
            f"Generating forecast sequence: {start_hour}h to {end_hour}h "
            f"every {interval}h"
        )
        
        # Generate lead times
        lead_times = list(range(start_hour, end_hour + 1, interval))
        logger.debug(f"Lead times: {lead_times}")
        
        # Validate lead times against model
        available_lead_times = get_available_lead_times(self.model)
        invalid_lead_times = [lt for lt in lead_times if lt not in available_lead_times]
        
        if invalid_lead_times:
            raise InvalidParameterError(
                f"Invalid lead times for {self.model}: {invalid_lead_times}. "
                f"Valid range: {min(available_lead_times)}-{max(available_lead_times)}"
            )
        
        # Generate frames
        return self.generate_frames(
            lead_times=lead_times,
            parallel=parallel,
            max_workers=max_workers,
            parallel_backend=parallel_backend,
            show_progress=show_progress,
        )
    
    def cleanup_frames(self, keep_latest: Optional[int] = None) -> int:
        """
        Remove generated frames from output directory.
        
        Args:
            keep_latest: If specified, keep N most recent frames
            
        Returns:
            Number of files deleted
            
        Example:
            >>> deleted = batch.cleanup_frames()
            >>> print(f"Deleted {deleted} frames")
            >>> 
            >>> # Keep 10 most recent frames
            >>> deleted = batch.cleanup_frames(keep_latest=10)
        """
        logger.info(f"Cleaning up frames from {self.output_dir}")
        
        if not self.output_dir.exists():
            logger.warning(f"Output directory does not exist: {self.output_dir}")
            return 0
        
        # Find all frame files
        frame_files = sorted(self.output_dir.glob("frame_*.png"))
        
        if not frame_files:
            logger.info("No frames found to clean up")
            return 0
        
        # Determine files to delete
        if keep_latest is not None and keep_latest > 0:
            # Keep N most recent, delete the rest
            files_to_delete = frame_files[:-keep_latest] if len(frame_files) > keep_latest else []
            logger.info(f"Keeping {min(keep_latest, len(frame_files))} most recent frames")
        else:
            # Delete all
            files_to_delete = frame_files
        
        # Delete files
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug(f"Deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleanup complete: deleted {deleted_count} frames")
        
        return deleted_count
    
    def _generate_single_frame(self, lead_time: int) -> Optional[str]:
        """
        Generate a single frame (internal method).
        
        Args:
            lead_time: Forecast hour
            
        Returns:
            Path to generated frame, or None if failed
        """
        frame_filename = f"frame_{lead_time:03d}.png"
        frame_path = self.output_dir / frame_filename
        
        try:
            create_chart(
                model=self.model,
                forecast_cycle=self.forecast_cycle,
                lead_time=lead_time,
                region=self.region,
                output_path=frame_path,
                config=self.config
            )
            return str(frame_path)
        except Exception as e:
            logger.error(f"Failed to generate frame F{lead_time:03d}: {e}")
            return None
