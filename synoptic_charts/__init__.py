"""
SynopticCharts - Lightweight Python package for creating synoptic meteorological charts.

This package provides tools for downloading model data (GFS, ECMWF) and creating
professional synoptic-style charts with MSLP contours, precipitation types,
thickness analysis, and surface feature detection.

Quick Start:
    >>> from synoptic_charts import create_chart
    >>> from datetime import datetime
    >>> 
    >>> # Create a single chart
    >>> create_chart(
    ...     model="GFS",
    ...     forecast_cycle=datetime(2024, 1, 15, 0),
    ...     lead_time=24,
    ...     output_path="chart_f024.png"
    ... )
    
    >>> # Batch processing for video
    >>> from synoptic_charts import BatchChartGenerator, create_video_from_batch
    >>> 
    >>> batch = BatchChartGenerator("GFS", datetime(2024, 1, 15, 0))
    >>> result = batch.generate_forecast_sequence(0, 48, 6)
    >>> create_video_from_batch(result, "forecast.mp4", fps=10)

Advanced Usage:
    >>> # Direct access to components
    >>> from synoptic_charts import Config, REGIONS, MODELS
    >>> from synoptic_charts import ModelDownloader, SynopticChart
    >>> 
    >>> # Custom configuration
    >>> config = Config(default_dpi=300, mslp_contour_interval=2.0)
    >>> 
    >>> # Manual workflow
    >>> downloader = ModelDownloader("GFS", datetime(2024, 1, 15, 0), config, 24)
    >>> data = downloader.fetch_all_data()
    >>> chart = SynopticChart("CONUS", config)
    >>> fig, ax = chart.render_chart(data, "GFS", init_time, valid_time, 24)
"""

__version__ = "0.1.0"
__author__ = "Keith Roberts"
__email__ = "keithrbt0@gmail.com"

# Initialize logging with default settings
from .logging_config import setup_logging
setup_logging()

# Core constants and configuration
from .constants import REGIONS, MODELS
from .config import Config

# Data acquisition
from .data import ModelDownloader

# Rendering components
from .rendering import BasemapRenderer, SynopticChart

# Calculations
from . import calculations

# User-facing API
from .api import create_chart, create_chart_from_data, get_available_lead_times

# Batch processing
from .batch import BatchChartGenerator

# Video generation
from .video import VideoGenerator, create_video_from_batch

# Exceptions
from .exceptions import (
    SynopticChartsError,
    DataFetchError,
    RenderError,
    VideoCreationError,
    InvalidParameterError
)

__all__ = [
    # Version info
    "__version__",
    
    # Constants and config
    "REGIONS",
    "MODELS",
    "Config",
    
    # Core components
    "ModelDownloader",
    "BasemapRenderer",
    "SynopticChart",
    "calculations",
    
    # User-facing API
    "create_chart",
    "create_chart_from_data",
    "get_available_lead_times",
    
    # Batch and video
    "BatchChartGenerator",
    "VideoGenerator",
    "create_video_from_batch",
    
    # Exceptions
    "SynopticChartsError",
    "DataFetchError",
    "RenderError",
    "VideoCreationError",
    "InvalidParameterError",
]
