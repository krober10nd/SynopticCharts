"""
Data retrieval and management for SynopticCharts.

This module provides functionality for downloading and managing meteorological
model data from various sources (GFS, ECMWF) using the Herbie library.

Main Classes:
    ModelDownloader: Downloads and manages model forecast data

Example:
    >>> from synoptic_charts.data import ModelDownloader
    >>> from synoptic_charts import Config
    >>> from datetime import datetime
    >>> 
    >>> downloader = ModelDownloader(
    ...     model="GFS",
    ...     forecast_cycle=datetime(2024, 1, 15, 0),
    ...     config=Config(),
    ...     lead_time=24
    ... )
    >>> data = downloader.fetch_all_data()
"""

from .downloader import ModelDownloader

__all__ = ["ModelDownloader"]
