"""
Meteorological calculations and feature detection for SynopticCharts.

This module provides functions for processing meteorological data including:
- MSLP smoothing and surface feature detection (highs/lows)
- Precipitation rate unit conversions and trace masking
- Atmospheric thickness calculations
- Coordinate transformations and distance calculations

The calculations module bridges the data retrieval layer (downloader) and
the visualization layer, performing necessary transformations and analyses
on xarray DataArrays while preserving coordinate systems and metadata.

Main Functions:
    From meteo module:
        - smooth_mslp: Apply Gaussian smoothing to MSLP data
        - convert_precip_rate_to_inches_per_hour: Convert precip units
        - calculate_thickness: Compute 1000-500mb thickness
        - mask_trace_precipitation: Apply trace threshold masking
    
    From features module:
        - detect_surface_features: Find pressure highs and lows
        - filter_nearby_features: Remove duplicate nearby features

Example:
    >>> from synoptic_charts.calculations import smooth_mslp, detect_surface_features
    >>> from synoptic_charts.data import ModelDownloader
    >>> from datetime import datetime
    >>> 
    >>> # Download data
    >>> downloader = ModelDownloader("GFS", datetime(2024, 1, 15, 0), Config(), 24)
    >>> data = downloader.fetch_all_data()
    >>> 
    >>> # Process MSLP and detect features
    >>> smoothed = smooth_mslp(data['mslp']['PRMSL'])
    >>> features = detect_surface_features(smoothed)
    >>> print(f"Found {len(features['highs'])} highs and {len(features['lows'])} lows")
"""

from .meteo import (
    smooth_mslp,
    convert_precip_rate_to_inches_per_hour,
    calculate_thickness,
    mask_trace_precipitation
)
from .features import (
    detect_surface_features,
    filter_nearby_features
)

__all__ = [
    "smooth_mslp",
    "convert_precip_rate_to_inches_per_hour",
    "calculate_thickness",
    "mask_trace_precipitation",
    "detect_surface_features",
    "filter_nearby_features",
]
