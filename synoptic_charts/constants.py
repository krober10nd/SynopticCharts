"""
Constants and fixed parameters for SynopticCharts package.

This module defines model configurations, regional parameters, precipitation types,
styling constants, and physical constants used throughout the package.
"""

import numpy as np

# ============================================================================
# Model Definitions
# ============================================================================

MODELS = {
    "GFS": {
        "name": "Global Forecast System",
        "herbie_name": "gfs",
        "resolution": "0.25°",
        "forecast_hours": list(range(0, 241, 3)) + list(range(240, 385, 12)),
    },
    "ECMWF": {
        "name": "European Centre for Medium-Range Weather Forecasts",
        "herbie_name": "ecmwf",
        "resolution": "0.1°",
        "forecast_hours": list(range(0, 145, 3)) + list(range(144, 241, 6)),
    },
}

# ============================================================================
# Region Definitions
# ============================================================================

REGIONS = {
    "CONUS": {
        "name": "Continental United States",
        "extent": [-125.0, -66.0, 24.0, 50.0],  # [west, east, south, north]
        "projection_params": {
            "central_longitude": -96.0,
            "central_latitude": 39.0,
            "standard_parallels": (33.0, 45.0),
        },
        "figure_size": (16, 12),  # width, height in inches
    },
    "NORTHEAST": {
        "name": "Northeast US + Mid-Atlantic",
        # Northeast + Mid-Atlantic focus (roughly VA/NC border up through ME).
        "extent": [-82.0, -66.0, 36.0, 47.5],  # [west, east, south, north]
        "projection_params": {
            "central_longitude": -74.5,
            "central_latitude": 41.5,
            "standard_parallels": (38.5, 45.5),
        },
        "figure_size": (16, 12),  # keep consistent with CONUS for now
    },
}

# ============================================================================
# Precipitation Type Definitions
# ============================================================================

PRECIP_TYPES = {
    "rain": {
        "name": "Rain",
        # 20 discrete color increments from 0.00 to 0.50 in/hr
        "colormap": "Greens",
        "levels": np.linspace(0.00, 0.50, 21).tolist(),  # inches/hr
        "herbie_variable": "PRATE",  # Precipitation rate
    },
    "snow": {
        "name": "Snow",
        # 20 discrete color increments from 0.00 to 0.50 in/hr (liquid equiv)
        "colormap": "Blues",
        "levels": np.linspace(0.00, 0.50, 21).tolist(),  # inches/hr liquid equiv
        "herbie_variable": "SNOD",  # Snow depth
    },
    "frzr": {
        "name": "Freezing Rain",
        # 20 discrete color increments from 0.00 to 0.50 in/hr
        "colormap": "Reds",
        "levels": np.linspace(0.00, 0.50, 21).tolist(),  # inches/hr
        "herbie_variable": "FRZR",  # Freezing rain
    },
    "sleet": {
        "name": "Sleet",
        # 20 discrete color increments from 0.00 to 0.50 in/hr
        "colormap": "Purples",
        "levels": np.linspace(0.00, 0.50, 21).tolist(),  # inches/hr
        "herbie_variable": "ICEP",  # Ice pellets
    },
     "unclassified": {
        "name": "Unclassified",
        # Fill precip areas where categorical flags are all 0.
        "colormap": "Greys",
        "levels": np.linspace(0.00, 0.50, 21).tolist(),  # inches/hr
        "herbie_variable": "PRATE",
    },
}

# Tick labels to match the standard categorical precipitation colorbars.
PRECIP_COLORBAR_TICKS = [0.05, 0.10, 0.20, 0.50]

# ============================================================================
# Thickness Visualization
# ============================================================================

THICKNESS_COLORMAP = "RdYlBu_r"
THICKNESS_LEVELS = np.arange(480, 600, 6)  # decameters

# ============================================================================
# Styling Constants
# ============================================================================

# MSLP Contour Styling
MSLP_CONTOUR_COLOR = "black"
MSLP_CONTOUR_LINEWIDTH = 0.5

# Surface Feature Markers
HIGH_MARKER_COLOR = "red"
LOW_MARKER_COLOR = "blue"

# Font Sizes
FEATURE_FONT_SIZE = 14
TITLE_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 10

# Annotation Positioning
# Figure coordinates (0..1). Kept close to the outer margins so the map can
# fill the figure without leaving large whitespace.
TITLE_POSITION = (0.008, 0.992)
MODEL_INFO_POSITION = (0.992, 0.992)

# Colorbar Layout
COLORBAR_BOTTOM = 0.05  # Bottom margin for colorbars
COLORBAR_HEIGHT = 0.02  # Height of colorbar
COLORBAR_SPACING = 0.02  # Horizontal spacing between colorbars
COLORBAR_LABEL_SIZE = 8  # Font size for colorbar labels

# ============================================================================
# Physical Constants
# ============================================================================

EARTH_RADIUS = 6371.0  # kilometers
GRAVITY = 9.80665  # m/s²
