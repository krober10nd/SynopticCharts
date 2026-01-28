"""
Rendering subsystem for SynopticCharts.

This module provides cartographic rendering capabilities for synoptic meteorological
charts. It handles map projection setup, geographic feature rendering, and basemap
creation using Cartopy and Matplotlib.

The rendering module separates cartographic concerns from data visualization, allowing
chart generation to focus on meteorological data while the basemap handles coordinate
systems, projections, and geographic context.

Main Classes:
    BasemapRenderer: Primary class for creating map projections with geographic features

Key Features:
    - Lambert Conformal projection configured from region parameters
    - Automatic feature resolution selection based on zoom level
    - State boundaries, country borders, and coastlines
    - Configurable grid lines with coordinate labels
    - Zoom functionality for regional focus
    - Integration with xarray data from downloader module

Coordinate System:
    - Uses Cartopy's Lambert Conformal projection for CONUS
    - Data layers should use transform=cartopy.crs.PlateCarree() when plotting
    - Standardized lat/lon coordinates from downloader are compatible

Example:
    >>> from synoptic_charts.rendering import BasemapRenderer
    >>> from synoptic_charts import Config
    >>> 
    >>> # Create basemap with default CONUS region
    >>> renderer = BasemapRenderer(region="CONUS", config=Config())
    >>> fig, ax = renderer.setup_complete_basemap()
    >>> 
    >>> # Now plot meteorological data on the axes
    >>> # ax.contour(lons, lats, data, transform=ccrs.PlateCarree())
    >>> 
    >>> # Save or display
    >>> fig.savefig("chart.png")
"""

from .basemap import BasemapRenderer
from .layers import (
    render_mslp_contours,
    render_precipitation,
    render_thickness,
    render_surface_features,
    create_colorbar_from_paint
)
from .annotations import (
    add_title_annotation,
    add_model_info_annotation,
    add_precipitation_colorbars,
    annotate_chart
)
from .chart import SynopticChart

__all__ = [
    "BasemapRenderer",
    "render_mslp_contours",
    "render_precipitation",
    "render_thickness",
    "render_surface_features",
    "create_colorbar_from_paint",
    "add_title_annotation",
    "add_model_info_annotation",
    "add_precipitation_colorbars",
    "annotate_chart",
    "SynopticChart"
]
