"""
Basemap rendering for synoptic charts using Cartopy and Matplotlib.

This module provides the BasemapRenderer class for creating cartographic basemaps
with Lambert Conformal projection, geographic features, and configurable zoom levels.
"""

import logging
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from ..config import Config
from ..constants import REGIONS, COLORBAR_BOTTOM, COLORBAR_HEIGHT, TITLE_POSITION

logger = logging.getLogger("synoptic_charts.rendering")

# Feature styling constants
STATE_LINEWIDTH = 0.5
STATE_COLOR = '#666666'
COUNTRY_LINEWIDTH = 0.8
COUNTRY_COLOR = 'black'
COASTLINE_LINEWIDTH = 0.6
COASTLINE_COLOR = 'black'

# Grid styling constants
GRID_LINEWIDTH = 0.3
GRID_COLOR = '#CCCCCC'
GRID_ALPHA = 0.5
GRID_LINESTYLE = '--'
GRID_LABEL_SIZE = 9


def _is_dark_color(color: Optional[str]) -> bool:
    """Return True when a Matplotlib color spec is visually dark."""

    if not color:
        return False
    try:
        r, g, b = mcolors.to_rgb(color)
    except Exception:
        return False

    # Relative luminance (sRGB-ish). Good enough for theme heuristics.
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance < 0.35

# Zoom level defaults
DEFAULT_ZOOM_LEVEL = 1.0
MIN_ZOOM_LEVEL = 0.1
MAX_ZOOM_LEVEL = 10.0

# Feature resolution mapping
FEATURE_RESOLUTION_MAP = {
    'coarse': '110m',      # zoom < 0.5
    'medium': '50m',       # 0.5 <= zoom <= 2.0
    'fine': '10m'          # zoom > 2.0
}

# Grid spacing mapping (in degrees)
GRID_SPACING_MAP = {
    'coarse': 10.0,        # zoom < 0.5
    'medium': 5.0,         # 0.5 <= zoom <= 1.0
    'fine': 2.0,           # 1.0 < zoom <= 2.0
    'very_fine': 1.0,      # 2.0 < zoom <= 5.0
    'ultra_fine': 0.5      # zoom > 5.0
}


def validate_region(region: str) -> bool:
    """
    Check if region name exists in REGIONS dictionary with required keys.
    
    Args:
        region: Region name to validate
        
    Returns:
        True if region is valid
        
    Raises:
        ValueError: If region is invalid or missing required keys
    """
    if region not in REGIONS:
        raise ValueError(
            f"Region '{region}' not found. "
            f"Available regions: {list(REGIONS.keys())}"
        )
    
    region_config = REGIONS[region]
    required_keys = ['extent', 'projection_params', 'figure_size']
    
    for key in required_keys:
        if key not in region_config:
            raise ValueError(
                f"Region '{region}' is missing required key '{key}'"
            )
    
    logger.debug(f"Region '{region}' validated successfully")
    return True


def validate_zoom_level(zoom_level: float) -> bool:
    """
    Validate that zoom_level is a positive number.
    
    Args:
        zoom_level: Zoom level to validate
        
    Returns:
        True if zoom_level is valid
        
    Raises:
        ValueError: If zoom_level is not positive
    """
    if zoom_level <= 0:
        raise ValueError(f"Zoom level must be positive, got {zoom_level}")
    
    if zoom_level < MIN_ZOOM_LEVEL or zoom_level > MAX_ZOOM_LEVEL:
        logger.warning(
            f"Extreme zoom level {zoom_level} detected. "
            f"Recommended range: {MIN_ZOOM_LEVEL}-{MAX_ZOOM_LEVEL}"
        )
    
    return True


def get_projection_from_region(region: str) -> ccrs.LambertConformal:
    """
    Create Cartopy Lambert Conformal projection from region parameters.
    
    Args:
        region: Region name from REGIONS dictionary
        
    Returns:
        Configured LambertConformal projection object
        
    Raises:
        ValueError: If region is invalid
        
    Example:
        >>> proj = get_projection_from_region("CONUS")
        >>> print(proj)
    """
    validate_region(region)
    
    proj_params = REGIONS[region]['projection_params']
    
    try:
        projection = ccrs.LambertConformal(
            central_longitude=proj_params['central_longitude'],
            central_latitude=proj_params['central_latitude'],
            standard_parallels=proj_params['standard_parallels']
        )
        logger.debug(f"Created Lambert Conformal projection for {region}")
        return projection
    except Exception as e:
        raise ValueError(f"Failed to create projection for {region}: {e}")


def get_feature_resolution_for_zoom(zoom_level: float) -> str:
    """
    Get appropriate Natural Earth feature resolution based on zoom level.
    
    Args:
        zoom_level: Current zoom level (1.0 = full region)
        
    Returns:
        Resolution string: '110m', '50m', or '10m'
        
    Example:
        >>> resolution = get_feature_resolution_for_zoom(0.3)
        >>> print(resolution)  # '110m' for zoomed out view
    """
    if zoom_level < 0.5:
        resolution = FEATURE_RESOLUTION_MAP['coarse']
    elif zoom_level <= 2.0:
        resolution = FEATURE_RESOLUTION_MAP['medium']
    else:
        resolution = FEATURE_RESOLUTION_MAP['fine']
    
    logger.debug(f"Selected {resolution} resolution for zoom level {zoom_level}")
    return resolution


def get_grid_spacing_for_zoom(zoom_level: float, extent: list) -> float:
    """
    Calculate appropriate grid line spacing based on zoom level.
    
    Args:
        zoom_level: Current zoom level (1.0 = full region)
        extent: Map extent [west, east, south, north]
        
    Returns:
        Grid spacing in degrees
        
    Example:
        >>> spacing = get_grid_spacing_for_zoom(1.5, [-125, -66, 24, 50])
        >>> print(spacing)  # 2.0 degrees
    """
    if zoom_level < 0.5:
        spacing = GRID_SPACING_MAP['coarse']
    elif zoom_level <= 1.0:
        spacing = GRID_SPACING_MAP['medium']
    elif zoom_level <= 2.0:
        spacing = GRID_SPACING_MAP['fine']
    elif zoom_level <= 5.0:
        spacing = GRID_SPACING_MAP['very_fine']
    else:
        spacing = GRID_SPACING_MAP['ultra_fine']
    
    logger.debug(f"Selected {spacing}° grid spacing for zoom level {zoom_level}")
    return spacing


class BasemapRenderer:
    """
    Create and configure cartographic basemaps for synoptic charts.
    
    This class handles map projection setup, geographic feature rendering, and
    coordinate grid configuration. It supports zoom functionality for regional
    focus and automatically adjusts feature resolution based on zoom level.
    
    The basemap uses Lambert Conformal projection configured from region parameters
    in the constants module. Data layers should use transform=ccrs.PlateCarree()
    when plotting to ensure proper coordinate transformation.
    
    Attributes:
        region: Region name from REGIONS dictionary
        config: Configuration object with display settings
        zoom_level: Current zoom level (1.0 = full region, <1.0 = zoomed in, >1.0 = zoomed out)
        
    Example:
        >>> from synoptic_charts.rendering import BasemapRenderer
        >>> from synoptic_charts import Config
        >>> 
        >>> # Basic usage
        >>> renderer = BasemapRenderer(region="CONUS", config=Config())
        >>> fig, ax = renderer.setup_complete_basemap()
        >>> 
        >>> # With zoom
        >>> renderer = BasemapRenderer(region="CONUS", zoom_level=0.5)
        >>> fig, ax = renderer.setup_complete_basemap()  # 2x zoomed in
        >>> 
        >>> # Adjust zoom after creation
        >>> renderer.set_zoom_level(0.3)
        >>> fig, ax = renderer.setup_complete_basemap()  # Further zoomed in
    """
    
    def __init__(
        self,
        region: str = "CONUS",
        config: Optional[Config] = None,
        zoom_level: float = DEFAULT_ZOOM_LEVEL
    ):
        """
        Initialize BasemapRenderer.
        
        Args:
            region: Region name from REGIONS dictionary (default: "CONUS")
            config: Configuration object (default: creates new Config instance)
            zoom_level: Zoom factor where 1.0 = full region, <1.0 = zoomed in,
                       >1.0 = zoomed out (default: 1.0)
                       
        Raises:
            ValueError: If region is invalid or zoom_level is not positive
        """
        # Validate and store region
        validate_region(region)
        self.region = region
        self._region_config = REGIONS[region]
        
        # Store or create config
        self.config = config if config is not None else Config()
        
        # Validate and store zoom level
        validate_zoom_level(zoom_level)
        self.zoom_level = zoom_level
        
        # Store base extent
        self._base_extent = self._region_config['extent']
        
        # Calculate zoomed extent
        self._extent = self._calculate_zoomed_extent(
            self._base_extent,
            self.zoom_level
        )
        
        logger.info(
            f"Initialized BasemapRenderer: region={region}, "
            f"zoom={zoom_level}, extent={self._extent}"
        )
    
    def _calculate_zoomed_extent(
        self,
        base_extent: list,
        zoom_level: float
    ) -> list:
        """
        Calculate extent adjusted for zoom level.
        
        Zoom level interpretation:
        - 1.0 = full region (no zoom)
        - 0.5 = 2x zoom in (extent reduced by half)
        - 2.0 = 0.5x zoom out (extent expanded by factor of 2)
        
        Args:
            base_extent: Base extent [west, east, south, north]
            zoom_level: Zoom factor
            
        Returns:
            Adjusted extent [west, east, south, north]
            
        Example:
            >>> extent = self._calculate_zoomed_extent([-125, -66, 24, 50], 0.5)
            >>> # Returns extent zoomed in 2x around region center
        """
        west, east, south, north = base_extent
        
        # Calculate center point
        center_lon = (west + east) / 2
        center_lat = (south + north) / 2
        
        # Calculate half-widths
        half_width_lon = (east - west) / 2
        half_height_lat = (north - south) / 2
        
        # Scale by zoom level
        scaled_half_width = half_width_lon * zoom_level
        scaled_half_height = half_height_lat * zoom_level
        
        # Calculate new extent
        new_west = center_lon - scaled_half_width
        new_east = center_lon + scaled_half_width
        new_south = center_lat - scaled_half_height
        new_north = center_lat + scaled_half_height
        
        # Validate bounds
        new_west = max(-180, new_west)
        new_east = min(180, new_east)
        new_south = max(-90, new_south)
        new_north = min(90, new_north)
        
        zoomed_extent = [new_west, new_east, new_south, new_north]
        
        logger.debug(
            f"Calculated zoomed extent: {zoomed_extent} "
            f"(zoom={zoom_level}, center=[{center_lon:.1f}, {center_lat:.1f}])"
        )
        
        return zoomed_extent
    
    def set_zoom_level(self, zoom_level: float) -> 'BasemapRenderer':
        """
        Update zoom level and recalculate extent.
        
        Args:
            zoom_level: New zoom factor where 1.0 = full region, <1.0 = zoomed in,
                       >1.0 = zoomed out
                       
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If zoom_level is not positive
            
        Example:
            >>> renderer = BasemapRenderer("CONUS")
            >>> renderer.set_zoom_level(0.5).setup_complete_basemap()
        """
        validate_zoom_level(zoom_level)
        
        self.zoom_level = zoom_level
        self._extent = self._calculate_zoomed_extent(
            self._base_extent,
            self.zoom_level
        )
        
        logger.info(f"Updated zoom level to {zoom_level}, new extent: {self._extent}")
        
        return self
    
    def create_basemap(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create matplotlib figure and axes with Lambert Conformal projection.
        
        Returns:
            Tuple of (figure, axes) ready for data plotting
            
        Example:
            >>> renderer = BasemapRenderer("CONUS")
            >>> fig, ax = renderer.create_basemap()
            >>> # Now add geographic features and data layers
        """
        logger.info(f"Creating basemap for {self.region}")
        
        # Get figure size - prefer config values, fall back to region config
        if hasattr(self.config, 'figure_width') and hasattr(self.config, 'figure_height'):
            fig_width = self.config.figure_width
            fig_height = self.config.figure_height
            logger.debug(f"Using figure size from Config: ({fig_width}x{fig_height})")
        else:
            fig_width, fig_height = self._region_config['figure_size']
            logger.debug(f"Using figure size from region config: ({fig_width}x{fig_height})")
        
        # Create figure
        fig = plt.figure(
            figsize=(fig_width, fig_height),
            dpi=self.config.default_dpi
        )

        # Use a dark background to help precip colors pop.
        try:
            bg = getattr(self.config, "background_color", "#1f2328")
            fig.patch.set_facecolor(bg)
        except Exception:
            pass
        
        logger.debug(
            f"Created figure: size=({fig_width}x{fig_height}), "
            f"dpi={self.config.default_dpi}"
        )
        
        # Get projection
        projection = get_projection_from_region(self.region)
        
        # Create axes with projection
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        try:
            bg = getattr(self.config, "background_color", "#1f2328")
            ax.set_facecolor(bg)
        except Exception:
            pass

        # Reduce default matplotlib padding so the map fills the figure while
        # still leaving space for figure-level annotations and bottom colorbars.
        try:
            bottom = min(max(COLORBAR_BOTTOM + COLORBAR_HEIGHT + 0.002, 0.0), 0.22)
            top = min(max(TITLE_POSITION[1] - 0.006, 0.78), 0.99)
            fig.subplots_adjust(left=0.006, right=0.994, bottom=bottom, top=top)
        except Exception:
            pass
        
        # Set extent
        ax.set_extent(self._extent, crs=ccrs.PlateCarree())
        
        logger.debug(f"Set map extent: {self._extent}")
        
        return fig, ax
    
    def add_geographic_features(self, ax: plt.Axes) -> None:
        """
        Add state boundaries, country borders, and coastlines to axes.
        
        Feature resolution is automatically selected based on zoom level:
        - '110m' for zoom < 0.5 (coarse, zoomed out)
        - '50m' for 0.5 <= zoom <= 2.0 (medium, standard view)
        - '10m' for zoom > 2.0 (fine, zoomed in)
        
        Args:
            ax: Matplotlib axes with Cartopy projection
            
        Example:
            >>> renderer = BasemapRenderer("CONUS")
            >>> fig, ax = renderer.create_basemap()
            >>> renderer.add_geographic_features(ax)
        """
        logger.info("Adding geographic features")

        bg = getattr(self.config, "background_color", None)
        use_dark_theme = _is_dark_color(bg)
        state_color = "#9aa4b2" if use_dark_theme else STATE_COLOR
        country_color = "#e6edf3" if use_dark_theme else COUNTRY_COLOR
        coastline_color = "#e6edf3" if use_dark_theme else COASTLINE_COLOR
        
        # Get appropriate resolution for zoom level
        resolution = get_feature_resolution_for_zoom(self.zoom_level)
        
        # Add state boundaries
        logger.debug("Adding state boundaries")
        states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale=resolution,
            facecolor='none'
        )
        ax.add_feature(
            states,
            linewidth=STATE_LINEWIDTH,
            edgecolor=state_color,
            zorder=20
        )
        
        # Add country borders
        logger.debug("Adding country borders")
        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale=resolution,
            facecolor='none'
        )
        ax.add_feature(
            countries,
            linewidth=COUNTRY_LINEWIDTH,
            edgecolor=country_color,
            zorder=21
        )
        
        # Add coastlines
        logger.debug("Adding coastlines")
        coastline = cfeature.NaturalEarthFeature(
            category='physical',
            name='coastline',
            scale=resolution,
            facecolor='none'
        )
        ax.add_feature(
            coastline,
            linewidth=COASTLINE_LINEWIDTH,
            edgecolor=coastline_color,
            zorder=22
        )
        
        logger.info(f"Geographic features added with {resolution} resolution")
    
    def add_gridlines(self, ax: plt.Axes) -> None:
        """
        Add latitude/longitude grid lines (labels optional).
        
        Grid spacing is automatically selected based on zoom level to ensure
        readability. Labels are placed on left and bottom edges only.
        
        Args:
            ax: Matplotlib axes with Cartopy projection
            
        Example:
            >>> renderer = BasemapRenderer("CONUS")
            >>> fig, ax = renderer.create_basemap()
            >>> renderer.add_gridlines(ax)
        """
        logger.info("Adding grid lines")

        bg = getattr(self.config, "background_color", None)
        use_dark_theme = _is_dark_color(bg)
        grid_color = "#7d8590" if use_dark_theme else GRID_COLOR
        grid_alpha = 0.60 if use_dark_theme else GRID_ALPHA
        
        # Get appropriate grid spacing for zoom level
        grid_spacing = get_grid_spacing_for_zoom(self.zoom_level, self._extent)
        
        # Add gridlines
        gl = ax.gridlines(
            draw_labels=bool(getattr(self.config, "show_grid_labels", False)),
            linewidth=GRID_LINEWIDTH,
            color=grid_color,
            alpha=grid_alpha,
            linestyle=GRID_LINESTYLE
        )

        if bool(getattr(self.config, "show_grid_labels", False)):
            # Configure labels (left and bottom only)
            gl.top_labels = False
            gl.right_labels = False
            label_color = "#e6edf3" if use_dark_theme else "black"
            gl.xlabel_style = {'size': GRID_LABEL_SIZE, 'color': label_color}
            gl.ylabel_style = {'size': GRID_LABEL_SIZE, 'color': label_color}

            # Set formatters for proper degree symbols
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        # Apply grid spacing using MultipleLocator
        gl.xlocator = mticker.MultipleLocator(grid_spacing)
        gl.ylocator = mticker.MultipleLocator(grid_spacing)
        
        if bool(getattr(self.config, "show_grid_labels", False)):
            logger.info(f"Grid lines (with labels) added with {grid_spacing}° spacing")
        else:
            logger.info(f"Grid lines (no labels) added with {grid_spacing}° spacing")
    
    def setup_complete_basemap(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create basemap with all geographic features and grid lines.
        
        This is the primary method for creating a complete basemap ready for
        data plotting. It combines create_basemap(), add_geographic_features(),
        and add_gridlines() into a single convenient call.
        
        Returns:
            Tuple of (figure, axes) ready for meteorological data plotting
            
        Example:
            >>> from synoptic_charts.rendering import BasemapRenderer
            >>> from synoptic_charts import Config
            >>> import cartopy.crs as ccrs
            >>> 
            >>> # Create complete basemap
            >>> renderer = BasemapRenderer(region="CONUS", config=Config())
            >>> fig, ax = renderer.setup_complete_basemap()
            >>> 
            >>> # Plot data (use PlateCarree transform for lat/lon data)
            >>> # ax.contour(lons, lats, mslp, transform=ccrs.PlateCarree())
            >>> 
            >>> # Save figure
            >>> fig.savefig("synoptic_chart.png", bbox_inches='tight', dpi=150)
        """
        logger.info(f"Setting up complete basemap for {self.region}")
        
        # Create basemap
        fig, ax = self.create_basemap()
        
        # Add geographic features
        self.add_geographic_features(ax)
        
        # Add gridlines
        self.add_gridlines(ax)
        
        logger.info("Complete basemap setup finished")
        
        return fig, ax
