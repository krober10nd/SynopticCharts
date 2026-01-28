"""
Individual rendering functions for meteorological layers.

This module provides low-level rendering functions for plotting meteorological
variables (MSLP, precipitation, thickness, surface features) on cartopy axes.
Each function handles proper coordinate transforms, styling, and colormap
configuration using herbie.paint for standardized meteorological visualization.
"""

import logging
from typing import Tuple, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

try:
    from herbie import paint
    HERBIE_PAINT_AVAILABLE = True
except ImportError:
    HERBIE_PAINT_AVAILABLE = False
    logging.warning("herbie.paint not available - using fallback colormaps")

from ..config import Config
from ..constants import (
    PRECIP_TYPES,
    THICKNESS_COLORMAP,
    THICKNESS_LEVELS,
    MSLP_CONTOUR_COLOR,
    MSLP_CONTOUR_LINEWIDTH,
    HIGH_MARKER_COLOR,
    LOW_MARKER_COLOR,
    FEATURE_FONT_SIZE
)

logger = logging.getLogger("synoptic_charts.rendering.layers")


def _is_dark_background(config: Config) -> bool:
    bg = getattr(config, "background_color", None)
    if not bg:
        return False
    try:
        r, g, b = mcolors.to_rgb(bg)
    except Exception:
        return False
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance < 0.35


def _interpolate_regular_latlon(
    da: xr.DataArray,
    *,
    factor: int,
    method: str,
) -> xr.DataArray:
    """Interpolate a DataArray on a regular 1D lat/lon grid.

    Only applies when both lat and lon coordinates are 1D and the data is 2D.
    For curvilinear (2D) lat/lon grids, returns the input unmodified.
    """

    if factor <= 1:
        return da

    lat_name = None
    lon_name = None
    for name in ("lat", "latitude", "y"):
        if name in da.coords:
            lat_name = name
            break
    for name in ("lon", "longitude", "x"):
        if name in da.coords:
            lon_name = name
            break

    if lat_name is None or lon_name is None:
        return da

    lat = da.coords[lat_name]
    lon = da.coords[lon_name]

    if lat.ndim != 1 or lon.ndim != 1 or da.ndim != 2:
        return da

    da_work = da
    try:
        da_work = da_work.sortby(lat_name)
    except Exception:
        pass
    try:
        da_work = da_work.sortby(lon_name)
    except Exception:
        pass

    lat_vals = np.asarray(da_work.coords[lat_name].values)
    lon_vals = np.asarray(da_work.coords[lon_name].values)

    if lat_vals.size < 2 or lon_vals.size < 2:
        return da

    new_lat = np.linspace(lat_vals.min(), lat_vals.max(), int(lat_vals.size * factor))
    new_lon = np.linspace(lon_vals.min(), lon_vals.max(), int(lon_vals.size * factor))

    try:
        return da_work.interp({lat_name: new_lat, lon_name: new_lon}, method=method)
    except Exception:
        logger.debug("Precip interpolation failed; plotting on native grid", exc_info=True)
        return da


def _truncate_cmap(base_cmap_name: str, *, n_bins: int, cmap_min: float) -> mcolors.Colormap:
    """Build a darker discrete colormap by sampling [cmap_min, 1.0] of a base cmap."""

    base = plt.get_cmap(base_cmap_name)
    lo = float(np.clip(cmap_min, 0.0, 0.999999))
    if n_bins <= 1:
        colors = [base(1.0)]
    else:
        colors = [base(x) for x in np.linspace(lo, 1.0, n_bins)]
    return mcolors.ListedColormap(colors, name=f"{base_cmap_name}_trunc_{lo:.2f}_{n_bins}")


def _extract_lat_lon(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (lats, lons, values) from a DataArray.

    Handles both 1D and 2D coordinate arrays and common coordinate naming
    differences across datasets/Herbie versions.
    """

    def _coord(name_options: Tuple[str, ...]) -> Optional[xr.DataArray]:
        for name in name_options:
            if name in da.coords:
                return da.coords[name]
            if name in da:
                return da[name]
        return None

    lat_da = _coord(("lat", "latitude", "y"))
    lon_da = _coord(("lon", "longitude", "x"))

    if lat_da is None or lon_da is None:
        raise ValueError(
            f"Data must have latitude/longitude coordinates. Found coords={list(da.coords)}"
        )

    values = da.values

    lats = lat_da.values
    lons = lon_da.values

    # Align values orientation for 1D coords.
    if lats.ndim == 1 and lons.ndim == 1 and values.ndim == 2:
        lat_dim = lat_da.dims[0] if lat_da.dims else None
        lon_dim = lon_da.dims[0] if lon_da.dims else None
        if lat_dim in da.dims and lon_dim in da.dims:
            if da.dims[:2] == (lon_dim, lat_dim):
                values = da.transpose(lat_dim, lon_dim).values
        # Validate shape
        if values.shape != (lats.size, lons.size):
            raise ValueError(
                f"Coordinate/value shape mismatch: values={values.shape}, "
                f"lat={lats.shape}, lon={lons.shape}"
            )

    # Validate 2D coords.
    if lats.ndim == 2 and lons.ndim == 2 and values.ndim == 2:
        if lats.shape != values.shape or lons.shape != values.shape:
            raise ValueError(
                f"2D coordinate/value shape mismatch: values={values.shape}, "
                f"lat={lats.shape}, lon={lons.shape}"
            )

    # Handle mixed 1D/2D by broadcasting to 2D.
    if values.ndim == 2 and (lats.ndim != lons.ndim):
        if lats.ndim == 1 and lons.ndim == 2 and lons.shape == values.shape:
            lats = np.broadcast_to(lats[:, None], values.shape)
        elif lons.ndim == 1 and lats.ndim == 2 and lats.shape == values.shape:
            lons = np.broadcast_to(lons[None, :], values.shape)

    return lats, lons, values


def render_mslp_contours(
    ax: plt.Axes,
    mslp_data: xr.DataArray,
    config: Config
) -> Any:
    """
    Render MSLP contour lines on cartopy axes.
    
    Args:
        ax: Matplotlib axes with cartopy projection
        mslp_data: MSLP data in hPa (already smoothed)
        config: Configuration with contour interval
        
    Returns:
        Contour object for label manipulation
        
    Example:
        >>> from synoptic_charts.rendering import render_mslp_contours
        >>> from synoptic_charts.calculations import smooth_mslp
        >>> 
        >>> smoothed = smooth_mslp(mslp_data, sigma=2.0)
        >>> contours = render_mslp_contours(ax, smoothed, config)
    """
    logger.info("Rendering MSLP contours")

    dark_theme = _is_dark_background(config)
    line_color = "#e6edf3" if dark_theme else MSLP_CONTOUR_COLOR
    label_color = "#e6edf3" if dark_theme else "black"
    halo_color = "#0b0f14" if dark_theme else "white"
    
    # Extract coordinates and values
    lats, lons, pressure_values = _extract_lat_lon(mslp_data)
    
    # Calculate contour levels
    valid_pressure = pressure_values[~np.isnan(pressure_values)]
    if valid_pressure.size == 0:
        logger.warning("No valid MSLP data to contour")
        return None
    
    p_min = np.floor(valid_pressure.min() / config.mslp_contour_interval) * config.mslp_contour_interval
    p_max = np.ceil(valid_pressure.max() / config.mslp_contour_interval) * config.mslp_contour_interval
    levels = np.arange(p_min, p_max + config.mslp_contour_interval, config.mslp_contour_interval)
    
    logger.debug(f"MSLP contour levels: {len(levels)} from {p_min} to {p_max} hPa")
    
    # Plot contours
    contours = ax.contour(
        lons, lats, pressure_values,
        levels=levels,
        colors=line_color,
        linewidths=MSLP_CONTOUR_LINEWIDTH,
        transform=ccrs.PlateCarree(),
        zorder=7
    )
    
    # Add contour labels (bold + halo so they stay readable over precip)
    label_texts = ax.clabel(contours, inline=True, fontsize=8, fmt='%d')
    for text in label_texts:
        text.set_fontweight('bold')
        text.set_color(label_color)
        text.set_path_effects([
            pe.Stroke(linewidth=2.2, foreground=halo_color),
            pe.Normal(),
        ])
        text.set_zorder(30)
    
    logger.info(f"Rendered {len(levels)} MSLP contours")
    
    return contours


def render_precipitation(
    ax: plt.Axes,
    precip_data: xr.DataArray,
    precip_type: str,
    config: Config
) -> Tuple[Any, Any]:
    """
    Render precipitation filled contours on cartopy axes.
    
    Uses herbie.paint.NWSPrecipitation for professional colormap and styling.
    
    Args:
        ax: Matplotlib axes with cartopy projection
        precip_data: Precipitation rate in inches/hour (already masked)
        precip_type: Precipitation type key ('rain', 'snow', 'frzr', 'sleet')
        config: Configuration object
        
    Returns:
        Tuple of (ContourSet object, paint_class) for colorbar creation
        
    Example:
        >>> contour_set, paint = render_precipitation(ax, masked_rain, 'rain', config)
        >>> # Later create colorbar: create_colorbar_from_paint(fig, contour_set, paint, position)
    """
    logger.info(f"Rendering {precip_type} precipitation")

    # Optional: interpolate to a finer regular lat/lon grid before plotting.
    precip_interp_factor =getattr(config, "precip_interp_factor", 1)
    precip_data = _interpolate_regular_latlon(
        precip_data,
        factor=precip_interp_factor,
        method=getattr(config, "precip_interp_method", "linear"),
    )
    
    # Extract coordinates and values
    lats, lons, precip_values = _extract_lat_lon(precip_data)
    
    # Check for valid data
    valid_precip = precip_values[~np.isnan(precip_values)]
    if valid_precip.size == 0 or valid_precip.max() == 0:
        logger.warning(f"No valid {precip_type} precipitation data to plot")
        return None, None
    
    logger.debug(f"{precip_type} range: {valid_precip.min():.4f} to {valid_precip.max():.2f} in/hr")

    # Enforce common categorical precipitation level intervals.
    enforced_levels = None
    enforced_cmap_name = None
    if precip_type in PRECIP_TYPES:
        enforced_levels = PRECIP_TYPES[precip_type]['levels']
        enforced_cmap_name = PRECIP_TYPES[precip_type].get('colormap')
    
    # Get per-type paint configuration
    paint_class = None
    plot_kwargs = {}
    
    # Map precipitation types to herbie paint classes (if available)
    herbie_paint_map = {
        'rain': 'NWSPrecipitation',
        'snow': 'NWSSnowfall',
        'frzr': 'NWSFreezingRain',
        'sleet': 'NWSIcePellets'
    }
    
    # Try to use herbie type-specific paint class
    if HERBIE_PAINT_AVAILABLE and precip_type in herbie_paint_map:
        paint_attr = herbie_paint_map[precip_type]
        if hasattr(paint, paint_attr):
            paint_class = getattr(paint, paint_attr)
            # Unpack herbie paint kwargs for consistent styling
            if hasattr(paint_class, 'kwargs2'):
                plot_kwargs = paint_class.kwargs2.copy()
            logger.debug(f"Using herbie.paint.{paint_attr} for {precip_type}")

    # Always apply our level intervals (requested) even if herbie.paint provides defaults.
    if enforced_levels is not None:
        plot_kwargs['levels'] = enforced_levels
        plot_kwargs['extend'] = 'max'

    # Always apply per-type colormap so precip types look distinct.
    if enforced_levels is not None and enforced_cmap_name:
        # 20 increments means 21 boundaries → 20 bins.
        n_bins = max(len(enforced_levels) - 1, 1)
        cmap_min = float(getattr(config, "precip_cmap_min", 0.0))
        cmap = _truncate_cmap(enforced_cmap_name, n_bins=n_bins, cmap_min=cmap_min)
        norm = mcolors.BoundaryNorm(enforced_levels, cmap.N)
        plot_kwargs['cmap'] = cmap
        plot_kwargs['norm'] = norm
    
    # Fallback to manual colormap from constants if herbie not available or lacks type-specific class
    if not plot_kwargs:
        if precip_type in PRECIP_TYPES:
            colormap = PRECIP_TYPES[precip_type].get('colormap', 'Blues')
            levels = PRECIP_TYPES[precip_type]['levels']
            cmap_min = float(getattr(config, "precip_cmap_min", 0.0))
            n_bins = max(len(levels) - 1, 1)
            cmap = _truncate_cmap(str(colormap), n_bins=n_bins, cmap_min=cmap_min)
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            plot_kwargs = {
                'levels': levels,
                'cmap': cmap,
                'norm': norm,
                'extend': 'max'
            }
            logger.debug(f"Using type-specific colormap '{colormap}' for {precip_type}")
        else:
            logger.warning(f"Unknown precip type {precip_type}, using default")
            plot_kwargs = {'cmap': 'Blues', 'extend': 'max'}
    
    # Add standard kwargs
    plot_kwargs.update({
        'transform': ccrs.PlateCarree(),
        'zorder': 1 if precip_type == 'unclassified' else 2,
        'alpha': 1.0,
        # Prevent white seams at boundaries between filled polygons.
        'antialiased': False,
        'linewidths': 0.0,
    })
    
    # Plot filled contours
    contour_set = ax.contourf(lons, lats, precip_values, **plot_kwargs)

    # Some Matplotlib backends still show thin white edges; force edges to match faces.
    try:
        for coll in contour_set.collections:
            coll.set_edgecolor('face')
            coll.set_linewidth(0.0)
            coll.set_antialiased(False)
    except Exception:
        pass
    
    logger.info(f"Rendered {precip_type} precipitation contours")
    
    return contour_set, paint_class


def render_precipitation_accumulation(
    ax: plt.Axes,
    precip_accum_data: xr.DataArray,
    config: Config,
) -> Tuple[Any, Any]:
    """Render accumulated precipitation (inches) filled contours.

    Uses herbie.paint.NWSPrecipitation for standardized styling.
    """

    logger.info("Rendering accumulated precipitation")

    precip_interp_factor = getattr(config, "precip_interp_factor", 1)
    precip_accum_data = _interpolate_regular_latlon(
        precip_accum_data,
        factor=precip_interp_factor,
        method=getattr(config, "precip_interp_method", "linear"),
    )

    lats, lons, values = _extract_lat_lon(precip_accum_data)

    valid = values[~np.isnan(values)]
    if valid.size == 0 or valid.max() == 0:
        logger.warning("No valid accumulated precipitation data to plot")
        return None, None

    logger.debug(f"Accum precip range: {valid.min():.4f} to {valid.max():.2f} in")

    paint_class = None
    plot_kwargs = {}

    if HERBIE_PAINT_AVAILABLE and hasattr(paint, "NWSPrecipitation"):
        paint_class = getattr(paint, "NWSPrecipitation")
        if hasattr(paint_class, "kwargs2"):
            plot_kwargs = paint_class.kwargs2.copy()
        logger.debug("Using herbie.paint.NWSPrecipitation for accumulated precip")

    # If herbie.paint provides a BoundaryNorm, also provide the explicit
    # boundaries as levels so contourf bins match the norm.
    try:
        norm = plot_kwargs.get("norm")
        if norm is not None and hasattr(norm, "boundaries"):
            boundaries = np.asarray(norm.boundaries)
            if boundaries.size >= 2:
                plot_kwargs["levels"] = boundaries
                plot_kwargs.setdefault("extend", "max")
    except Exception:
        pass

    # Mask out values below the first non-zero boundary (prevents uniform
    # "under" color fill when tiny nonzero values exist everywhere).
    try:
        min_bin = None
        norm = plot_kwargs.get("norm")
        if norm is not None and hasattr(norm, "boundaries"):
            b = np.asarray(norm.boundaries)
            positives = b[b > 0]
            if positives.size:
                min_bin = float(positives.min())
        thr_cfg = float(getattr(config, "accum_trace_threshold", 0.0) or 0.0)
        thr = max(thr_cfg, min_bin) if min_bin is not None else thr_cfg
        if thr > 0:
            values = np.where(values >= thr, values, np.nan)
    except Exception:
        pass

    # Fallback if herbie.paint isn't available.
    if not plot_kwargs:
        plot_kwargs = {
            "cmap": "Blues",
            "extend": "max",
        }

    plot_kwargs.update(
        {
            "transform": ccrs.PlateCarree(),
            "zorder": 2,
            "alpha": 1.0,
            "antialiased": False,
            "linewidths": 0.0,
        }
    )

    contour_set = ax.contourf(lons, lats, values, **plot_kwargs)

    try:
        for coll in contour_set.collections:
            coll.set_edgecolor("face")
            coll.set_linewidth(0.0)
            coll.set_antialiased(False)
    except Exception:
        pass

    logger.info("Rendered accumulated precipitation contours")

    return contour_set, paint_class


def render_thickness(
    ax: plt.Axes,
    thickness_data: xr.DataArray,
    config: Config
) -> Tuple[Any, Any]:
    """
    Render 1000-500mb thickness line contours on cartopy axes.
    
    Args:
        ax: Matplotlib axes with cartopy projection
        thickness_data: Thickness in decameters
        config: Configuration object
        
    Returns:
        Tuple of (ContourSet object, paint_class). Thickness is line-only, so
        paint_class is typically None.
        
    Example:
        >>> contour_set, paint = render_thickness(ax, thickness_data, config)
    """
    logger.info("Rendering thickness line contours")

    dark_theme = _is_dark_background(config)
    label_color = "#e6edf3" if dark_theme else "black"
    halo_color = "#0b0f14" if dark_theme else "white"
    
    # Extract coordinates and values
    lats, lons, thickness_values = _extract_lat_lon(thickness_data)
    
    # Check for valid data
    valid_thickness = thickness_values[~np.isnan(thickness_values)]
    if valid_thickness.size == 0:
        logger.warning("No valid thickness data to plot")
        return None, None
    
    logger.debug(f"Thickness range: {valid_thickness.min():.0f} to {valid_thickness.max():.0f} dam")

    # Derive thickness levels from config to avoid extremely dense contouring.
    t_min = float(np.nanmin(thickness_values))
    t_max = float(np.nanmax(thickness_values))
    interval = float(getattr(config, "thickness_contour_interval", 10.0) or 10.0)
    if interval <= 0:
        interval = 10.0

    def _make_levels(step: float) -> np.ndarray:
        lo = np.floor(t_min / step) * step
        hi = np.ceil(t_max / step) * step
        return np.arange(lo, hi + step, step)

    levels = _make_levels(interval)
    # Cap the number of levels for performance; coarsen if needed.
    while levels.size > 40:
        interval *= 2.0
        levels = _make_levels(interval)
    
    # Thickness is intentionally line-only (no fill) so the precipitation layer
    # remains easy to see.
    paint_class = None
    cmap = plt.get_cmap(THICKNESS_COLORMAP, max(levels.size - 1, 1))
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    contour_set = ax.contour(
        lons,
        lats,
        thickness_values,
        levels=levels,
        cmap=cmap,
        norm=norm,
        linewidths=2.2,
        linestyles='--',
        transform=ccrs.PlateCarree(),
        zorder=6,
        alpha=1.0,
    )

    # Label thickness contours in decameters (dam) every 10 dam.
    labels = ax.clabel(
        contour_set,
        levels=levels,
        inline=True,
        fontsize=11,
        fmt=lambda v: f"{int(round(v))}",
    )
    for text in labels:
        text.set_fontweight('bold')
        text.set_color(label_color)
        text.set_path_effects([
            pe.Stroke(linewidth=2.2, foreground=halo_color),
            pe.Normal(),
        ])
        text.set_zorder(30)

    logger.info("Rendered thickness line contours")

    return contour_set, paint_class


def render_surface_features(
    ax: plt.Axes,
    features_dict: dict
) -> None:
    """
    Render surface pressure features (highs and lows) on cartopy axes.
    
    Args:
        ax: Matplotlib axes with cartopy projection
        features_dict: Dictionary with 'highs' and 'lows' keys, each containing
                      list of (lat, lon, pressure) tuples
                      
    Example:
        >>> from synoptic_charts.calculations import detect_surface_features, filter_nearby_features
        >>> 
        >>> features = detect_surface_features(smoothed_mslp)
        >>> filtered_highs = filter_nearby_features(features['highs'], feature_type='high')
        >>> filtered_lows = filter_nearby_features(features['lows'], feature_type='low')
        >>> render_surface_features(ax, {'highs': filtered_highs, 'lows': filtered_lows})
    """
    logger.info("Rendering surface features")
    
    highs_count = 0
    lows_count = 0

    def _norm_lon(lon: float) -> float:
        try:
            lon_f = float(lon)
        except Exception:
            return lon
        return ((lon_f + 180.0) % 360.0) - 180.0
    
    # Render high pressure centers
    if 'highs' in features_dict and features_dict['highs']:
        for lat, lon, pressure in features_dict['highs']:
            lon = _norm_lon(lon)
            # Plot 'H' marker
            ax.text(
                lon, lat, 'H',
                color=HIGH_MARKER_COLOR,
                fontsize=FEATURE_FONT_SIZE,
                fontweight='bold',
                ha='center',
                va='center',
                transform=ccrs.PlateCarree(),
                zorder=10,
                clip_on=True,
            )
            
            # Optionally add pressure value below
            ax.text(
                lon, lat - 0.5, f'{int(pressure)}',
                color=HIGH_MARKER_COLOR,
                fontsize=FEATURE_FONT_SIZE - 2,
                ha='center',
                va='top',
                transform=ccrs.PlateCarree(),
                zorder=10,
                clip_on=True,
            )
            
            highs_count += 1
    
    # Render low pressure centers
    if 'lows' in features_dict and features_dict['lows']:
        for lat, lon, pressure in features_dict['lows']:
            lon = _norm_lon(lon)
            # Plot 'L' marker
            ax.text(
                lon, lat, 'L',
                color=LOW_MARKER_COLOR,
                fontsize=FEATURE_FONT_SIZE,
                fontweight='bold',
                ha='center',
                va='center',
                transform=ccrs.PlateCarree(),
                zorder=10,
                clip_on=True,
            )
            
            # Optionally add pressure value below
            ax.text(
                lon, lat - 0.5, f'{int(pressure)}',
                color=LOW_MARKER_COLOR,
                fontsize=FEATURE_FONT_SIZE - 2,
                ha='center',
                va='top',
                transform=ccrs.PlateCarree(),
                zorder=10,
                clip_on=True,
            )
            
            lows_count += 1
    
    logger.info(f"Rendered {highs_count} highs and {lows_count} lows")


def create_colorbar_from_paint(
    fig: plt.Figure,
    contour_set: Any,
    paint_class: Any,
    position: list,
    label_override: Optional[str] = None
) -> Any:
    """
    Create colorbar using herbie paint class configuration.
    
    Args:
        fig: Matplotlib figure
        contour_set: ContourSet from rendering function
        paint_class: Herbie paint class (e.g., NWSPrecipitation)
        position: List [left, bottom, width, height] for colorbar axes
        label_override: Optional string to override default label
        
    Returns:
        Colorbar object
        
    Example:
        >>> # After rendering precipitation
        >>> contour_set, paint = render_precipitation(ax, precip_data, 'rain', config)
        >>> cbar = create_colorbar_from_paint(
        ...     fig, contour_set, paint,
        ...     position=[0.15, 0.05, 0.7, 0.02],
        ...     label_override="Rainfall Rate (in/hr)"
        ... )
    """
    logger.info("Creating colorbar from paint class")
    
    if contour_set is None:
        logger.warning("Cannot create colorbar: contour_set is None")
        return None
    
    # Create colorbar axes
    cax = fig.add_axes(position)
    
    # Get colorbar kwargs from paint class if available
    cbar_kwargs = {'orientation': 'horizontal'}
    
    if paint_class is not None and hasattr(paint_class, 'cbar_kwargs2'):
        # Use herbie paint colorbar configuration
        paint_cbar_kwargs = paint_class.cbar_kwargs2.copy()
        # Merge with defaults, preserving orientation
        paint_cbar_kwargs['orientation'] = 'horizontal'
        cbar_kwargs.update(paint_cbar_kwargs)
        logger.debug("Using herbie paint colorbar configuration")
    else:
        # Use basic matplotlib colorbar defaults
        logger.debug("Using basic matplotlib colorbar (paint_class unavailable)")
    
    # Create colorbar
    cbar = fig.colorbar(contour_set, cax=cax, **cbar_kwargs)
    
    # Override label if provided
    if label_override:
        cbar.set_label(label_override)
    
    # Style ticks and labels
    cbar.ax.tick_params(labelsize=8)
    
    logger.info("Colorbar created")
    
    return cbar
