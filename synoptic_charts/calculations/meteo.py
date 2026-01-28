"""
Meteorological calculations for synoptic chart data processing.

This module provides functions for transforming and processing meteorological
variables including MSLP smoothing, precipitation conversions, and thickness
calculations using MetPy's unit system.
"""

import logging
from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, binary_dilation

logger = logging.getLogger("synoptic_charts.calculations.meteo")


def expand_categorical_mask(
    categorical_mask: xr.DataArray,
    *,
    iterations: int = 1,
) -> xr.DataArray:
    """Expand a categorical precip mask by one or more grid cells.

    This is a morphological dilation (8-neighbor) that fills small gaps between
    adjacent precipitation-type regions that often appear due to blocky GRIB
    categorical fields.

    Args:
        categorical_mask: Mask where precip type exists (bool/int/float).
        iterations: Number of dilation iterations (0 disables).

    Returns:
        DataArray with the same coords/dims as input, values in {0,1} (int8).
    """

    if categorical_mask is None:
        raise ValueError("categorical_mask is None")

    iters = int(iterations)
    if iters <= 0:
        # Normalize to 0/1 int8 for downstream use.
        return xr.where(categorical_mask > 0, 1, 0).astype("int8")

    mask_bool = np.asarray(categorical_mask.values > 0)
    # 8-neighbor connectivity.
    structure = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(mask_bool, structure=structure, iterations=iters)

    out = xr.DataArray(
        dilated.astype("int8"),
        coords=categorical_mask.coords,
        dims=categorical_mask.dims,
        name=getattr(categorical_mask, "name", None),
        attrs=getattr(categorical_mask, "attrs", {}).copy(),
    )
    out.attrs["dilation_iterations"] = iters
    return out


def _normalize_height_to_meters(height: xr.DataArray) -> xr.DataArray:
    """Normalize common height units to meters.

    Herbie/GRIB sources may provide geopotential heights in meters (gpm) or
    decameters. This function converts decameters to meters and normalizes
    common unit labels.
    """
    units = (height.attrs.get("units") or "").strip().lower()

    # Common labels for meters / geopotential meters
    meter_like = {"m", "meter", "meters", "metre", "metres", "gpm"}
    dam_like = {"dam", "decameter", "decameters", "dekameter", "dekameters"}

    if units in dam_like:
        out = height * 10.0
        out.attrs = height.attrs.copy()
        out.attrs["units"] = "meter"
        return out

    if units in meter_like or units == "":
        # If units are missing, assume meters.
        out = height.copy()
        if units == "":
            out.attrs = height.attrs.copy()
            out.attrs["units"] = "meter"
        return out

    # Best-effort: leave as-is and let MetPy attempt conversion.
    return height


def smooth_mslp(
    mslp_data: xr.DataArray,
    sigma: float = 2.0
) -> xr.DataArray:
    """
    Apply Gaussian smoothing to MSLP data for cleaner contours.
    
    This function smooths mean sea level pressure data using a Gaussian filter
    to reduce noise and produce cleaner contour lines while preserving the
    overall pressure pattern and gradient structure.
    
    Args:
        mslp_data: MSLP data in hPa with lat/lon coordinates
        sigma: Standard deviation for Gaussian kernel (default: 2.0)
               Higher values = more smoothing
    
    Returns:
        Smoothed MSLP DataArray with preserved coordinates and updated attributes
        
    Raises:
        ValueError: If MSLP values are outside reasonable range (900-1050 hPa)
        
    Example:
        >>> import xarray as xr
        >>> from synoptic_charts.calculations import smooth_mslp
        >>> 
        >>> # Assume mslp_data is from downloader
        >>> smoothed = smooth_mslp(mslp_data, sigma=2.5)
        >>> print(f"Smoothing reduced range from {mslp_data.std():.2f} to {smoothed.std():.2f} hPa")
    """
    logger.info(f"Smoothing MSLP data with Gaussian filter (sigma={sigma})")
    
    # Validate input
    if mslp_data is None or mslp_data.size == 0:
        raise ValueError("MSLP data is None or empty")
    
    # Check for reasonable MSLP range (900-1050 hPa)
    valid_data = mslp_data.where(~np.isnan(mslp_data), drop=True)
    if valid_data.size > 0:
        min_val = float(valid_data.min())
        max_val = float(valid_data.max())
        
        if min_val < 900 or max_val > 1050:
            logger.warning(
                f"MSLP values outside typical range: [{min_val:.1f}, {max_val:.1f}] hPa"
            )
            if min_val < 850 or max_val > 1100:
                raise ValueError(
                    f"MSLP values outside reasonable range: [{min_val:.1f}, {max_val:.1f}] hPa"
                )
    
    # Handle NaN values by creating a mask
    data_values = mslp_data.values
    nan_mask = np.isnan(data_values)
    
    if nan_mask.any():
        # Fill NaNs with mean for smoothing, will restore later
        fill_value = np.nanmean(data_values)
        data_filled = np.where(nan_mask, fill_value, data_values)
        logger.debug(f"Filled {nan_mask.sum()} NaN values for smoothing")
    else:
        data_filled = data_values
    
    # Apply Gaussian filter
    try:
        smoothed_values = gaussian_filter(data_filled, sigma=sigma, mode='nearest')
    except Exception as e:
        logger.error(f"Gaussian filter failed: {e}")
        raise
    
    # Restore NaN mask
    if nan_mask.any():
        smoothed_values = np.where(nan_mask, np.nan, smoothed_values)
    
    # Create new DataArray with smoothed values
    smoothed = xr.DataArray(
        smoothed_values,
        coords=mslp_data.coords,
        dims=mslp_data.dims,
        attrs=mslp_data.attrs.copy()
    )
    
    # Update attributes to note smoothing
    smoothed.attrs['processing'] = f'Gaussian smoothed (sigma={sigma})'
    if 'history' in smoothed.attrs:
        smoothed.attrs['history'] += f'; smoothed with sigma={sigma}'
    else:
        smoothed.attrs['history'] = f'Gaussian smoothed (sigma={sigma})'
    
    logger.info(f"MSLP smoothing complete (reduced std from {mslp_data.std().values:.2f} to {smoothed.std().values:.2f} hPa)")
    
    return smoothed


def convert_precip_rate_to_inches_per_hour(
    precip_rate: xr.DataArray
) -> xr.DataArray:
    """
    Convert precipitation rate from kg/m²/s to inches/hour.
    
    Applies the conversion: kg/m²/s = mm/s → mm/hr → inches/hr
    Conversion factor: 3600 (s to hr) * 0.0393701 (mm to inches) = 141.732
    
    Args:
        precip_rate: Precipitation rate in kg/m²/s
        
    Returns:
        Precipitation rate in inches/hour with updated units attribute
        
    Example:
        >>> from synoptic_charts.calculations import convert_precip_rate_to_inches_per_hour
        >>> 
        >>> # Assume precip_rate is from downloader in kg/m²/s
        >>> precip_inches = convert_precip_rate_to_inches_per_hour(precip_rate)
        >>> print(f"Max precip: {precip_inches.max().values:.2f} in/hr")
    """
    logger.info("Converting precipitation rate from kg/m²/s to inches/hour")
    
    # Validate input
    if precip_rate is None or precip_rate.size == 0:
        raise ValueError("Precipitation rate data is None or empty")

    # Avoid double conversion if upstream already produced inches/hour.
    units = str(precip_rate.attrs.get("units", "") or "").lower()
    if units in {"inches/hour", "inch/hour", "in/hr", "in hr-1", "inches per hour"}:
        return precip_rate
    
    # Conversion: kg/m²/s → mm/s (×1) → mm/hr (×3600) → inches/hr (×0.0393701)
    conversion_factor = 3600.0 * 0.0393701  # = 141.732
    
    # Apply conversion
    precip_inches = precip_rate * conversion_factor
    
    # Update attributes
    precip_inches.attrs = precip_rate.attrs.copy()
    precip_inches.attrs['units'] = 'inches/hour'
    precip_inches.attrs['long_name'] = 'Precipitation Rate'
    
    if 'history' in precip_inches.attrs:
        precip_inches.attrs['history'] += '; converted to inches/hour'
    else:
        precip_inches.attrs['history'] = 'Converted from kg/m²/s to inches/hour'
    
    # Log statistics
    valid_precip = precip_inches.where(precip_inches > 0, drop=True)
    if valid_precip.size > 0:
        logger.debug(
            f"Precipitation range: {valid_precip.min().values:.4f} to "
            f"{valid_precip.max().values:.2f} inches/hour"
        )
    
    logger.info("Precipitation rate conversion complete")
    
    return precip_inches


def calculate_thickness(
    height_500mb: xr.DataArray,
    height_1000mb: xr.DataArray
) -> Optional[xr.DataArray]:
    """
    Calculate 1000-500 mb thickness in decameters.
    
    Thickness is the difference in geopotential height between two pressure
    levels and is proportional to the mean temperature of the layer. It's
    commonly used to identify air mass boundaries and forecast precipitation type.
    
    Args:
        height_500mb: 500mb geopotential height in meters
        height_1000mb: 1000mb geopotential height in meters
        
    Returns:
        Thickness in decameters, or None if data incompatible
        
    Raises:
        ValueError: If input arrays have incompatible grids
        
    Note:
        The 1000mb level may not exist over high terrain. Missing values
        are handled with NaN in the output array.
        
    Example:
        >>> from synoptic_charts.calculations import calculate_thickness
        >>> 
        >>> # Assume heights are from downloader
        >>> thickness = calculate_thickness(height_500mb, height_1000mb)
        >>> print(f"Thickness range: {thickness.min().values:.0f} to {thickness.max().values:.0f} dam")
        >>> # Typical range: 480-600 decameters
    """
    logger.info("Calculating 1000-500mb thickness")
    
    # Validate inputs
    if height_500mb is None or height_1000mb is None:
        logger.error("One or both height arrays are None")
        return None
    
    if height_500mb.size == 0 or height_1000mb.size == 0:
        logger.error("One or both height arrays are empty")
        return None
    
    # Check for compatible grids
    if height_500mb.shape != height_1000mb.shape:
        logger.error(
            f"Incompatible grid shapes: 500mb={height_500mb.shape}, "
            f"1000mb={height_1000mb.shape}"
        )
        raise ValueError("Height arrays must have the same shape")
    
    # Check coordinate compatibility
    for coord in ['lat', 'lon']:
        if coord in height_500mb.coords and coord in height_1000mb.coords:
            if not np.allclose(height_500mb[coord].values, height_1000mb[coord].values):
                logger.warning(f"Coordinate '{coord}' values differ between height arrays")
    
    # Use MetPy units for proper unit handling
    try:
        # Normalize common unit variants to meters before quantifying.
        height_500mb = _normalize_height_to_meters(height_500mb)
        height_1000mb = _normalize_height_to_meters(height_1000mb)

        # Quantify the arrays with MetPy units (assumes meters)
        logger.debug("Attaching MetPy units to height arrays")
        height_500_quantified = height_500mb.metpy.quantify()
        height_1000_quantified = height_1000mb.metpy.quantify()
        
        # Calculate thickness (difference in geopotential height)
        thickness_with_units = height_500_quantified - height_1000_quantified
        
        # Convert to decameters using MetPy
        try:
            # Try MetPy's convert_units if available
            thickness_dam_quantified = thickness_with_units.metpy.convert_units('decameter')
        except (AttributeError, ValueError):
            # Fallback: manual conversion if MetPy doesn't support 'decameter' directly
            logger.debug("Converting to decameters manually")
            thickness_dam_quantified = thickness_with_units.metpy.convert_units('meter') / 10.0
        
        # Dequantify to return standard xarray DataArray
        thickness_dam = thickness_dam_quantified.metpy.dequantify()
        
    except Exception as e:
        logger.error(f"Failed to calculate thickness with MetPy units: {e}")
        return None
    
    # Update attributes
    thickness_dam.attrs['units'] = 'decameters'
    thickness_dam.attrs['long_name'] = '1000-500mb Thickness'
    thickness_dam.attrs['standard_name'] = 'thickness'
    thickness_dam.attrs['description'] = 'Difference in geopotential height between 1000mb and 500mb'
    
    # Validate thickness range (typical: 450-600 dam)
    valid_thickness = thickness_dam.where(~np.isnan(thickness_dam), drop=True)
    if valid_thickness.size > 0:
        min_thick = float(valid_thickness.min())
        max_thick = float(valid_thickness.max())
        
        logger.debug(f"Thickness range: {min_thick:.1f} to {max_thick:.1f} dam")
        
        if min_thick < 450 or max_thick > 600:
            logger.warning(
                f"Thickness values outside typical range: [{min_thick:.1f}, {max_thick:.1f}] dam"
            )
            if min_thick < 400 or max_thick > 650:
                logger.error(
                    f"Thickness values outside reasonable range: [{min_thick:.1f}, {max_thick:.1f}] dam"
                )
    
    logger.info("Thickness calculation complete")
    
    return thickness_dam


def mask_trace_precipitation(
    precip_data: xr.DataArray,
    trace_threshold: float = 0.001,
    categorical_mask: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """
    Mask precipitation values below trace threshold.
    
    Removes precipitation values below a specified threshold to eliminate
    trace amounts that are not meteorologically significant. Optionally
    applies a categorical precipitation mask first.
    
    Args:
        precip_data: Precipitation rate in inches/hour
        trace_threshold: Minimum value to display (default: 0.001 in/hr)
        categorical_mask: Optional binary mask (1 where precip type exists, 0 elsewhere)
        
    Returns:
        Masked precipitation DataArray with values below threshold set to NaN
        
    Example:
        >>> from synoptic_charts.calculations import mask_trace_precipitation
        >>> 
        >>> # Mask light precipitation
        >>> masked = mask_trace_precipitation(precip_inches, trace_threshold=0.01)
        >>> 
        >>> # Mask with categorical rain mask
        >>> rain_masked = mask_trace_precipitation(
        ...     precip_inches,
        ...     trace_threshold=0.01,
        ...     categorical_mask=rain_categorical
        ... )
    """
    logger.info(f"Masking trace precipitation (threshold={trace_threshold} in/hr)")
    
    # Validate input
    if precip_data is None or precip_data.size == 0:
        raise ValueError("Precipitation data is None or empty")
    
    # Start with original data
    masked_data = precip_data.copy()
    
    # Apply categorical mask first if provided
    if categorical_mask is not None:
        logger.debug("Applying categorical precipitation mask")
        
        # Ensure masks have compatible shapes
        if categorical_mask.shape != precip_data.shape:
            logger.warning(
                f"Categorical mask shape {categorical_mask.shape} differs from "
                f"precipitation data shape {precip_data.shape}"
            )
        
        # Apply mask: keep only where categorical mask indicates precip.
        # Some sources provide boolean masks, others provide integer/category codes.
        masked_data = xr.where(categorical_mask > 0, masked_data, np.nan)
    
    # Always drop non-positive values so "no precip" (0) does not render as a
    # near-white fill that looks like transparency when plotting.
    masked_data = xr.where(masked_data > 0, masked_data, np.nan)

    # Apply trace threshold (when enabled). If threshold is 0, do not mask.
    if trace_threshold > 0:
        masked_data = xr.where(masked_data >= trace_threshold, masked_data, np.nan)
    
    # Count masked points
    original_valid = np.isfinite(precip_data.values).sum()
    masked_valid = np.isfinite(masked_data.values).sum()
    masked_count = original_valid - masked_valid
    
    if original_valid > 0:
        masked_percent = (masked_count / original_valid) * 100
        logger.debug(
            f"Masked {masked_count} points ({masked_percent:.1f}%) below threshold"
        )
    
    # Update attributes
    masked_data.attrs = precip_data.attrs.copy()
    masked_data.attrs['trace_threshold'] = trace_threshold
    masked_data.attrs['trace_threshold_units'] = 'inches/hour'
    
    if 'history' in masked_data.attrs:
        masked_data.attrs['history'] += f'; masked below {trace_threshold} in/hr'
    else:
        masked_data.attrs['history'] = f'Masked below {trace_threshold} in/hr'
    
    if categorical_mask is not None:
        masked_data.attrs['categorical_mask_applied'] = True
    
    logger.info("Trace precipitation masking complete")
    
    return masked_data
