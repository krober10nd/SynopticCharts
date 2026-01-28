"""
Surface feature detection for synoptic meteorological analysis.

This module provides functions for detecting and filtering surface pressure
features (highs and lows) from MSLP data using spatial filtering techniques.
"""

import logging
import math
from typing import List, Tuple, Dict

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter, minimum_filter

from ..constants import EARTH_RADIUS

logger = logging.getLogger("synoptic_charts.calculations.features")


def calculate_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate great circle distance between two lat/lon points using haversine formula.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Distance in kilometers
        
    Example:
        >>> from synoptic_charts.calculations.features import calculate_haversine_distance
        >>> 
        >>> # Distance from New York to Los Angeles
        >>> dist = calculate_haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        >>> print(f"Distance: {dist:.0f} km")
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    distance_km = EARTH_RADIUS * c
    
    return distance_km


def detect_surface_features(
    mslp_data: xr.DataArray,
    min_pressure_difference: float = 4.0,
    search_radius: int = 10
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Detect surface high and low pressure centers.
    
    Uses morphological filtering to find local pressure extrema that represent
    surface pressure systems. Returns coordinates and pressure values of detected
    features.
    
    Args:
        mslp_data: Smoothed MSLP data in hPa with lat/lon coordinates
        min_pressure_difference: Minimum pressure difference from surroundings in hPa (default: 4.0)
        search_radius: Radius in grid points for local extrema search (default: 10)
        
    Returns:
        Dictionary with keys 'highs' and 'lows', each containing a list of
        (latitude, longitude, pressure) tuples
        
    Raises:
        ValueError: If MSLP data lacks required coordinates
        
    Example:
        >>> from synoptic_charts.calculations import smooth_mslp, detect_surface_features
        >>> 
        >>> # Smooth MSLP first
        >>> smoothed = smooth_mslp(mslp_data, sigma=2.0)
        >>> 
        >>> # Detect features
        >>> features = detect_surface_features(smoothed, min_pressure_difference=4.0)
        >>> 
        >>> print(f"Found {len(features['highs'])} highs:")
        >>> for lat, lon, pressure in features['highs']:
        ...     print(f"  High at ({lat:.1f}°N, {lon:.1f}°E): {pressure:.1f} hPa")
    """
    logger.info(
        f"Detecting surface features (min_diff={min_pressure_difference} hPa, "
        f"radius={search_radius} grid points)"
    )
    
    # Validate input
    if mslp_data is None or mslp_data.size == 0:
        raise ValueError("MSLP data is None or empty")
    
    if 'lat' not in mslp_data.coords or 'lon' not in mslp_data.coords:
        raise ValueError("MSLP data must have 'lat' and 'lon' coordinates")
    
    # Extract data and coordinates
    pressure_values = mslp_data.values
    lats = mslp_data.coords['lat'].values
    lons = mslp_data.coords['lon'].values
    
    # Handle NaN values
    nan_mask = np.isnan(pressure_values)
    if nan_mask.all():
        logger.warning("All MSLP values are NaN, no features can be detected")
        return {'highs': [], 'lows': []}
    
    # Calculate filter size from radius: size = 2*radius + 1
    filter_size = 2 * search_radius + 1
    logger.debug(f"Using filter size {filter_size} for radius {search_radius}")
    
    # Find local maxima (highs)
    logger.debug("Finding local pressure maxima (highs)")
    local_max = maximum_filter(pressure_values, size=filter_size, mode='constant', cval=np.nan)
    high_mask = (pressure_values == local_max) & ~nan_mask
    
    # Find local minima (lows)
    logger.debug("Finding local pressure minima (lows)")
    local_min = minimum_filter(pressure_values, size=filter_size, mode='constant', cval=np.nan)
    low_mask = (pressure_values == local_min) & ~nan_mask
    
    # Extract high features
    highs = []
    high_indices = np.argwhere(high_mask)
    
    for idx in high_indices:
        i, j = idx
        pressure = float(pressure_values[i, j])
        
        # Get surrounding values for pressure difference check (using radius)
        i_min = max(0, i - search_radius)
        i_max = min(pressure_values.shape[0], i + search_radius + 1)
        j_min = max(0, j - search_radius)
        j_max = min(pressure_values.shape[1], j + search_radius + 1)
        
        surrounding = pressure_values[i_min:i_max, j_min:j_max]
        surrounding_mean = np.nanmean(surrounding)
        
        # Check if pressure difference is significant
        if pressure - surrounding_mean >= min_pressure_difference:
            lat = float(lats[i] if lats.ndim == 1 else lats[i, j])
            lon = float(lons[j] if lons.ndim == 1 else lons[i, j])
            
            # Validate coordinates
            if -90 <= lat <= 90 and -180 <= lon <= 360:
                highs.append((lat, lon, pressure))
    
    # Extract low features
    lows = []
    low_indices = np.argwhere(low_mask)
    
    for idx in low_indices:
        i, j = idx
        pressure = float(pressure_values[i, j])
        
        # Get surrounding values for pressure difference check (using radius)
        i_min = max(0, i - search_radius)
        i_max = min(pressure_values.shape[0], i + search_radius + 1)
        j_min = max(0, j - search_radius)
        j_max = min(pressure_values.shape[1], j + search_radius + 1)
        
        surrounding = pressure_values[i_min:i_max, j_min:j_max]
        surrounding_mean = np.nanmean(surrounding)
        
        # Check if pressure difference is significant
        if surrounding_mean - pressure >= min_pressure_difference:
            lat = float(lats[i] if lats.ndim == 1 else lats[i, j])
            lon = float(lons[j] if lons.ndim == 1 else lons[i, j])
            
            # Validate coordinates
            if -90 <= lat <= 90 and -180 <= lon <= 360:
                lows.append((lat, lon, pressure))
    
    logger.info(f"Detected {len(highs)} highs and {len(lows)} lows (before filtering)")
    
    return {'highs': highs, 'lows': lows}


def filter_nearby_features(
    features: List[Tuple[float, float, float]],
    min_distance_km: float = 500.0,
    feature_type: str = "high"
) -> List[Tuple[float, float, float]]:
    """
    Remove duplicate features that are too close together.
    
    Filters a list of pressure features to ensure minimum separation distance.
    When features are closer than min_distance_km, keeps only the strongest
    (highest pressure for highs, lowest for lows).
    
    Args:
        features: List of (latitude, longitude, pressure) tuples
        min_distance_km: Minimum separation distance in kilometers (default: 500.0)
        feature_type: Type of feature - "high" or "low" (affects which to keep)
        
    Returns:
        Filtered list of features with minimum separation enforced
        
    Example:
        >>> from synoptic_charts.calculations import detect_surface_features, filter_nearby_features
        >>> 
        >>> # Detect features
        >>> features = detect_surface_features(smoothed_mslp)
        >>> 
        >>> # Filter nearby highs
        >>> filtered_highs = filter_nearby_features(
        ...     features['highs'],
        ...     min_distance_km=500.0,
        ...     feature_type="high"
        ... )
        >>> 
        >>> # Filter nearby lows
        >>> filtered_lows = filter_nearby_features(
        ...     features['lows'],
        ...     min_distance_km=500.0,
        ...     feature_type="low"
        ... )
    """
    logger.info(
        f"Filtering nearby {feature_type}s (min_distance={min_distance_km} km, "
        f"input count={len(features)})"
    )
    
    if not features:
        return []
    
    if len(features) == 1:
        logger.debug("Only one feature, no filtering needed")
        return features
    
    # Sort by pressure (descending for highs, ascending for lows)
    if feature_type.lower() == "high":
        sorted_features = sorted(features, key=lambda x: x[2], reverse=True)
    else:
        sorted_features = sorted(features, key=lambda x: x[2])
    
    # Keep track of filtered features
    filtered = []
    
    # Iteratively keep strongest features and remove nearby ones
    for feature in sorted_features:
        lat, lon, pressure = feature
        
        # Check if this feature is too close to any already kept feature
        is_too_close = False
        
        for kept_feature in filtered:
            kept_lat, kept_lon, _ = kept_feature
            distance = calculate_haversine_distance(lat, lon, kept_lat, kept_lon)
            
            if distance < min_distance_km:
                is_too_close = True
                logger.debug(
                    f"Filtering {feature_type} at ({lat:.1f}°, {lon:.1f}°, {pressure:.1f} hPa) - "
                    f"{distance:.0f} km from stronger feature"
                )
                break
        
        # Keep this feature if it's not too close to others
        if not is_too_close:
            filtered.append(feature)
    
    logger.info(
        f"Filtered {feature_type}s: {len(features)} → {len(filtered)} "
        f"({len(features) - len(filtered)} removed)"
    )
    
    return filtered
