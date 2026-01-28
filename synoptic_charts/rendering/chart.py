"""
Orchestration module for complete synoptic chart creation.

This module provides the SynopticChart class that coordinates data processing,
layer rendering, and chart assembly. It manages the complete workflow from
raw model data to finished synoptic charts with all meteorological layers.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .basemap import BasemapRenderer
from .layers import (
    render_mslp_contours,
    render_precipitation,
    render_precipitation_accumulation,
    render_thickness,
    render_surface_features
)
from .annotations import annotate_chart
from ..calculations.meteo import (
    smooth_mslp,
    convert_precip_rate_to_inches_per_hour,
    calculate_thickness,
    mask_trace_precipitation
    ,expand_categorical_mask
)
from ..calculations.features import (
    detect_surface_features,
    filter_nearby_features
)
from ..config import Config

logger = logging.getLogger("synoptic_charts.rendering.chart")


class SynopticChart:
    """
    Orchestrate complete synoptic chart creation.
    
    This class coordinates data processing through the calculations module
    and layer rendering through the layers module to create complete synoptic
    meteorological charts with proper layer stacking and styling.
    
    The rendering workflow:
    1. Setup basemap with geographic features
    2. Render thickness layer (zorder=2)
    3. Render precipitation layers (zorder=3)
    4. Render MSLP contours (zorder=5)
    5. Render surface features H/L (zorder=10)
    
    Attributes:
        region: Region name from REGIONS dictionary
        config: Configuration object with display settings
        basemap_renderer: BasemapRenderer instance for cartographic setup
        fig: Matplotlib Figure (None until render_chart called)
        ax: Matplotlib Axes (None until render_chart called)
        
    Example:
        >>> from synoptic_charts.rendering import SynopticChart
        >>> from synoptic_charts import Config
        >>> from synoptic_charts.data import ModelDownloader
        >>> from datetime import datetime
        >>> 
        >>> # Download data
        >>> downloader = ModelDownloader("GFS", datetime(2024, 1, 15, 0), Config(), 24)
        >>> data = downloader.fetch_all_data()
        >>> 
        >>> # Create and render chart
        >>> chart = SynopticChart(region="CONUS", config=Config())
        >>> fig, ax = chart.render_chart(
        ...     data, "GFS",
        ...     init_time=datetime(2024, 1, 15, 0),
        ...     valid_time=datetime(2024, 1, 16, 0),
        ...     lead_time=24
        ... )
        >>> 
        >>> # Save chart
        >>> chart.save_chart("synoptic_chart.png")
    """
    
    def __init__(
        self,
        region: str = "CONUS",
        config: Optional[Config] = None
    ):
        """
        Initialize SynopticChart orchestrator.
        
        Args:
            region: Region name from REGIONS dictionary (default: "CONUS")
            config: Configuration object (default: creates new Config instance)
        """
        self.region = region
        self.config = config if config is not None else Config()
        
        # Create basemap renderer
        self.basemap_renderer = BasemapRenderer(
            region=region,
            config=self.config
        )
        
        # Chart components (created during rendering)
        self.fig = None
        self.ax = None
        
        # Track rendered layers for external access (e.g., colorbar creation)
        self._rendered_layers = {}
        
        # Track annotations (created during rendering if annotate=True)
        self._annotations = None
        
        logger.info(f"Initialized SynopticChart for region '{region}'")
    
    def render_chart(
        self,
        data_dict: Dict[str, Any],
        model_name: str,
        init_time: datetime,
        valid_time: datetime,
        lead_time: int,
        annotate: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render complete synoptic chart from model data.
        
        Args:
            data_dict: Dictionary from ModelDownloader.fetch_all_data() with keys:
                      'mslp', 'precip_rate', 'precip_categorical', 'geopotential_heights'
            model_name: Model name for annotation (e.g., "GFS", "ECMWF")
            init_time: Datetime of model initialization
            valid_time: Datetime of valid time
            lead_time: Forecast hour
            annotate: If True, add title, model info, and colorbars (default: True)
            
        Returns:
            Tuple of (figure, axes) with rendered chart
            
        Example:
            >>> chart = SynopticChart("CONUS")
            >>> fig, ax = chart.render_chart(
            ...     data_dict=downloader.fetch_all_data(),
            ...     model_name="GFS",
            ...     init_time=datetime(2024, 1, 15, 0),
            ...     valid_time=datetime(2024, 1, 16, 0),
            ...     lead_time=24,
            ...     annotate=True
            ... )
        """
        logger.info(
            f"Rendering synoptic chart: {model_name} "
            f"init={init_time.strftime('%Y%m%d %HZ')} "
            f"valid={valid_time.strftime('%Y%m%d %HZ')} "
            f"f{lead_time:03d}"
        )
        
        # Setup basemap
        self.fig, self.ax = self.basemap_renderer.setup_complete_basemap()
        
        # Render layers in proper z-order
        try:
            # Layer 1: Precipitation (lowest)
            precip_mode = str(getattr(self.config, "precip_mode", "rate") or "rate").strip().lower()
            if precip_mode == "accumulated":
                if data_dict.get("precip_accum") is not None:
                    self._render_precipitation_accumulation_layer(data_dict["precip_accum"])
                else:
                    logger.warning("Precip accumulation data not available, skipping precipitation layer")
            else:
                if 'precip_rate' in data_dict and data_dict['precip_rate'] is not None:
                    self._render_precipitation_layers(
                        data_dict['precip_rate'],
                        data_dict.get('precip_categorical')
                    )
                else:
                    logger.warning("Precipitation data not available, skipping precipitation layers")

            # Layer 2: Thickness contours (above precip)
            if 'geopotential_heights' in data_dict and data_dict['geopotential_heights']:
                self._render_thickness_layer(data_dict['geopotential_heights'])
            else:
                logger.warning("Geopotential heights not available, skipping thickness layer")
            
            # Layer 3: MSLP contours (above precip)
            if 'mslp' in data_dict and data_dict['mslp'] is not None:
                self._render_mslp_layer(data_dict['mslp'])
            else:
                logger.error("MSLP data not available - required for synoptic chart")
            
            # Layer 4: Surface features (zorder=10)
            if 'mslp' in data_dict and data_dict['mslp'] is not None:
                self._render_surface_features_layer(data_dict['mslp'])
            else:
                logger.warning("Cannot render surface features without MSLP data")
                
        except Exception as e:
            logger.error(f"Error during chart rendering: {e}", exc_info=True)
            raise
        
        # Add annotations if requested
        if annotate:
            try:
                self._annotations = annotate_chart(
                    self.fig, self.ax,
                    model_name, init_time, valid_time, lead_time,
                    self._rendered_layers
                )
                logger.info("Chart annotations added")
            except Exception as e:
                logger.warning(f"Failed to add annotations: {e}", exc_info=True)
        
        logger.info("Synoptic chart rendering complete")
        
        return self.fig, self.ax
    
    def _render_thickness_layer(self, heights_dict: Dict[str, Any]) -> None:
        """
        Render 1000-500mb thickness layer.
        
        Args:
            heights_dict: Dictionary with keys '500mb' and '1000mb' containing
                         geopotential height DataArrays
        """
        logger.info("Processing thickness layer")
        
        try:
            # Extract height arrays
            height_500mb = heights_dict.get('500mb')
            height_1000mb = heights_dict.get('1000mb')
            
            if height_500mb is None or height_1000mb is None:
                logger.warning("Missing height data for thickness calculation")
                return
            
            # Calculate thickness
            thickness_data = calculate_thickness(height_500mb, height_1000mb)
            
            if thickness_data is None:
                logger.warning("Thickness calculation returned None")
                return
            
            # Render thickness
            contour_set, paint_class = render_thickness(
                self.ax,
                thickness_data,
                self.config
            )
            
            # Store for potential colorbar creation
            self._rendered_layers['thickness'] = (contour_set, paint_class)
            
            logger.info("Thickness layer rendered successfully")
            
        except Exception as e:
            logger.error(f"Failed to render thickness layer: {e}", exc_info=True)
    
    def _render_precipitation_layers(
        self,
        precip_rate_dataset: Any,
        precip_categorical_dict: Optional[Dict[str, Any]]
    ) -> None:
        """
        Render precipitation layers for each precipitation type.
        
        Args:
            precip_rate_dataset: xarray Dataset with precipitation rate
            precip_categorical_dict: Dictionary with keys 'rain', 'snow', 'frzr', 'sleet'
                                    containing categorical masks (optional)
        """
        logger.info("Processing precipitation layers")
        
        try:
            # Extract precipitation rate variable
            # Try common variable names
            precip_rate_var = None
            for var_name in ['PRATE', 'prate', 'tp', 'precip_rate']:
                if hasattr(precip_rate_dataset, 'data_vars') and var_name in precip_rate_dataset.data_vars:
                    precip_rate_var = precip_rate_dataset[var_name]
                    break
                elif hasattr(precip_rate_dataset, 'name') and precip_rate_dataset.name == var_name:
                    precip_rate_var = precip_rate_dataset
                    break
            
            if precip_rate_var is None:
                # Try first data variable if specific names not found
                if hasattr(precip_rate_dataset, 'data_vars'):
                    data_vars = list(precip_rate_dataset.data_vars)
                    if data_vars:
                        precip_rate_var = precip_rate_dataset[data_vars[0]]
                        logger.debug(f"Using first data variable: {data_vars[0]}")
                else:
                    precip_rate_var = precip_rate_dataset
            
            if precip_rate_var is None:
                logger.warning("Could not extract precipitation rate variable")
                return
            
            # Convert to inches/hour
            precip_inches = convert_precip_rate_to_inches_per_hour(precip_rate_var)

            # If we don't have categorical precip-type flags (e.g., ECMWF),
            # render total precip once using the rain colormap (styling choice)
            # and avoid mislabeling as a specific precip type.
            if not precip_categorical_dict:
                try:
                    masked_total = mask_trace_precipitation(
                        precip_inches,
                        trace_threshold=self.config.trace_threshold,
                        categorical_mask=None,
                    )
                    valid_total = masked_total.values[~np.isnan(masked_total.values)]
                    if valid_total.size > 0 and valid_total.max() > 0:
                        contour_set, paint_class = render_precipitation(
                            self.ax,
                            masked_total,
                            'rain',
                            self.config
                        )
                        if contour_set is not None:
                            self._rendered_layers['precip_rain'] = (contour_set, paint_class)
                            logger.info(
                                "Rendered total precipitation using rain colormap "
                                "(no categorical masks available)"
                            )
                except Exception as e:
                    logger.warning(f"Failed to render total precipitation: {e}")
                return

            # Optionally expand categorical masks to fill small boundary gaps.
            expanded_masks: Dict[str, Any] = {}
            assigned = None
            dilation_iters = int(getattr(self.config, "categorical_mask_dilation", 0) or 0)
            if precip_categorical_dict:
                # Priority so mixed-ptype regions don't flip-flop when dilated.
                priority = ["frzr", "sleet", "snow", "rain"]
                for ptype in priority:
                    raw = precip_categorical_dict.get(ptype)
                    if raw is None:
                        continue
                    try:
                        expanded = expand_categorical_mask(raw, iterations=dilation_iters)
                    except Exception:
                        expanded = raw

                    mask_bool = (expanded > 0)
                    if assigned is None:
                        exclusive = mask_bool
                        assigned = exclusive
                    else:
                        exclusive = mask_bool & (~assigned)
                        assigned = assigned | exclusive

                    expanded_masks[ptype] = exclusive.astype("int8")

            # If categorical precip flags are present, render an "unclassified" layer
            # anywhere precip exists but none of the categorical types are flagged.
            try:
                if precip_categorical_dict:
                    any_flag = assigned
                    if any_flag is None:
                        # Fall back to raw masks if for some reason exclusivity wasn't built.
                        for ptype in ['rain', 'snow', 'frzr', 'sleet']:
                            mask = precip_categorical_dict.get(ptype)
                            if mask is None:
                                continue
                            mask_bool = (mask > 0)
                            any_flag = mask_bool if any_flag is None else (any_flag | mask_bool)

                    if any_flag is not None:
                        # Unclassified where precip rate exists but no type flag exists.
                        unclassified_mask = (~any_flag) & (precip_inches > 0)
                        unclassified = mask_trace_precipitation(
                            precip_inches,
                            trace_threshold=self.config.trace_threshold,
                            categorical_mask=unclassified_mask.astype('int8')
                        )
                        valid_unclassified = unclassified.values[~np.isnan(unclassified.values)]
                        if valid_unclassified.size > 0 and valid_unclassified.max() > 0:
                            contour_set, paint_class = render_precipitation(
                                self.ax,
                                unclassified,
                                'unclassified',
                                self.config
                            )
                            if contour_set is not None:
                                self._rendered_layers['precip_unclassified'] = (contour_set, paint_class)
            except Exception as e:
                logger.debug(f"Skipping unclassified precip layer: {e}")
            
            rendered_count = 0
            
            # Render each precipitation type
            precip_types_to_render = ['rain', 'snow', 'frzr', 'sleet']
            
            for precip_type in precip_types_to_render:
                try:
                    # Get categorical mask if available
                    categorical_mask = None
                    if precip_categorical_dict and precip_type in precip_categorical_dict:
                        categorical_mask = expanded_masks.get(precip_type, precip_categorical_dict[precip_type])
                        logger.debug(f"Using categorical mask for {precip_type}")
                    
                    # Apply trace masking
                    trace_threshold = (
                        self.config.snow_trace_threshold
                        if precip_type == 'snow'
                        else self.config.trace_threshold
                    )
                    masked_precip = mask_trace_precipitation(
                        precip_inches,
                        trace_threshold=trace_threshold,
                        categorical_mask=categorical_mask
                    )
                    
                    # Check if there's any data to plot
                    valid_data = masked_precip.values[~np.isnan(masked_precip.values)]
                    if valid_data.size == 0 or valid_data.max() == 0:
                        logger.debug(f"No significant {precip_type} to render")
                        continue
                    
                    # Render precipitation
                    contour_set, paint_class = render_precipitation(
                        self.ax,
                        masked_precip,
                        precip_type,
                        self.config
                    )
                    
                    if contour_set is not None:
                        # Store for potential colorbar creation
                        self._rendered_layers[f'precip_{precip_type}'] = (contour_set, paint_class)
                        rendered_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to render {precip_type}: {e}")
                    continue
            
            logger.info(f"Rendered {rendered_count} precipitation type(s)")
            
        except Exception as e:
            logger.error(f"Failed to render precipitation layers: {e}", exc_info=True)

    def _render_precipitation_accumulation_layer(self, precip_accum_dataset: Any) -> None:
        """Render accumulated precipitation (init -> lead) as a single filled-contour layer."""

        logger.info("Processing accumulated precipitation layer")

        try:
            da = None
            if hasattr(precip_accum_dataset, "data_vars"):
                for name in ["precip_accum", "APCP", "apcp", "tp"]:
                    if name in precip_accum_dataset.data_vars:
                        da = precip_accum_dataset[name]
                        break
                if da is None:
                    data_vars = list(precip_accum_dataset.data_vars)
                    if data_vars:
                        da = precip_accum_dataset[data_vars[0]]
            else:
                da = precip_accum_dataset

            if da is None:
                logger.warning("Could not extract accumulated precipitation variable")
                return

            # Apply trace masking (threshold is in inches of accumulation).
            thr = float(getattr(self.config, "accum_trace_threshold", 0.0) or 0.0)
            masked = da.where(da >= thr) if thr > 0 else da

            valid = masked.values[~np.isnan(masked.values)]
            if valid.size == 0 or valid.max() == 0:
                logger.debug("No significant accumulated precipitation to render")
                return

            contour_set, paint_class = render_precipitation_accumulation(
                self.ax,
                masked,
                self.config,
            )

            if contour_set is not None:
                self._rendered_layers["precip_accum"] = (contour_set, paint_class)
                logger.info("Accumulated precipitation rendered successfully")

        except Exception as e:
            logger.error(f"Failed to render accumulated precipitation layer: {e}", exc_info=True)
    
    def _render_mslp_layer(self, mslp_dataset: Any) -> None:
        """
        Render MSLP contour lines.
        
        Args:
            mslp_dataset: xarray Dataset with MSLP data
        """
        logger.info("Processing MSLP layer")
        
        try:
            # Extract MSLP variable
            # Try common variable names
            mslp_var = None
            for var_name in ['PRMSL', 'prmsl', 'msl', 'pressure', 'slp']:
                if hasattr(mslp_dataset, 'data_vars') and var_name in mslp_dataset.data_vars:
                    mslp_var = mslp_dataset[var_name]
                    break
                elif hasattr(mslp_dataset, 'name') and mslp_dataset.name == var_name:
                    mslp_var = mslp_dataset
                    break
            
            if mslp_var is None:
                # Try first data variable
                if hasattr(mslp_dataset, 'data_vars'):
                    data_vars = list(mslp_dataset.data_vars)
                    if data_vars:
                        mslp_var = mslp_dataset[data_vars[0]]
                        logger.debug(f"Using first data variable: {data_vars[0]}")
                else:
                    mslp_var = mslp_dataset
            
            if mslp_var is None:
                logger.error("Could not extract MSLP variable")
                return
            
            # Smooth MSLP
            smoothed_mslp = smooth_mslp(mslp_var, sigma=2.0)
            
            # Render contours
            contours = render_mslp_contours(self.ax, smoothed_mslp, self.config)
            
            # Store for reference
            self._rendered_layers['mslp'] = contours
            
            logger.info("MSLP layer rendered successfully")
            
        except Exception as e:
            logger.error(f"Failed to render MSLP layer: {e}", exc_info=True)
    
    def _render_surface_features_layer(self, mslp_dataset: Any) -> None:
        """
        Render surface pressure features (highs and lows).
        
        Args:
            mslp_dataset: xarray Dataset with MSLP data
        """
        logger.info("Processing surface features layer")
        
        try:
            # Extract and smooth MSLP (same as _render_mslp_layer)
            mslp_var = None
            for var_name in ['PRMSL', 'prmsl', 'msl', 'pressure', 'slp']:
                if hasattr(mslp_dataset, 'data_vars') and var_name in mslp_dataset.data_vars:
                    mslp_var = mslp_dataset[var_name]
                    break
                elif hasattr(mslp_dataset, 'name') and mslp_dataset.name == var_name:
                    mslp_var = mslp_dataset
                    break
            
            if mslp_var is None:
                if hasattr(mslp_dataset, 'data_vars'):
                    data_vars = list(mslp_dataset.data_vars)
                    if data_vars:
                        mslp_var = mslp_dataset[data_vars[0]]
                else:
                    mslp_var = mslp_dataset
            
            if mslp_var is None:
                logger.error("Could not extract MSLP variable for feature detection")
                return
            
            # Smooth MSLP
            smoothed_mslp = smooth_mslp(mslp_var, sigma=2.0)
            
            # Detect features
            features = detect_surface_features(
                smoothed_mslp,
                min_pressure_difference=4.0,
                search_radius=10
            )
            
            # Filter nearby features
            filtered_highs = filter_nearby_features(
                features['highs'],
                min_distance_km=500.0,
                feature_type='high'
            )
            
            filtered_lows = filter_nearby_features(
                features['lows'],
                min_distance_km=500.0,
                feature_type='low'
            )

            # Remove features outside current plot extent to avoid out-of-bounds
            # annotations affecting saved figure framing.
            try:
                west, east, south, north = self.ax.get_extent(crs=ccrs.PlateCarree())
                margin = 0.25  # degrees

                def _norm_lon(lon: float) -> float:
                    try:
                        lon_f = float(lon)
                    except Exception:
                        return lon
                    return ((lon_f + 180.0) % 360.0) - 180.0

                def _in_bounds(lat: float, lon: float) -> bool:
                    lon = _norm_lon(lon)
                    if west <= east:
                        lon_ok = (west - margin) <= lon <= (east + margin)
                    else:
                        # Dateline crossing: accept if in either interval.
                        lon_ok = lon >= (west - margin) or lon <= (east + margin)
                    lat_ok = (south - margin) <= lat <= (north + margin)
                    return lon_ok and lat_ok

                filtered_highs = [(lat, _norm_lon(lon), p) for (lat, lon, p) in filtered_highs if _in_bounds(lat, lon)]
                filtered_lows = [(lat, _norm_lon(lon), p) for (lat, lon, p) in filtered_lows if _in_bounds(lat, lon)]
            except Exception:
                pass
            
            # Create filtered features dict
            filtered_features = {
                'highs': filtered_highs,
                'lows': filtered_lows
            }
            
            # Render features
            render_surface_features(self.ax, filtered_features)
            
            # Store for reference
            self._rendered_layers['features'] = filtered_features
            
            logger.info(
                f"Surface features rendered: {len(filtered_highs)} highs, "
                f"{len(filtered_lows)} lows"
            )
            
        except Exception as e:
            logger.error(f"Failed to render surface features layer: {e}", exc_info=True)
    
    def get_rendered_layers(self) -> Dict[str, Any]:
        """
        Get dictionary of rendered layers for external access.
        
        Returns:
            Copy of _rendered_layers dict containing ContourSets and paint classes
            for colorbar creation and other post-processing
            
        Example:
            >>> chart = SynopticChart("CONUS")
            >>> fig, ax = chart.render_chart(data, "GFS", init_time, valid_time, 24)
            >>> layers = chart.get_rendered_layers()
            >>> if 'precip_rain' in layers:
            ...     contour_set, paint = layers['precip_rain']
            ...     # Create colorbar
        """
        return self._rendered_layers.copy()
    
    def get_annotations(self) -> Optional[Dict[str, Any]]:
        """
        Get dictionary of annotation objects for external access.
        
        Returns:
            Dictionary with keys 'title', 'model_info', 'colorbars' containing
            annotation objects, or None if annotations were not created
            
        Example:
            >>> chart = SynopticChart("CONUS")
            >>> fig, ax = chart.render_chart(data, "GFS", init_time, valid_time, 24)
            >>> annotations = chart.get_annotations()
            >>> if annotations:
            ...     print(f"Created {len(annotations['colorbars'])} colorbars")
        """
        return self._annotations
    
    def save_chart(
        self,
        output_path: str,
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight',
        pad_inches: float = 0.001
    ) -> str:
        """
        Save rendered chart to file.
        
        Args:
            output_path: Path or string for output file
            dpi: Optional DPI override (uses config.default_dpi if None)
            bbox_inches: Bbox setting for savefig (default: 'tight')
            pad_inches: Padding (inches) around tight bbox (default: 0.05)
            
        Returns:
            Path to saved file
            
        Raises:
            ValueError: If chart has not been rendered yet
            
        Example:
            >>> chart.save_chart("synoptic_chart.png", dpi=300)
            'synoptic_chart.png'
        """
        if self.fig is None:
            raise ValueError("Chart has not been rendered yet. Call render_chart() first.")
        
        # Use config DPI if not specified
        if dpi is None:
            dpi = self.config.default_dpi
        
        logger.info(f"Saving chart to {output_path} (dpi={dpi})")
        
        # Save figure
        self.fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        
        # Log file size if possible
        try:
            import os
            file_size = os.path.getsize(output_path)
            logger.info(f"Chart saved: {output_path} ({file_size / 1024:.1f} KB)")
        except Exception:
            logger.info(f"Chart saved: {output_path}")
        
        return output_path
