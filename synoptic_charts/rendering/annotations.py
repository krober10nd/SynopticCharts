"""
Annotation module for synoptic chart text and colorbars.

This module provides functions for adding professional meteorological annotations
to synoptic charts, including titles, model information, and precipitation colorbars.
All annotations use consistent styling from constants and integrate seamlessly with
the SynopticChart rendering pipeline.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .layers import create_colorbar_from_paint
from ..constants import (
    PRECIP_TYPES,
    PRECIP_COLORBAR_TICKS,
    TITLE_POSITION,
    MODEL_INFO_POSITION,
    COLORBAR_BOTTOM,
    COLORBAR_HEIGHT,
    COLORBAR_SPACING,
    COLORBAR_LABEL_SIZE,
    TITLE_FONT_SIZE,
    ANNOTATION_FONT_SIZE
)

logger = logging.getLogger("synoptic_charts.rendering.annotations")


def _is_dark_figure(fig: plt.Figure) -> bool:
    try:
        r, g, b, _a = fig.get_facecolor()
    except Exception:
        return False
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance < 0.35


def _annotation_color(fig: plt.Figure) -> str:
    return "white" if _is_dark_figure(fig) else "black"


def add_title_annotation(
    fig: plt.Figure,
    lead_time: int,
    valid_time: datetime,
    *,
    precip_mode: str = "rate"
) -> Any:
    """
    Add top-left title annotation with chart description and forecast info.
    
    Displays precipitation rate and thickness description on first line,
    followed by forecast hour and valid time on second line. Text is placed
    in figure coordinates above the map axes to avoid overlapping data.
    
    Args:
        fig: Matplotlib figure containing the chart
        lead_time: Forecast hour (e.g., 24)
        valid_time: Valid datetime for forecast
        
    Returns:
        Text artist object
        
    Example:
        >>> title = add_title_annotation(fig, 24, datetime(2024, 1, 16, 0))
    """
    logger.info(f"Adding title annotation for F{lead_time:03d}")
    
    if fig is None:
        logger.error("Cannot add title annotation: fig is None")
        raise ValueError("fig cannot be None")
    
    if not isinstance(valid_time, datetime):
        logger.error("valid_time must be a datetime object")
        raise TypeError("valid_time must be a datetime object")
    
    # Format title text with two lines
    mode = str(precip_mode or "rate").strip().lower()
    if mode == "accumulated":
        title_line1 = "Accumulated precipitation (mm), 1000-500 mb thickness (dam)"
    else:
        title_line1 = "Precipitation rate (in per hr), 1000-500 mb thickness (dam)"
    title_line2 = f"F{lead_time:03d} Valid: {valid_time.strftime('%Y-%m-%d %HZ')}"
    
    title_text = f"{title_line1}\n{title_line2}"
    
    # Add text annotation in figure coordinates (outside map axes)
    text_artist = fig.text(
        TITLE_POSITION[0], TITLE_POSITION[1],
        title_text,
        transform=fig.transFigure,
        ha='left',
        va='top',
        fontsize=ANNOTATION_FONT_SIZE,
        weight='bold',
        family='sans-serif',
        color=_annotation_color(fig)
    )
    
    logger.debug(f"Title annotation added at figure position {TITLE_POSITION}")
    
    return text_artist


def add_model_info_annotation(
    fig: plt.Figure,
    model_name: str,
    init_time: datetime
) -> Any:
    """
    Add top-right model initialization info annotation.
    
    Displays model name and initialization time in upper right corner.
    Text is placed in figure coordinates above the map axes to avoid
    overlapping data.
    
    Args:
        fig: Matplotlib figure containing the chart
        model_name: Model name (e.g., "GFS", "ECMWF")
        init_time: Model initialization datetime
        
    Returns:
        Text artist object
        
    Example:
        >>> info = add_model_info_annotation(fig, "GFS", datetime(2024, 1, 15, 0))
    """
    logger.info(f"Adding model info annotation for {model_name}")
    
    if fig is None:
        logger.error("Cannot add model info annotation: fig is None")
        raise ValueError("fig cannot be None")
    
    if not isinstance(init_time, datetime):
        logger.error("init_time must be a datetime object")
        raise TypeError("init_time must be a datetime object")
    
    # Format model info text
    info_text = f"{model_name}\nInit: {init_time.strftime('%Y-%m-%d %HZ')}"
    
    # Add text annotation in figure coordinates (outside map axes)
    text_artist = fig.text(
        MODEL_INFO_POSITION[0], MODEL_INFO_POSITION[1],
        info_text,
        transform=fig.transFigure,
        ha='right',
        va='top',
        fontsize=ANNOTATION_FONT_SIZE,
        family='sans-serif',
        color=_annotation_color(fig)
    )
    
    logger.debug(f"Model info annotation added at figure position {MODEL_INFO_POSITION}")
    
    return text_artist


def add_precipitation_colorbars(
    fig: plt.Figure,
    rendered_layers: Dict[str, Tuple[Any, Any]]
) -> List[Any]:
    """
    Create horizontal colorbars at bottom for each precipitation type.
    
    Examines rendered_layers dict for precipitation types (rain, snow, frzr, sleet)
    and creates a horizontal colorbar for each available type. Colorbars are
    distributed evenly across the bottom margin.
    
    Args:
        fig: Matplotlib figure containing the chart
        rendered_layers: Dictionary from SynopticChart.get_rendered_layers()
                        with keys like 'precip_rain', 'precip_snow', etc.
                        Values are tuples of (contour_set, paint_class)
        
    Returns:
        List of colorbar objects (may be empty if no precipitation data)
        
    Example:
        >>> layers = chart.get_rendered_layers()
        >>> colorbars = add_precipitation_colorbars(fig, layers)
        >>> print(f"Created {len(colorbars)} colorbars")
    """
    logger.info("Adding precipitation colorbars")
    
    if fig is None:
        logger.error("Cannot add colorbars: fig is None")
        raise ValueError("fig cannot be None")
    
    if not rendered_layers:
        logger.warning("No rendered layers provided, skipping colorbars")
        return []
    
    # Special case: accumulated precipitation uses a single NWSPrecipitation colorbar.
    if "precip_accum" in rendered_layers:
        contour_set, paint_class = rendered_layers.get("precip_accum", (None, None))
        if contour_set is None:
            return []

        dark_theme = _is_dark_figure(fig)
        cbar_text_color = "white" if dark_theme else "black"

        position = [0.05, COLORBAR_BOTTOM, 0.90, COLORBAR_HEIGHT]
        try:
            cbar = create_colorbar_from_paint(
                fig,
                contour_set,
                paint_class,
                position,
                label_override="Accumulated precipitation (mm)",
            )
            if cbar is not None:
                cbar.ax.tick_params(
                    labelsize=COLORBAR_LABEL_SIZE,
                    colors=cbar_text_color,
                    labelcolor=cbar_text_color,
                )
                try:
                    cbar.ax.xaxis.label.set_color(cbar_text_color)
                except Exception:
                    pass
                try:
                    cbar.ax.set_facecolor('none')
                except Exception:
                    pass
                return [cbar]
        except Exception as e:
            logger.error(f"Error creating accumulation colorbar: {e}", exc_info=True)
            return []

    # Determine which precipitation types are available
    precip_types = ['rain', 'snow', 'frzr', 'sleet']
    available_types = []
    
    for ptype in precip_types:
        layer_key = f'precip_{ptype}'
        if layer_key in rendered_layers:
            contour_set, paint_class = rendered_layers[layer_key]
            if contour_set is not None:
                available_types.append(ptype)
                logger.debug(f"Found {ptype} precipitation layer")
    
    if not available_types:
        logger.info("No precipitation layers to create colorbars for")
        return []
    
    # Calculate colorbar positions
    num_colorbars = len(available_types)
    logger.info(f"Creating {num_colorbars} precipitation colorbar(s)")
    
    # Calculate total width available (leaving margins)
    total_margin = 0.1  # 5% on each side
    available_width = 1.0 - total_margin
    
    # Calculate width per colorbar (accounting for spacing)
    total_spacing = COLORBAR_SPACING * (num_colorbars - 1)
    colorbar_width = (available_width - total_spacing) / num_colorbars
    
    dark_theme = _is_dark_figure(fig)
    cbar_text_color = "white" if dark_theme else "black"

    # Create colorbars
    colorbars = []
    for i, ptype in enumerate(available_types):
        # Calculate position
        left = 0.05 + i * (colorbar_width + COLORBAR_SPACING)
        position = [left, COLORBAR_BOTTOM, colorbar_width, COLORBAR_HEIGHT]
        
        # Get layer data
        layer_key = f'precip_{ptype}'
        contour_set, paint_class = rendered_layers[layer_key]
        
        # Get label from constants
        label = f"{PRECIP_TYPES[ptype]['name']} (in/hr)"
        
        # Create colorbar
        try:
            cbar = create_colorbar_from_paint(
                fig,
                contour_set,
                paint_class,
                position,
                label_override=label
            )
            
            if cbar is not None:
                # Apply label styling
                cbar.ax.tick_params(labelsize=COLORBAR_LABEL_SIZE, colors=cbar_text_color, labelcolor=cbar_text_color)
                try:
                    cbar.ax.xaxis.label.set_color(cbar_text_color)
                except Exception:
                    pass
                try:
                    cbar.ax.set_facecolor('none')
                except Exception:
                    pass
                try:
                    cbar.set_ticks(PRECIP_COLORBAR_TICKS)
                except Exception:
                    pass
                colorbars.append(cbar)
                logger.debug(f"Created {ptype} colorbar at position {position}")
            else:
                logger.warning(f"Failed to create colorbar for {ptype}")
                
        except Exception as e:
            logger.error(f"Error creating colorbar for {ptype}: {e}", exc_info=True)
    
    logger.info(f"Successfully created {len(colorbars)} colorbar(s)")
    
    return colorbars


def annotate_chart(
    fig: plt.Figure,
    ax: plt.Axes,
    model_name: str,
    init_time: datetime,
    valid_time: datetime,
    lead_time: int,
    rendered_layers: Dict[str, Tuple[Any, Any]]
) -> Dict[str, Any]:
    """
    Add all annotations to synoptic chart (convenience function).
    
    Calls add_title_annotation(), add_model_info_annotation(), and
    add_precipitation_colorbars() in sequence to fully annotate a chart.
    
    Args:
        fig: Matplotlib figure containing the chart
        ax: Matplotlib axes with synoptic chart
        model_name: Model name (e.g., "GFS", "ECMWF")
        init_time: Model initialization datetime
        valid_time: Valid datetime for forecast
        lead_time: Forecast hour (e.g., 24)
        rendered_layers: Dictionary from SynopticChart.get_rendered_layers()
        
    Returns:
        Dictionary with keys:
            - 'title': Title text artist
            - 'model_info': Model info text artist
            - 'colorbars': List of colorbar objects
            
    Example:
        >>> annotations = annotate_chart(
        ...     fig, ax, "GFS",
        ...     datetime(2024, 1, 15, 0),
        ...     datetime(2024, 1, 16, 0),
        ...     24,
        ...     chart.get_rendered_layers()
        ... )
        >>> print(f"Created {len(annotations['colorbars'])} colorbars")
    """
    logger.info(f"Annotating chart: {model_name} F{lead_time:03d}")
    
    annotations = {}
    
    try:
        # Add title (in figure coordinates)
        precip_mode = "accumulated" if "precip_accum" in (rendered_layers or {}) else "rate"
        annotations['title'] = add_title_annotation(
            fig,
            lead_time,
            valid_time,
            precip_mode=precip_mode,
        )
        
        # Add model info (in figure coordinates)
        annotations['model_info'] = add_model_info_annotation(fig, model_name, init_time)
        
        # Add precipitation colorbars
        annotations['colorbars'] = add_precipitation_colorbars(fig, rendered_layers)
        
        logger.info("Chart annotation complete")
        
    except Exception as e:
        logger.error(f"Error during chart annotation: {e}", exc_info=True)
        raise
    
    return annotations
