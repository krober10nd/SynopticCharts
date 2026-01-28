"""
Main API module for SynopticCharts package.

This module provides simplified user-facing functions that abstract away the
complexity of initializing downloaders and renderers. The primary function
`create_chart()` handles the complete workflow from data acquisition to
chart rendering in a single call.

Example:
    >>> from synoptic_charts import create_chart
    >>> from datetime import datetime
    >>> 
    >>> # Create a single chart
    >>> create_chart(
    ...     model="GFS",
    ...     forecast_cycle=datetime(2024, 1, 15, 0),
    ...     lead_time=24,
    ...     output_path="chart_f024.png"
    ... )
    
    >>> # Interactive use (returns figure and axes)
    >>> fig, ax = create_chart(
    ...     model="GFS",
    ...     forecast_cycle=datetime(2024, 1, 15, 0),
    ...     lead_time=24
    ... )
    >>> plt.show()
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt

from .config import Config
from .constants import MODELS
from .data import ModelDownloader
from .rendering import SynopticChart
from .exceptions import DataFetchError, RenderError, InvalidParameterError

logger = logging.getLogger(__name__)


def get_available_lead_times(model: str) -> List[int]:
    """
    Get list of valid forecast hours for a model.
    
    Args:
        model: Model name (e.g., "GFS", "ECMWF")
        
    Returns:
        List of valid forecast hours
        
    Raises:
        InvalidParameterError: If model name is not recognized
        
    Example:
        >>> lead_times = get_available_lead_times("GFS")
        >>> print(f"GFS has {len(lead_times)} forecast hours")
    """
    if model not in MODELS:
        available = ", ".join(MODELS.keys())
        raise InvalidParameterError(
            f"Unknown model '{model}'. Available models: {available}"
        )
    
    forecast_hours = MODELS[model]["forecast_hours"]
    logger.debug(f"Model {model} has {len(forecast_hours)} forecast hours")
    
    return forecast_hours


def create_chart(
    model: str,
    forecast_cycle: datetime,
    lead_time: int,
    region: str = "CONUS",
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None
) -> Union[str, Tuple[plt.Figure, plt.Axes]]:
    """
    Create a synoptic chart from model data.
    
    This is the primary API function that handles the complete workflow:
    1. Initialize ModelDownloader with parameters
    2. Fetch all required meteorological data
    3. Initialize SynopticChart renderer
    4. Render chart with data
    5. Save to file or return figure/axes for interactive use
    
    Args:
        model: Model name (e.g., "GFS", "ECMWF")
        forecast_cycle: Model initialization datetime
        lead_time: Forecast hour (e.g., 24 for 24-hour forecast)
        region: Region name from REGIONS dictionary (default: "CONUS")
        output_path: Output file path; if None, returns (fig, ax) for interactive use
        config: Optional Config object; if None, uses default configuration
        
    Returns:
        If output_path provided: path to saved chart file
        If output_path is None: tuple of (figure, axes) for interactive use
        
    Raises:
        InvalidParameterError: If model or region is invalid
        DataFetchError: If data download fails
        RenderError: If chart rendering fails
        
    Example:
        >>> # Save to file
        >>> path = create_chart(
        ...     model="GFS",
        ...     forecast_cycle=datetime(2024, 1, 15, 0),
        ...     lead_time=24,
        ...     output_path="chart_f024.png"
        ... )
        >>> print(f"Chart saved to {path}")
        
        >>> # Interactive use
        >>> fig, ax = create_chart(
        ...     model="GFS",
        ...     forecast_cycle=datetime(2024, 1, 15, 0),
        ...     lead_time=48
        ... )
        >>> plt.show()
    """
    logger.info(
        f"Creating chart: {model} init={forecast_cycle.strftime('%Y%m%d %HZ')} "
        f"f{lead_time:03d} region={region}"
    )
    
    # Validate model
    if model not in MODELS:
        available = ", ".join(MODELS.keys())
        raise InvalidParameterError(
            f"Unknown model '{model}'. Available models: {available}"
        )
    
    # Validate lead time
    available_lead_times = get_available_lead_times(model)
    if lead_time not in available_lead_times:
        raise InvalidParameterError(
            f"Invalid lead_time {lead_time} for {model}. "
            f"Valid range: {min(available_lead_times)}-{max(available_lead_times)}"
        )
    
    # Use default config if not provided
    if config is None:
        config = Config()
        logger.debug("Using default configuration")
    
    # Calculate valid time
    valid_time = forecast_cycle + timedelta(hours=lead_time)
    
    # Step 1: Download data
    logger.info("Initializing data downloader")
    try:
        downloader = ModelDownloader(
            model_name=model,
            init_time=forecast_cycle,
            config=config,
            forecast_hour=lead_time
        )
    except Exception as e:
        raise InvalidParameterError(f"Failed to initialize downloader: {e}") from e
    
    logger.info(f"Fetching data for {model} F{lead_time:03d}")
    try:
        data_dict = downloader.fetch_all_data()
    except Exception as e:
        raise DataFetchError(
            f"Failed to fetch data for {model} "
            f"init={forecast_cycle.strftime('%Y%m%d %HZ')} f{lead_time:03d}: {e}"
        ) from e
    
    if not data_dict or 'mslp' not in data_dict:
        raise DataFetchError(
            f"No data retrieved for {model} "
            f"init={forecast_cycle.strftime('%Y%m%d %HZ')} f{lead_time:03d}"
        )
    
    logger.info("Data fetch complete")
    
    # Step 2: Render chart
    logger.info(f"Initializing chart renderer for region '{region}'")
    try:
        chart = SynopticChart(region=region, config=config)
    except Exception as e:
        raise InvalidParameterError(f"Failed to initialize chart renderer: {e}") from e
    
    logger.info("Rendering chart")
    try:
        fig, ax = chart.render_chart(
            data_dict=data_dict,
            model_name=model,
            init_time=forecast_cycle,
            valid_time=valid_time,
            lead_time=lead_time,
            annotate=True
        )
    except Exception as e:
        raise RenderError(f"Failed to render chart: {e}") from e
    
    logger.info("Chart rendering complete")
    
    # Step 3: Save or return
    if output_path is not None:
        output_path = Path(output_path)
        logger.info(f"Saving chart to {output_path}")
        
        try:
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            saved_path = chart.save_chart(str(output_path))
            logger.info(f"Chart saved successfully to {saved_path}")

            # In batch/video workflows, ensure the figure is closed after save
            # to avoid memory growth and pyplot state bleed between frames.
            try:
                plt.close(fig)
            except Exception:
                pass
            
            return saved_path
            
        except Exception as e:
            raise RenderError(f"Failed to save chart to {output_path}: {e}") from e
    else:
        logger.info("Returning figure and axes for interactive use")
        return fig, ax


def create_chart_from_data(
    data_dict: Dict[str, Any],
    model_name: str,
    init_time: datetime,
    valid_time: datetime,
    lead_time: int,
    region: str = "CONUS",
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None
) -> Union[str, Tuple[plt.Figure, plt.Axes]]:
    """
    Create a synoptic chart from pre-fetched data.
    
    Useful when you have already downloaded data and want to create multiple
    charts with different settings or regions without re-downloading.
    
    Args:
        data_dict: Dictionary from ModelDownloader.fetch_all_data() with keys:
                  'mslp', 'precip_rate', 'precip_categorical', 'geopotential_heights'
        model_name: Model name for annotation (e.g., "GFS", "ECMWF")
        init_time: Model initialization datetime
        valid_time: Valid datetime for forecast
        lead_time: Forecast hour for annotation
        region: Region name from REGIONS dictionary (default: "CONUS")
        output_path: Output file path; if None, returns (fig, ax)
        config: Optional Config object
        
    Returns:
        If output_path provided: path to saved chart file
        If output_path is None: tuple of (figure, axes)
        
    Raises:
        InvalidParameterError: If region is invalid or data_dict is incomplete
        RenderError: If chart rendering or saving fails
        
    Example:
        >>> # Download data once
        >>> downloader = ModelDownloader("GFS", datetime(2024, 1, 15, 0), Config(), 24)
        >>> data = downloader.fetch_all_data()
        >>> 
        >>> # Create multiple charts from same data
        >>> create_chart_from_data(data, "GFS", init, valid, 24, output_path="chart1.png")
        >>> create_chart_from_data(data, "GFS", init, valid, 24, output_path="chart2.png")
    """
    logger.info(
        f"Creating chart from pre-fetched data: {model_name} "
        f"f{lead_time:03d} region={region}"
    )
    
    # Validate data
    if not data_dict:
        raise InvalidParameterError("data_dict cannot be empty")
    
    if 'mslp' not in data_dict:
        raise InvalidParameterError("data_dict must contain 'mslp' data")
    
    # Use default config if not provided
    if config is None:
        config = Config()
    
    # Initialize renderer
    logger.info(f"Initializing chart renderer for region '{region}'")
    try:
        chart = SynopticChart(region=region, config=config)
    except Exception as e:
        raise InvalidParameterError(f"Failed to initialize chart renderer: {e}") from e
    
    # Render chart
    logger.info("Rendering chart from provided data")
    try:
        fig, ax = chart.render_chart(
            data_dict=data_dict,
            model_name=model_name,
            init_time=init_time,
            valid_time=valid_time,
            lead_time=lead_time,
            annotate=True
        )
    except Exception as e:
        raise RenderError(f"Failed to render chart: {e}") from e
    
    logger.info("Chart rendering complete")
    
    # Save or return
    if output_path is not None:
        output_path = Path(output_path)
        logger.info(f"Saving chart to {output_path}")
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path = chart.save_chart(str(output_path))
            try:
                plt.close(fig)
            except Exception:
                pass
            logger.info(f"Chart saved successfully to {saved_path}")
            return saved_path
        except Exception as e:
            raise RenderError(f"Failed to save chart to {output_path}: {e}") from e
    else:
        logger.info("Returning figure and axes for interactive use")
        return fig, ax
