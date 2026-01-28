"""
Custom exceptions for SynopticCharts package.

This module defines exception classes for better error handling and messaging
across the package, particularly in the API, batch processing, and video
generation workflows.
"""


class SynopticChartsError(Exception):
    """Base exception class for all SynopticCharts errors."""
    pass


class DataFetchError(SynopticChartsError):
    """
    Raised when model data cannot be downloaded.
    
    This typically occurs due to network issues, invalid model parameters,
    data unavailability, or Herbie-related errors.
    """
    pass


class RenderError(SynopticChartsError):
    """
    Raised when chart rendering fails.
    
    This can occur due to invalid data, coordinate system issues, or
    matplotlib/cartopy errors during the rendering process.
    """
    pass


class VideoCreationError(SynopticChartsError):
    """
    Raised when video encoding fails.
    
    This typically occurs when ffmpeg is unavailable, frames are missing,
    or encoding parameters are invalid.
    """
    pass


class InvalidParameterError(SynopticChartsError):
    """
    Raised for invalid user inputs.
    
    This exception is used for parameter validation failures such as
    invalid model names, out-of-range lead times, or incompatible
    configuration options.
    """
    pass
