"""
Logging configuration for SynopticCharts package.

This module provides centralized logging setup with configurable verbosity
levels, formatters, and output destinations. It supports console and file
logging with proper hierarchy management.
"""

import logging
import os
import sys
from typing import Optional


# Default log format with timestamp and level
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def setup_logging(
    verbosity: int = 0,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for SynopticCharts package.
    
    Sets up logger hierarchy with appropriate levels and handlers:
    - Root logger at WARNING level
    - synoptic_charts.* loggers at INFO level by default
    - DEBUG level when verbosity > 0
    
    Args:
        verbosity: Verbosity level (0=INFO, 1=DEBUG, -1=WARNING, -2=ERROR)
        log_file: Optional path to log file for file output
        format_string: Optional custom format string for log messages
        
    Environment Variables:
        SYNOPTIC_CHARTS_LOG_LEVEL: Override log level (DEBUG, INFO, WARNING, ERROR)
        
    Example:
        >>> setup_logging(verbosity=1)  # Enable DEBUG logging
        >>> setup_logging(log_file="synoptic.log")  # Log to file
    """
    # Determine log level from verbosity
    if verbosity >= 1:
        level = logging.DEBUG
    elif verbosity == 0:
        level = logging.INFO
    elif verbosity == -1:
        level = logging.WARNING
    else:  # verbosity <= -2
        level = logging.ERROR
    
    # Check for environment variable override
    env_level = os.environ.get("SYNOPTIC_CHARTS_LOG_LEVEL", "").upper()
    if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level = getattr(logging, env_level)
    
    # Choose format
    if format_string is None:
        format_string = DEFAULT_FORMAT if verbosity >= 0 else SIMPLE_FORMAT
    
    # Configure root logger to WARNING to suppress external libraries
    logging.root.setLevel(logging.WARNING)
    
    # Get or create synoptic_charts logger
    logger = logging.getLogger("synoptic_charts")
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to root
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_formatter = logging.Formatter(DEFAULT_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to create log file {log_file}: {e}")
    
    logger.debug(f"Logging configured: level={logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
        
    Returns:
        Logger instance under synoptic_charts hierarchy
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


# Initialize default logging when module is imported
setup_logging()
