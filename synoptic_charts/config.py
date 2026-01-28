"""
Configuration management for SynopticCharts package.

This module provides configuration options for chart generation including
figure sizing, DPI, cache directories, and meteorological thresholds.
"""

import json
import yaml
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for synoptic chart generation.
    
    Attributes:
        default_dpi: Resolution for output images (dots per inch).
        figure_width: Width of generated figures in inches.
        figure_height: Height of generated figures in inches.
        cache_dir: Directory for caching downloaded model data.
        output_dir: Directory for saving generated charts.
        trace_threshold: Minimum precipitation rate in inches/hr to display.
        snow_trace_threshold: Minimum snowfall rate in inches/hr to display.
        mslp_contour_interval: Spacing between MSLP contours in hPa.
        thickness_contour_interval: Spacing between thickness contours in decameters.
        show_grid_labels: Whether to draw lat/lon gridline labels.
        precip_interp_factor: Integer upscaling factor for precipitation plotting grid.
        precip_interp_method: Interpolation method passed to xarray (e.g., 'linear', 'nearest').
        precip_cmap_min: Lower bound in [0, 1) when sampling colormaps (higher = darker).
        background_color: Figure and map background color (any Matplotlib color spec).
        categorical_mask_dilation: Number of grid-cell dilation iterations to apply to
            categorical precip masks before masking/plotting.
        precip_mode: Precipitation visualization mode. "rate" plots precip rate
            (in/hr) and optional categorical precip types. "accumulated" plots
            init-to-lead accumulated precipitation (mm).
        accum_trace_threshold: Minimum accumulated precip in mm to display when
            precip_mode="accumulated".
    """
    
    default_dpi: int = 150
    figure_width: float = 16.0
    figure_height: float = 12.0
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".synoptic_charts" / "cache")
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    trace_threshold: float = 0.000
    snow_trace_threshold: float = 0.0000
    mslp_contour_interval: float = 4.0
    thickness_contour_interval: float = 15.0
    show_grid_labels: bool = False
    precip_interp_factor: int = 1
    precip_interp_method: str = "linear"
    precip_cmap_min: float = 0.0
    # Dark default so low precip/snow bins don't blend into the basemap.
    background_color: str = "#1f2328"
    categorical_mask_dilation: int = 0
    precip_mode: str = "rate"
    accum_trace_threshold: float = 0.0
    
    def __post_init__(self):
        """Convert string paths to Path objects if necessary."""
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
    
    @classmethod
    def load_from_file(cls, path: Path) -> "Config":
        """Load configuration from a YAML or JSON file.
        
        Args:
            path: Path to configuration file (.yaml, .yml, or .json).
            
        Returns:
            Config instance with loaded settings.
            
        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")
        
        # Convert path strings to Path objects
        if 'cache_dir' in data:
            data['cache_dir'] = Path(data['cache_dir'])
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
        
        return cls(**data)
    
    def save_to_file(self, path: Path) -> None:
        """Save configuration to a YAML or JSON file.
        
        Args:
            path: Path where configuration should be saved.
            
        Raises:
            ValueError: If file format is not supported.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary with string paths
        data = asdict(self)
        data['cache_dir'] = str(data['cache_dir'])
        data['output_dir'] = str(data['output_dir'])
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            elif path.suffix == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")
    
    def validate(self) -> bool:
        """Validate configuration parameters.
        
        Returns:
            True if configuration is valid.
            
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.default_dpi <= 0:
            raise ValueError("default_dpi must be positive")
        
        if self.figure_width <= 0 or self.figure_height <= 0:
            raise ValueError("Figure dimensions must be positive")
        
        if self.trace_threshold < 0:
            raise ValueError("trace_threshold must be non-negative")

        if self.snow_trace_threshold < 0:
            raise ValueError("snow_trace_threshold must be non-negative")
        
        if self.mslp_contour_interval <= 0:
            raise ValueError("mslp_contour_interval must be positive")
        
        if self.thickness_contour_interval <= 0:
            raise ValueError("thickness_contour_interval must be positive")

        if not isinstance(self.show_grid_labels, bool):
            raise ValueError("show_grid_labels must be a boolean")

        if not isinstance(self.precip_interp_factor, int) or self.precip_interp_factor < 1:
            raise ValueError("precip_interp_factor must be an integer >= 1")

        if not isinstance(self.precip_interp_method, str) or not self.precip_interp_method:
            raise ValueError("precip_interp_method must be a non-empty string")

        if not (0.0 <= float(self.precip_cmap_min) < 1.0):
            raise ValueError("precip_cmap_min must be in the range [0.0, 1.0)")

        if not isinstance(self.background_color, str) or not self.background_color:
            raise ValueError("background_color must be a non-empty string")

        if not isinstance(self.categorical_mask_dilation, int) or self.categorical_mask_dilation < 0:
            raise ValueError("categorical_mask_dilation must be an integer >= 0")

        if str(self.precip_mode).strip().lower() not in {"rate", "accumulated"}:
            raise ValueError("precip_mode must be one of: 'rate', 'accumulated'")

        if self.accum_trace_threshold < 0:
            raise ValueError("accum_trace_threshold must be non-negative")
        
        return True
    
    def ensure_directories(self) -> None:
        """Create cache and output directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """Get a Config instance with default settings.
    
    Returns:
        Config instance initialized with default values.
    """
    return Config()
