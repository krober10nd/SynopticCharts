# Configuration Guide

The SynopticCharts package provides extensive configuration options to customize chart appearance, behavior, and output quality. This guide covers all configuration parameters and how to use them.

## Table of Contents

- [Overview](#overview)
- [Configuration Files](#configuration-files)
- [Configuration Parameters](#configuration-parameters)
- [Region Configuration](#region-configuration)
- [Model Settings](#model-settings)
- [Styling Customization](#styling-customization)
- [Advanced Options](#advanced-options)

## Overview

Configuration can be provided in three ways:

1. **Default configuration**: Built-in sensible defaults
2. **Configuration files**: YAML or JSON files
3. **Programmatic**: Direct modification of `Config` objects

### Priority Order

When multiple configuration sources are present:

```
Programmatic overrides > Config file > Default values
```

## Configuration Files

### YAML Format

```yaml
# Chart output settings
default_dpi: 300
figure_width: 16.0
figure_height: 12.0
cache_dir: "~/.synoptic_charts_cache"

# Precipitation thresholds (inches)
trace_threshold: 0.01
light_threshold: 0.1
moderate_threshold: 0.25
heavy_threshold: 0.5

# Contour intervals
mslp_contour_interval: 4.0  # hPa
thickness_contour_interval: 6.0  # decameters
```

### JSON Format

```json
{
  "default_dpi": 300,
  "figure_width": 16.0,
  "figure_height": 12.0,
  "cache_dir": "~/.synoptic_charts_cache",
  "trace_threshold": 0.01,
  "light_threshold": 0.1,
  "moderate_threshold": 0.25,
  "heavy_threshold": 0.5,
  "mslp_contour_interval": 4.0,
  "thickness_contour_interval": 6.0
}
```

### Loading Configuration Files

```python
from synoptic_charts import Config

# Load from YAML
config = Config.load_from_file("config.yaml")

# Load from JSON
config = Config.load_from_file("config.json")

# Use in chart creation
from synoptic_charts import create_chart
from datetime import datetime

create_chart(
    model="GFS",
    forecast_cycle=datetime(2024, 1, 15, 0),
    lead_time=24,
    output_path="chart.png",
    config=config
)
```

### Saving Configuration Files

```python
from synoptic_charts import Config

# Create custom config
config = Config(
    default_dpi=300,
    figure_width=16.0,
    mslp_contour_interval=2.0
)

# Save to file
config.save_to_file("my_config.yaml")
```

## Configuration Parameters

### Output Settings

#### `default_dpi` (int)
Resolution for saved figures in dots per inch.

- **Default**: `150`
- **Range**: `72` (screen) to `600` (print)
- **Common values**:
  - `100`: Quick previews
  - `150`: Standard quality
  - `300`: Publication quality
  - `600`: High-resolution prints

**Impact**: Higher DPI increases file size and rendering time but improves quality.

```python
config = Config(default_dpi=300)  # High quality
```

#### `figure_width` (float)
Width of the figure in inches.

- **Default**: `16.0`
- **Range**: `8.0` to `24.0`
- **Common values**:
  - `12.0`: Compact, good for web
  - `16.0`: Standard, balanced
  - `20.0`: Large, detailed presentations

```python
config = Config(figure_width=20.0)
```

#### `figure_height` (float)
Height of the figure in inches.

- **Default**: `12.0`
- **Range**: `6.0` to `18.0`
- **Typical aspect ratios**:
  - `4:3` (16.0 × 12.0): Standard
  - `16:9` (16.0 × 9.0): Widescreen
  - `3:2` (15.0 × 10.0): Photography

```python
config = Config(figure_width=16.0, figure_height=9.0)  # 16:9 ratio
```

### Cache Settings

#### `cache_dir` (str)
Directory for caching downloaded model data.

- **Default**: `"~/.synoptic_charts_cache"`
- **Purpose**: Avoid re-downloading data for repeated operations
- **Size**: Can grow to several GB with regular use

```python
config = Config(cache_dir="/path/to/cache")
```

**Cache Management:**

```python
import os
import shutil
from synoptic_charts import Config

# Clear cache
config = Config()
cache_path = os.path.expanduser(config.cache_dir)
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
```

### Precipitation Thresholds

These control precipitation classification and shading.

#### `trace_threshold` (float)
Minimum precipitation for "trace" amounts (inches).

- **Default**: `0.01`
- **Range**: `0.005` to `0.02`

#### `light_threshold` (float)
Threshold for light precipitation (inches).

- **Default**: `0.1`
- **Range**: `0.05` to `0.15`

#### `moderate_threshold` (float)
Threshold for moderate precipitation (inches).

- **Default**: `0.25`
- **Range**: `0.2` to `0.35`

#### `heavy_threshold` (float)
Threshold for heavy precipitation (inches).

- **Default**: `0.5`
- **Range**: `0.4` to `0.75`

**Example:**

```python
# More sensitive precipitation classification
config = Config(
    trace_threshold=0.005,
    light_threshold=0.05,
    moderate_threshold=0.15,
    heavy_threshold=0.3
)
```

### Contour Settings

#### `mslp_contour_interval` (float)
Interval between MSLP contours in hPa (millibars).

- **Default**: `4.0`
- **Common values**:
  - `2.0`: Fine detail, complex patterns
  - `4.0`: Standard, clear patterns
  - `8.0`: Major features only

```python
config = Config(mslp_contour_interval=2.0)  # Fine detail
```

#### `thickness_contour_interval` (float)
Interval between thickness contours in decameters.

- **Default**: `6.0` (60 meters)
- **Common values**:
  - `3.0`: Very fine detail
  - `6.0`: Standard
  - `12.0`: Major features

```python
config = Config(thickness_contour_interval=3.0)  # Fine detail
```

**Meteorological Note**: The 540 dam thickness line is critical for rain/snow discrimination. Finer intervals help identify the rain/snow boundary more precisely.

## Region Configuration

Regions define the geographic extent and map projection for charts.

### Built-in Regions

The package includes the following pre-defined region:

- `CONUS`: Continental United States (default)
- `NORTHEAST`: Northeast US + Mid-Atlantic

### Using Built-in Regions

```python
from synoptic_charts import create_chart
from datetime import datetime

# Use default CONUS region
create_chart(
    model="GFS",
    forecast_cycle=datetime(2024, 1, 15, 0),
    lead_time=24,
    output_path="chart.png",
    region="CONUS"  # This is the default
)
```

### Custom Regions

Define custom regions with a dictionary containing `extent` and `projection` keys.

#### Region Structure

```python
custom_region = {
    "extent": [west_lon, east_lon, south_lat, north_lat],
    "projection": {
        "proj": "lcc",  # Lambert Conformal Conic
        "lat_0": central_latitude,
        "lon_0": central_longitude,
        "lat_1": standard_parallel_1,
        "lat_2": standard_parallel_2,
        "datum": "WGS84",
        "units": "m"
    }
}
```

#### Extent Format

The `extent` defines the geographic boundaries:

```python
extent = [west_longitude, east_longitude, south_latitude, north_latitude]
```

**Conventions:**
- Longitudes: Negative for Western Hemisphere (e.g., -120.0 for 120°W)
- Latitudes: Positive for Northern Hemisphere
- Order: West, East, South, North

**Example:**

```python
# Northeast US
extent = [-80.0, -66.0, 37.0, 47.0]  # 80°W to 66°W, 37°N to 47°N
```

#### Projection Parameters

**Lambert Conformal Conic (recommended for mid-latitudes):**

```python
projection = {
    "proj": "lcc",
    "lat_0": 42.0,    # Central latitude (center of region)
    "lon_0": -73.0,   # Central longitude (center of region)
    "lat_1": 39.0,    # First standard parallel (south)
    "lat_2": 45.0,    # Second standard parallel (north)
    "datum": "WGS84",
    "units": "m"
}
```

**Guidelines for selecting parameters:**

1. **Central Latitude (`lat_0`)**: Center of your region's latitude range
   ```python
   lat_0 = (south_lat + north_lat) / 2
   ```

2. **Central Longitude (`lon_0`)**: Center of your region's longitude range
   ```python
   lon_0 = (west_lon + east_lon) / 2
   ```

3. **Standard Parallels (`lat_1`, `lat_2`)**: Place at 1/6 and 5/6 of latitude range
   ```python
   lat_range = north_lat - south_lat
   lat_1 = south_lat + lat_range / 6  # Lower standard parallel
   lat_2 = north_lat - lat_range / 6  # Upper standard parallel
   ```

**Complete Example:**

```python
from synoptic_charts import create_chart, REGIONS
from datetime import datetime

# Southeast US region
southeast_extent = [-90.0, -75.0, 25.0, 37.0]

# Calculate projection parameters
south_lat, north_lat = 25.0, 37.0
west_lon, east_lon = -90.0, -75.0
lat_range = north_lat - south_lat

southeast_region = {
    "name": "Southeast US",
    "extent": southeast_extent,
    "projection_params": {
        "central_longitude": (west_lon + east_lon) / 2,    # -82.5
        "central_latitude": (south_lat + north_lat) / 2,  # 31.0
        "standard_parallels": (
            south_lat + lat_range / 6,     # 27.0
            north_lat - lat_range / 6      # 35.0
        )
    },
    "figure_size": (12, 10)  # width, height in inches
}

# Add to REGIONS dictionary for reuse
REGIONS["SOUTHEAST"] = southeast_region

# Now use it
create_chart(
    model="GFS",
    forecast_cycle=datetime(2024, 1, 15, 0),
    lead_time=24,
    output_path="southeast.png",
    region="SOUTHEAST"
)
```

#### Alternative Projections

**Mercator (low latitudes, small regions):**

```python
projection = {
    "proj": "merc",
    "lat_ts": 0.0,    # Latitude of true scale
    "lon_0": -80.0,   # Central longitude
    "datum": "WGS84",
    "units": "m"
}
```

**Stereographic (polar regions):**

```python
projection = {
    "proj": "stere",
    "lat_0": 90.0,    # 90 for North Pole, -90 for South Pole
    "lon_0": -150.0,  # Central longitude
    "lat_ts": 70.0,   # Latitude of true scale
    "datum": "WGS84",
    "units": "m"
}
```

### Regional Configuration Tips

1. **Adjust figure size to match region aspect ratio:**
   ```python
   lon_range = east_lon - west_lon
   lat_range = north_lat - south_lat
   aspect_ratio = lon_range / lat_range
   
   figure_height = 12.0
   figure_width = figure_height * aspect_ratio
   
   config = Config(figure_width=figure_width, figure_height=figure_height)
   ```

2. **Use appropriate contour intervals for region size:**
   - Small regions: Finer intervals (2 hPa MSLP, 3 dam thickness)
   - Large regions: Standard intervals (4 hPa MSLP, 6 dam thickness)

3. **Test your region definition:**
   ```python
   # Quick test with low DPI
   test_config = Config(default_dpi=100)
   create_chart(..., region=custom_region, config=test_config)
   ```

## Model Settings

Model settings are defined in `synoptic_charts/constants.py` but can be queried programmatically.

### Available Models

```python
from synoptic_charts.constants import MODELS

print(MODELS)
# {
#     "GFS": {"name": "Global Forecast System", "interval": 3, ...},
#     "ECMWF": {"name": "European Centre Model", "interval": 6, ...},
#     ...
# }
```

### Forecast Lead Times

Each model has specific forecast hours available:

```python
from synoptic_charts import get_available_lead_times

# Get valid lead times for GFS
lead_times = get_available_lead_times("GFS")
print(f"GFS forecasts available: 0-{max(lead_times)} hours")

# Check if specific lead time is available
if 72 in lead_times:
    print("72-hour forecast available")
```

### Model Characteristics

| Model | Max Forecast | Interval | Update Frequency |
|-------|-------------|----------|------------------|
| GFS   | 384 hours   | 3 hours  | Every 6 hours    |
| ECMWF | 240 hours   | 6 hours  | Twice daily      |

## Styling Customization

### Precipitation Type Styling

Precipitation types are defined in `synoptic_charts/constants.py`:

```python
PRECIP_TYPES = {
    "rain": {"color": "green", "hatch": None, "label": "Rain"},
    "freezing_rain": {"color": "magenta", "hatch": "///", "label": "Freezing Rain"},
    "ice_pellets": {"color": "purple", "hatch": "...", "label": "Ice Pellets"},
    "snow": {"color": "blue", "hatch": None, "label": "Snow"}
}
```

**Customization (advanced):**

```python
from synoptic_charts import constants

# Modify precipitation colors
constants.PRECIP_TYPES["rain"]["color"] = "darkgreen"
constants.PRECIP_TYPES["snow"]["color"] = "navy"
```

### Colormap and Level Customization

Precipitation shading uses matplotlib colormaps and discrete levels.

**Defined in constants.py:**

```python
# Colormap for precipitation shading
PRECIP_CMAP = "YlGnBu"  # Yellow-Green-Blue

# Precipitation levels (inches)
PRECIP_LEVELS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
```

**Customization:**

```python
from synoptic_charts import constants

# Use different colormap
constants.PRECIP_CMAP = "Blues"  # Blue shades only

# Finer precipitation levels
constants.PRECIP_LEVELS = [0.01, 0.03, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
```

**Available colormaps:** See [Matplotlib Colormap Reference](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

## Advanced Options

### Thickness Contour Styling

Thickness contours highlight the thermal structure of the atmosphere.

**Key contour: 540 dam (5400 meters)**
- Typically shown in bold red
- Represents the rain/snow transition zone
- Warm air (above 540 dam): Rain likely
- Cold air (below 540 dam): Snow likely

**Customization:**

```python
# In your rendering code
thickness_contours = ax.contour(
    lons, lats, thickness_data,
    levels=np.arange(480, 600, config.thickness_contour_interval),
    colors='red',
    linewidths=1.0,
    linestyles='dashed'
)

# Highlight 540 dam line
thickness_540 = ax.contour(
    lons, lats, thickness_data,
    levels=[540],
    colors='red',
    linewidths=2.5,  # Thicker line
    linestyles='solid'
)
```

### MSLP Contour Styling

Mean sea level pressure contours are the primary synoptic feature.

**Standard styling:**
- Black solid lines
- Labels showing pressure in hPa/mb
- Bold labels at regular intervals

**Customization:**

```python
# Draw MSLP contours
mslp_contours = ax.contour(
    lons, lats, mslp_data,
    levels=np.arange(960, 1048, config.mslp_contour_interval),
    colors='black',
    linewidths=1.5
)

# Add labels
ax.clabel(mslp_contours, inline=True, fontsize=10, fmt='%d')
```

### Font and Label Customization

Control text appearance across the chart.

```python
import matplotlib.pyplot as plt

# Set default font properties
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

# Title font
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

# Label font
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'normal'
```

### Chart Annotations

Add custom annotations to charts:

```python
# Add timestamp annotation
from datetime import datetime

ax.text(
    0.02, 0.02,  # Position (2% from left, 2% from bottom)
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
    transform=ax.transAxes,  # Use axes coordinates (0-1)
    fontsize=8,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)
```

### Output Format Options

**PNG (default):**
- Best for: Web, presentations, general use
- Pros: Good quality, reasonable file size
- Cons: Raster format, pixelated when enlarged

```python
output_path = "chart.png"
```

**PDF (vector graphics):**
- Best for: Publications, printing, scalable graphics
- Pros: Infinite resolution, small file size for simple charts
- Cons: Can be large with complex contours

```python
output_path = "chart.pdf"
```

**SVG (vector graphics):**
- Best for: Web graphics, editing in Illustrator/Inkscape
- Pros: Editable, scalable, web-friendly
- Cons: Not all viewers support complex SVG features

```python
output_path = "chart.svg"
```

**Format-specific settings:**

```python
# High-quality PNG
fig.savefig("chart.png", dpi=300, bbox_inches='tight', facecolor='white')

# Compressed PDF
fig.savefig("chart.pdf", bbox_inches='tight', metadata={'Creator': 'SynopticCharts'})

# Clean SVG
fig.savefig("chart.svg", bbox_inches='tight', format='svg')
```

## Configuration Templates

### Publication Quality

For papers, presentations, and high-resolution outputs:

```yaml
default_dpi: 300
figure_width: 16.0
figure_height: 12.0
mslp_contour_interval: 2.0
thickness_contour_interval: 4.0
trace_threshold: 0.01
light_threshold: 0.1
moderate_threshold: 0.25
heavy_threshold: 0.5
```

### Quick Preview

For testing, development, and rapid iteration:

```yaml
default_dpi: 100
figure_width: 12.0
figure_height: 9.0
mslp_contour_interval: 4.0
thickness_contour_interval: 12.0
trace_threshold: 0.05
light_threshold: 0.15
moderate_threshold: 0.35
heavy_threshold: 0.75
```

### Video Generation

Optimized for smooth animations:

```yaml
default_dpi: 150
figure_width: 14.0
figure_height: 10.5  # 4:3 ratio
mslp_contour_interval: 4.0
thickness_contour_interval: 6.0
trace_threshold: 0.01
light_threshold: 0.1
moderate_threshold: 0.25
heavy_threshold: 0.5
```

### Regional Detail

For small regions needing fine detail:

```yaml
default_dpi: 200
figure_width: 14.0
figure_height: 12.0
mslp_contour_interval: 2.0
thickness_contour_interval: 3.0
trace_threshold: 0.005
light_threshold: 0.05
moderate_threshold: 0.15
heavy_threshold: 0.3
```

## See Also

- [Examples Directory](../examples/README.md) - Example scripts using configurations
- [Main README](../README.md) - Package overview and quick start
- [API Reference](../README.md#api-reference) - Detailed API documentation
