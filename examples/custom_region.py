"""
Custom Region Configuration Example

This example demonstrates how to create synoptic charts for custom geographic
regions by defining new region parameters. It shows how to inspect existing
region configurations and create new ones with appropriate projection settings.

Output: Charts for custom-defined regions (Northeast US, West Coast).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from synoptic_charts import create_chart, REGIONS


def get_recent_forecast_cycle() -> datetime:
    """Get most recent 00Z or 12Z GFS cycle."""
    now = datetime.utcnow()
    cycle_hour = (now.hour // 6) * 6
    cycle = datetime(now.year, now.month, now.day, cycle_hour) - timedelta(hours=6)
    return cycle


def validate_region(region: dict) -> None:
    """Basic validation that region parameters are reasonable."""
    extent = region.get("extent")
    if not (isinstance(extent, (list, tuple)) and len(extent) == 4):
        raise ValueError("Region 'extent' must be a list/tuple of [W, E, S, N]")
    west, east, south, north = extent
    if not (west < east and south < north):
        raise ValueError(f"Region extent invalid: {extent}")

    proj = region.get("projection_params")
    if not isinstance(proj, dict):
        raise ValueError("Region 'projection_params' must be a dict")
    for key in ("central_longitude", "central_latitude", "standard_parallels"):
        if key not in proj:
            raise ValueError(f"Region projection_params missing '{key}'")

    fig_size = region.get("figure_size")
    if not (isinstance(fig_size, (list, tuple)) and len(fig_size) == 2):
        raise ValueError("Region 'figure_size' must be a tuple (width, height)")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ============================================================================
# Inspect Existing Region Configuration
# ============================================================================

print("Existing CONUS Region Configuration")
print("=" * 60)

# Access the predefined CONUS region
conus = REGIONS["CONUS"]

print(f"Name: CONUS (Continental United States)")
print(f"Extent: {conus['extent']}")
print(f"  West: {conus['extent'][0]}°")
print(f"  East: {conus['extent'][1]}°")
print(f"  South: {conus['extent'][2]}°")
print(f"  North: {conus['extent'][3]}°")
print()
print(f"Projection: Lambert Conformal")
print(f"  Central Longitude: {conus['projection_params']['central_longitude']}°")
print(f"  Central Latitude: {conus['projection_params']['central_latitude']}°")
print(f"  Standard Parallels: {conus['projection_params']['standard_parallels']}")
print()
print(f"Figure Size: {conus['figure_size'][0]}\" × {conus['figure_size'][1]}\"")
print()

# ============================================================================
# Define Custom Region: Northeast US
# ============================================================================

print("Defining Custom Region: Northeast US")
print("-" * 60)

# Northeast US region covering New England and Mid-Atlantic states
# Geographic extent: Maine to Virginia, Atlantic Ocean to Ohio
northeast_region = {
    "name": "Northeast US",
    "extent": [-80.0, -66.0, 38.0, 48.0],  # [west, east, south, north] in degrees
    "projection_params": {
        "central_longitude": -73.0,  # Center of region (near NYC)
        "central_latitude": 43.0,  # Center of region (central NY)
        "standard_parallels": (40.0, 46.0),  # Two latitudes where scale is true
    },
    "figure_size": (12.0, 10.0),  # (width, height) in inches
}

print(f"Extent: {northeast_region['extent']}")
print(f"Projection center: {northeast_region['projection_params']['central_longitude']}°, "
      f"{northeast_region['projection_params']['central_latitude']}°")
print()

# Add custom region to REGIONS dictionary
validate_region(northeast_region)
REGIONS["NORTHEAST"] = northeast_region

print("Region 'NORTHEAST' added to REGIONS dictionary")
print()

# ============================================================================
# Define Custom Region: West Coast
# ============================================================================

print("Defining Custom Region: West Coast")
print("-" * 60)

# West Coast region covering California, Oregon, Washington
# Geographic extent: Pacific coast to Rockies
west_coast_region = {
    "name": "West Coast",
    "extent": [-130.0, -110.0, 32.0, 50.0],  # [west, east, south, north]
    "projection_params": {
        "central_longitude": -120.0,  # Center of region (central California)
        "central_latitude": 41.0,  # Center of region (northern California)
        "standard_parallels": (35.0, 47.0),  # Span the region's latitude range
    },
    "figure_size": (10.0, 12.0),  # (width, height) in inches - taller for N-S extent
}

print(f"Extent: {west_coast_region['extent']}")
print(f"Projection center: {west_coast_region['projection_params']['central_longitude']}°, "
      f"{west_coast_region['projection_params']['central_latitude']}°")
print()

# Add custom region to REGIONS dictionary
validate_region(west_coast_region)
REGIONS["WEST_COAST"] = west_coast_region

print("Region 'WEST_COAST' added to REGIONS dictionary")
print()

# ============================================================================
# Generate Charts for Custom Regions
# ============================================================================

print("Generating Charts")
print("=" * 60)

# Forecast parameters
model = "GFS"
forecast_cycle = get_recent_forecast_cycle()
lead_time = 24

print(f"Selected recent forecast cycle: {forecast_cycle.strftime('%Y-%m-%d %H:00 UTC')}")
print()

# Ensure output directory exists
Path("output").mkdir(parents=True, exist_ok=True)

# Generate Northeast chart
print("Creating chart for Northeast US...")
try:
    northeast_output = create_chart(
        model=model,
        forecast_cycle=forecast_cycle,
        lead_time=lead_time,
        region="NORTHEAST",
        output_path="output/northeast_chart.png"
    )
    print(f"  Success: {northeast_output}")
except Exception as e:
    print(f"  Error: {e}")

print()

# Generate West Coast chart
print("Creating chart for West Coast...")
try:
    west_coast_output = create_chart(
        model=model,
        forecast_cycle=forecast_cycle,
        lead_time=lead_time,
        region="WEST_COAST",
        output_path="output/west_coast_chart.png"
    )
    print(f"  Success: {west_coast_output}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 60)
print("Custom region charts complete!")
print()

# ============================================================================
# Projection Parameter Guidelines
# ============================================================================

"""
Lambert Conformal Projection Parameters
----------------------------------------

The Lambert Conformal projection is ideal for mid-latitude regions. Key parameters:

1. central_longitude: Geographic center of your region
   - Should be near the middle of your west-east extent
   - Example: For extent [-130, -110], use -120

2. central_latitude: Latitude center of your region
   - Should be near the middle of your south-north extent
   - Example: For extent [32, 50], use 41

3. standard_parallels: Two latitudes where scale is exact
   - Choose latitudes that span your region
   - Typically: (south + 1/6 of range, north - 1/6 of range)
   - Example: For [32, 50], use (35, 47)
   - These minimize distortion across the region

Region Extent Guidelines
-------------------------

Format: [west_lon, east_lon, south_lat, north_lat]

- Longitudes: Negative for Western Hemisphere (-180 to 0)
- Latitudes: Positive for Northern Hemisphere (0 to 90)
- Keep aspect ratio reasonable (1:1 to 2:1 typically)
- Account for projection distortion at edges

Example Regions to Try
----------------------

Southeast US:
    extent: [-92, -75, 25, 38]
    central_longitude: -83.5
    central_latitude: 31.5
    standard_parallels: (27, 35)

Great Lakes:
    extent: [-95, -75, 40, 50]
    central_longitude: -85
    central_latitude: 45
    standard_parallels: (42, 48)

Northern Rockies:
    extent: [-116, -100, 40, 50]
    central_longitude: -108
    central_latitude: 45
    standard_parallels: (42, 48)

Alaska:
    extent: [-170, -130, 54, 72]
    central_longitude: -150
    central_latitude: 63
    standard_parallels: (57, 69)

Figure Size Guidelines
----------------------

- Width × Height in inches (standard DPI = 150)
- Match aspect ratio to geographic extent
- Wider extent → wider figure
- Taller extent → taller figure
- Typical range: 10-16 inches for width, 8-14 for height
"""
