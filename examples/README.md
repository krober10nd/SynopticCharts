# SynopticCharts Examples

This directory contains example scripts demonstrating various features of the SynopticCharts package.

## Prerequisites

### Python Packages

The examples require the SynopticCharts package and its dependencies:

```bash
# Install from the project directory
pip install -e ..

# Or install with optional dependencies
pip install -e "..[video,cli]"
```

### System Requirements

- **ffmpeg** (required for video generation examples):
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

- **Internet connection**: Required for downloading model data from NOAA servers

- **Disk space**: 
  - ~1-2 MB per chart frame
  - ~10-20 MB for a typical 48-hour video

## Running Examples

### 1. basic_chart.py

Creates a single synoptic chart for a 24-hour GFS forecast.

**Run:**
```bash
python basic_chart.py
```

**Output:**
- `output/basic_chart.png` - Synoptic chart showing MSLP, precipitation, thickness, and surface features

**Expected runtime:** 30-60 seconds (depending on network speed)

**Customization:**
- Change `forecast_cycle` to use different dates
- Modify `lead_time` for different forecast hours (0-384 for GFS)
- Set `model="ECMWF"` to use ECMWF data instead of GFS

### 2. video_generation.py

Generates a forecast animation video spanning 48 hours with 3-hour intervals.

**Run:**
```bash
python video_generation.py
```

**Output:**
- `output/frames/` - Directory containing 17 PNG frame files
- `output/forecast_animation.mp4` - Animated forecast video

**Expected runtime:** 8-15 minutes (with parallel processing on 4 cores)

**Customization:**
- Adjust `start_hour`, `end_hour`, `interval` for different forecast ranges
- Change `fps` (frames per second) to control animation speed
- Set `parallel=False` to disable parallel processing (slower but uses less CPU)
- Modify `config.default_dpi` for different video quality

**Performance tips:**
- Enable parallel processing: `parallel=True, max_workers=4`
- Use shorter forecast range for testing: `end_hour=24`
- Lower DPI for faster generation: `default_dpi=150`

### 3. custom_region.py

Demonstrates creating charts for custom geographic regions (Northeast US, West Coast).

**Run:**
```bash
python custom_region.py
```

**Output:**
- `output/northeast_chart.png` - Chart for Northeast US region
- `output/west_coast_chart.png` - Chart for West Coast region

**Expected runtime:** 60-90 seconds (generates 2 charts)

**Customization:**
- Define your own regions by modifying the region dictionaries
- Follow the projection parameter guidelines in the script comments
- Adjust figure sizes to match your region's aspect ratio

### 4. side_by_side_video.py

Generates a scan-friendly MP4 that compares **GFS vs ECMWF** side-by-side for the
same initialization time and lead times.

**Run:**
```bash
python side_by_side_video.py
```

**Output:**
- `output/compare_side_by_side_frames/` - stitched PNG frames (GFS left, ECMWF right)
- `output/gfs_vs_ecmwf_side_by_side.mp4` - side-by-side comparison video

**Notes:**
- Requires `ffmpeg` to be installed.
- ECMWF precipitation is rendered as total precip using the rain colormap (since
  ECMWF does not provide categorical precip-type masks like GFS).

## Configuration Examples

### Using Configuration Files

Load a configuration file when creating charts:

```python
from synoptic_charts import Config, create_chart

# Load high-quality config
config = Config.load_from_file("high_quality.yaml")

# Create chart with config
create_chart(
    model="GFS",
    forecast_cycle=datetime(2024, 1, 15, 0),
    lead_time=24,
    output_path="chart.png",
    config=config
)
```

### Configuration Files

- **high_quality.yaml** - Publication-quality settings (300 DPI, finer contours)
  - Use for: Papers, presentations, high-resolution prints
  - Trade-off: Larger files, slower rendering

- **quick_preview.yaml** - Fast preview settings (100 DPI, wider contours)
  - Use for: Testing, development, quick checks
  - Trade-off: Lower quality, less detail

## Troubleshooting

### Data Download Issues

**Problem:** `DataFetchError: Failed to fetch data`

**Solutions:**
- Check internet connection
- Verify forecast cycle is recent (within past week)
- Wait 2-3 hours after cycle time for data availability
- Try a different forecast cycle

### Video Creation Issues

**Problem:** `VideoCreationError: ffmpeg not found`

**Solutions:**
- Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
- Verify installation: `ffmpeg -version`
- Add ffmpeg to system PATH

**Problem:** Video encoding fails with codec error

**Solutions:**
- Try different codec: `codec="mpeg4"` instead of `"libx264"`
- Update ffmpeg to latest version
- Check ffmpeg codec support: `ffmpeg -codecs`

### Performance Issues

**Problem:** Frame generation is very slow

**Solutions:**
- Enable parallel processing: `parallel=True`
- Increase workers: `max_workers=4` (match your CPU cores)
- Use lower DPI: `default_dpi=150` instead of 300
- Reduce forecast range: `end_hour=24` instead of 120
- Use quick_preview.yaml configuration

### Memory Issues

**Problem:** Out of memory errors during batch processing

**Solutions:**
- Disable parallel processing: `parallel=False`
- Reduce number of workers: `max_workers=2`
- Process shorter forecast ranges
- Close other applications
- Use lower DPI and smaller figure sizes

## Common Modifications

### Change Forecast Model

```python
# Use ECMWF instead of GFS
model = "ECMWF"
```

### Adjust Video Speed

```python
# Slower animation (easier to see details)
fps = 5

# Faster animation
fps = 15
```

### Modify Forecast Range

```python
# Short-term forecast (0-24 hours, hourly)
start_hour = 0
end_hour = 24
interval = 1

# Long-term forecast (0-10 days, 6-hour intervals)
start_hour = 0
end_hour = 240
interval = 6
```

### Custom Video Quality

```python
# Higher quality (larger file)
crf = 18  # Lower CRF = better quality
preset = "slow"  # Slower preset = better compression

# Lower quality (smaller file)
crf = 28
preset = "fast"
```

## Additional Resources

- **Configuration Guide**: `../docs/configuration.md` - Comprehensive configuration documentation
- **Main README**: `../README.md` - Package overview and installation
- **API Documentation**: See docstrings in `synoptic_charts/api.py`, `batch.py`, `video.py`

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all prerequisites are installed
4. Check for recent package updates
5. Open an issue on GitHub with:
   - Error message
   - Example script that reproduces the issue
   - System information (OS, Python version, package versions)
