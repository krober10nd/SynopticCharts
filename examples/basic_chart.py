"""
Basic Chart Generation Example

This example demonstrates how to create a single synoptic chart using the
SynopticCharts package. It shows the simplest workflow: specify a model,
forecast cycle, and lead time, then generate a chart.

Output: A PNG file showing MSLP contours, precipitation, thickness, and
surface features for a 24-hour forecast (GFS or ECMWF).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from synoptic_charts import (
    Config,
    DataFetchError,
    ModelDownloader,
    RenderError,
    create_chart_from_data,
)
from synoptic_charts.calculations.meteo import (
    calculate_thickness,
    convert_precip_rate_to_inches_per_hour,
    smooth_mslp,
    expand_categorical_mask,
)
from synoptic_charts.constants import REGIONS
from synoptic_charts.rendering.basemap import BasemapRenderer, get_projection_from_region
from synoptic_charts.rendering.layers import _extract_lat_lon


def _resolve_herbie_paint(path: str):
    """Best-effort resolve of a Herbie paint object from a dotted path.

    Supports both older style namespaces (e.g., 'nws.pop_snow2') and the current
    Herbie class-style names (e.g., 'NWSProbabilityOfPrecipitationSnow').
    Returns None if not available.
    """

    try:
        from herbie import paint as herbie_paint
    except Exception:
        return None

    obj = herbie_paint
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _paint_to_cmap(paint_obj):
    if paint_obj is None:
        return None
    for attr in ("cmap",):
        if hasattr(paint_obj, attr):
            return getattr(paint_obj, attr)
    for attr in ("kwargs2", "kwargs"):
        if hasattr(paint_obj, attr):
            d = getattr(paint_obj, attr)
            if isinstance(d, dict) and "cmap" in d:
                return d["cmap"]
    return None


def _pick_cmap_color(cmap, t: float = 0.75) -> str:
    try:
        return mcolors.to_hex(cmap(float(t)))
    except Exception:
        return "#444444"


def save_precip_mask_diagnostic(
    *,
    data_dict: dict,
    model: str,
    forecast_cycle: datetime,
    valid_time: datetime,
    lead_time: int,
    region: str,
    config: Config,
    output_path: Path,
) -> Path:
    """Save a single-panel overlay plot of categorical precip masks.

    Plots all precipitation-type boolean masks on one map (semi-transparent)
    and overlays MSLP + thickness contours for context.
    """

    precip_categorical = data_dict.get("precip_categorical") or {}
    precip_rate_ds = data_dict.get("precip_rate")
    if isinstance(precip_rate_ds, (list, tuple)) and precip_rate_ds:
        precip_rate_ds = precip_rate_ds[0]
    heights = data_dict.get("geopotential_heights") or {}

    # Extract MSLP DataArray
    mslp_ds = data_dict.get("mslp")
    mslp_var = None
    if hasattr(mslp_ds, "data_vars"):
        for name in ["PRMSL", "prmsl", "msl", "pressure", "slp"]:
            if name in mslp_ds.data_vars:
                mslp_var = mslp_ds[name]
                break
        if mslp_var is None:
            data_vars = list(mslp_ds.data_vars)
            if data_vars:
                mslp_var = mslp_ds[data_vars[0]]
    else:
        mslp_var = mslp_ds

    if mslp_var is None:
        raise ValueError("Could not extract MSLP variable for mask diagnostic")

    smoothed_mslp = smooth_mslp(mslp_var, sigma=2.0)

    # Thickness
    height_500 = heights.get("500mb")
    height_1000 = heights.get("1000mb")
    thickness = None
    if height_500 is not None and height_1000 is not None:
        thickness = calculate_thickness(height_500, height_1000)

    # Figure + axes (single panel)
    extent = REGIONS[region]["extent"]
    projection = get_projection_from_region(region)
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(config.figure_width, config.figure_height),
        dpi=config.default_dpi,
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    bg = getattr(config, "background_color", "#f2f3f5")
    fig.patch.set_facecolor(bg)

    def _is_dark(color: str) -> bool:
        try:
            r, g, b = mcolors.to_rgb(color)
        except Exception:
            return False
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.35

    dark_theme = _is_dark(bg)
    text_color = "white" if dark_theme else "black"
    mslp_color = "#e6edf3" if dark_theme else "black"
    thickness_color = "#c9d1d9" if dark_theme else "#333333"

    basemap = BasemapRenderer(region=region, config=config)

    panel_types = ["rain", "snow", "frzr", "sleet"]

    # Expand masks by adjacent cells to remove artificial gaps at boundaries,
    # and make them mutually exclusive in a stable priority order.
    dilation_iters = int(getattr(config, "categorical_mask_dilation", 0) or 0)
    priority = ["frzr", "sleet", "snow", "rain"]
    exclusive_masks = {}
    assigned = None
    for ptype in priority:
        raw = precip_categorical.get(ptype)
        if raw is None:
            continue
        try:
            expanded = expand_categorical_mask(raw, iterations=dilation_iters)
        except Exception:
            expanded = raw
        mask_bool = (_extract_lat_lon(expanded)[2] > 0) if expanded is not None else None
        if mask_bool is None:
            continue
        if assigned is None:
            excl = mask_bool
            assigned = excl
        else:
            excl = mask_bool & (~assigned)
            assigned = assigned | excl
        exclusive_masks[ptype] = excl

    # Prefer the requested Herbie paint colormaps (if available), otherwise
    # fall back to the closest available class colormaps, and finally to
    # static hex colors.
    requested_paint = {
        "rain": [
            "radar.reflectivity2",
            "radar.reflectivity",
            "RadarReflectivity",
        ],
        "snow": [
            "nws.pop_snow2",
            "NWSProbabilityOfPrecipitationSnow",
            "NWSPrecipitationSnow",
        ],
        "sleet": [
            "nws.pop_ice2",
            "NWSProbabilityOfPrecipitationIce",
        ],
        "frzr": [
            "nws.pcp_ice2",
            "NWSPrecipitationIce",
        ],
    }
    fallback_hex = {
        "rain": "#2e7d32",
        "snow": "#1565c0",
        "frzr": "#c62828",
        "sleet": "#6a1b9a",
        "unclassified": "#9aa4b2",
    }
    panel_colors = {}
    for ptype in panel_types:
        cmap = None
        for path in requested_paint.get(ptype, []):
            cmap = _paint_to_cmap(_resolve_herbie_paint(path))
            if cmap is not None:
                break
        panel_colors[ptype] = _pick_cmap_color(cmap, 0.75) if cmap is not None else fallback_hex[ptype]

    panel_colors["unclassified"] = fallback_hex["unclassified"]

    # Compute an unclassified mask: precip exists but none of the categorical
    # flags are set (after optional dilation/exclusivity).
    precip_inches = None
    if precip_rate_ds is not None:
        try:
            precip_rate_var = None
            for var_name in ["PRATE", "prate", "tp", "precip_rate"]:
                if hasattr(precip_rate_ds, "data_vars") and var_name in precip_rate_ds.data_vars:
                    precip_rate_var = precip_rate_ds[var_name]
                    break
                if hasattr(precip_rate_ds, "name") and precip_rate_ds.name == var_name:
                    precip_rate_var = precip_rate_ds
                    break
            if precip_rate_var is None and hasattr(precip_rate_ds, "data_vars"):
                data_vars = list(precip_rate_ds.data_vars)
                if data_vars:
                    precip_rate_var = precip_rate_ds[data_vars[0]]
            if precip_rate_var is None:
                precip_rate_var = precip_rate_ds

            precip_inches = convert_precip_rate_to_inches_per_hour(precip_rate_var)
        except Exception:
            precip_inches = None

    # Precompute contour inputs on the native grid.
    mslp_lats, mslp_lons, mslp_vals = _extract_lat_lon(smoothed_mslp)
    mslp_interval = float(getattr(config, "mslp_contour_interval", 4.0) or 4.0)
    if mslp_interval <= 0:
        mslp_interval = 4.0
    mslp_lo = np.floor(np.nanmin(mslp_vals) / mslp_interval) * mslp_interval
    mslp_hi = np.ceil(np.nanmax(mslp_vals) / mslp_interval) * mslp_interval
    mslp_levels = np.arange(mslp_lo, mslp_hi + mslp_interval, mslp_interval)

    thickness_levels = None
    thickness_lats = thickness_lons = thickness_vals = None
    if thickness is not None:
        thickness_lats, thickness_lons, thickness_vals = _extract_lat_lon(thickness)
        t_int = float(getattr(config, "thickness_contour_interval", 10.0) or 10.0)
        if t_int <= 0:
            t_int = 10.0
        t_lo = np.floor(np.nanmin(thickness_vals) / t_int) * t_int
        t_hi = np.ceil(np.nanmax(thickness_vals) / t_int) * t_int
        thickness_levels = np.arange(t_lo, t_hi + t_int, t_int)

    ax.set_facecolor(bg)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    basemap.add_geographic_features(ax)

    # Plot masks as semi-transparent overlays.
    legend_handles = []

    # First: unclassified (so true categorical types plot over it).
    if precip_inches is not None and assigned is not None:
        try:
            _plats, _plons, pr_vals = _extract_lat_lon(precip_inches)
            thr = float(getattr(config, "trace_threshold", 0.0) or 0.0)
            precip_present = pr_vals > 0 if thr <= 0 else pr_vals >= thr
            if assigned.shape == precip_present.shape:
                unclassified = precip_present & (~assigned)
                unclassified_plot = np.where(unclassified, 1.0, np.nan)
                ax.contourf(
                    _plons,
                    _plats,
                    unclassified_plot,
                    levels=[0.5, 1.5],
                    colors=[panel_colors.get("unclassified", "#9aa4b2")],
                    transform=ccrs.PlateCarree(),
                    alpha=0.35,
                    antialiased=False,
                    zorder=1,
                )
                legend_handles.append(
                    mpatches.Patch(
                        facecolor=panel_colors.get("unclassified", "#9aa4b2"),
                        edgecolor="none",
                        alpha=0.6,
                        label="UNCLASS",
                    )
                )
        except Exception:
            pass

    for ptype in panel_types:
        mask = precip_categorical.get(ptype)
        if mask is None:
            continue

        lats, lons, _mask_vals = _extract_lat_lon(mask)
        plot_vals = exclusive_masks.get(ptype)
        if plot_vals is None:
            # Fallback to raw if exclusivity construction failed.
            plot_vals = np.where(_mask_vals > 0, 1.0, np.nan)
        else:
            plot_vals = np.where(plot_vals, 1.0, np.nan)
        color = panel_colors.get(ptype, "#444444")

        ax.contourf(
            lons,
            lats,
            plot_vals,
            levels=[0.5, 1.5],
            colors=[color],
            transform=ccrs.PlateCarree(),
            alpha=0.55,
            antialiased=False,
            zorder=2,
        )

        legend_handles.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor="none",
                alpha=0.75,
                label=ptype.upper(),
            )
        )

    # Overlay contours for context (no labels to keep readable).
    ax.contour(
        mslp_lons,
        mslp_lats,
        mslp_vals,
        levels=mslp_levels,
        colors=mslp_color,
        linewidths=0.6,
        alpha=0.70,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    if thickness_levels is not None and thickness_vals is not None:
        ax.contour(
            thickness_lons,
            thickness_lats,
            thickness_vals,
            levels=thickness_levels,
            colors=thickness_color,
            linewidths=0.8,
            linestyles="--",
            alpha=0.65,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )

    ax.set_title(
        f"{model} categorical precipitation masks (overlay)  |  init {forecast_cycle:%Y-%m-%d %HZ}  "
        f"valid {valid_time:%Y-%m-%d %HZ}  f{lead_time:03d}",
        fontsize=14,
        fontweight="bold",
        color=text_color,
        pad=10,
    )

    if legend_handles:
        leg = ax.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=True,
            framealpha=0.35 if dark_theme else 0.85,
            facecolor="#0b0f14" if dark_theme else "white",
            edgecolor="none",
            fontsize=11,
        )
        for t in leg.get_texts():
            t.set_color(text_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.default_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def _cycle_hours_for_model(model_name: str) -> list[int]:
    """Return valid cycle hours (UTC) for the example.

    Notes:
        - GFS runs 4 cycles/day (00/06/12/18Z)
        - ECMWF runs 2 cycles/day (00/12Z)
    """

    model = (model_name or "").upper()
    if model == "ECMWF":
        return [0, 12]
    return [0, 6, 12, 18]


def get_recent_forecast_cycle(model_name: str, *, min_age_hours: int = 6) -> datetime:
    """Get the most recent model cycle that is likely available.

    Args:
        model_name: "GFS" or "ECMWF"
        min_age_hours: How old the cycle must be (helps avoid partially
            published cycles).
    """

    now = datetime.utcnow() - timedelta(hours=max(0, int(min_age_hours)))
    cycle_hours = _cycle_hours_for_model(model_name)

    # Pick the latest cycle hour at or before `now`.
    eligible = [h for h in cycle_hours if h <= now.hour]
    if eligible:
        cycle_hour = max(eligible)
        cycle_day = now
    else:
        cycle_hour = max(cycle_hours)
        cycle_day = now - timedelta(days=1)

    return datetime(cycle_day.year, cycle_day.month, cycle_day.day, cycle_hour)


def get_yesterdays_forecast_cycle(model_name: str) -> datetime:
    """Pick a cycle far enough in the past to be reliably available.

    This is meant for examples/tutorials where "it just works" is preferred.
    """

    # Use ~36 hours back to reduce the chance we're picking a not-yet-published cycle.
    reference = datetime.utcnow() - timedelta(days=1.5)
    cycle_hours = _cycle_hours_for_model(model_name)
    eligible = [h for h in cycle_hours if h <= reference.hour]
    if eligible:
        cycle_hour = max(eligible)
        cycle_day = reference
    else:
        cycle_hour = max(cycle_hours)
        cycle_day = reference - timedelta(days=1)
    return datetime(cycle_day.year, cycle_day.month, cycle_day.day, cycle_hour)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ============================================================================
# Basic Chart Generation
# ============================================================================

# Define forecast parameters
model = "ECMWF"  # Available models: "GFS", "ECMWF"
forecast_cycle = get_yesterdays_forecast_cycle(model)
lead_time = 24  # Forecast hour (aim for a winter-precip window)
region = "CONUS"  # Region name (currently only CONUS is predefined)
precip_mode = "rate"  # "rate" or "accumulated"

print(f"Creating synoptic chart:")
print(f"  Model: {model}")
print(f"  Cycle: {forecast_cycle.strftime('%Y-%m-%d %H:00 UTC')}")
print(f"  Lead time: F{lead_time:03d}")
print(f"  Region: {region}")
print(f"  Precip mode: {precip_mode}")
print()

print(f"Selected example forecast cycle: {forecast_cycle.strftime('%Y-%m-%d %H:00 UTC')}")
print()

try:
    # Load high-quality (publication-style) rendering settings
    config_path = Path(__file__).with_name("high_quality.yaml")
    config = Config.load_from_file(config_path)
    config.precip_mode = precip_mode

    # Fetch data once so we can create both the primary chart and diagnostics.
    downloader = ModelDownloader(
        model_name=model,
        forecast_cycle=forecast_cycle,
        lead_time=lead_time,
        config=config,
    )
    data_dict = downloader.fetch_all_data()
    if not data_dict:
        raise DataFetchError("ModelDownloader.fetch_all_data() returned no data")
    valid_time = forecast_cycle + timedelta(hours=lead_time)

    # Create chart and save to file
    suffix = "accum" if precip_mode.lower() == "accumulated" else "rate"
    output_path = Path(f"output/basic_chart_{suffix}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = create_chart_from_data(
        data_dict=data_dict,
        model_name=model,
        init_time=forecast_cycle,
        valid_time=valid_time,
        lead_time=lead_time,
        region=region,
        output_path=output_path,
        config=config,
    )

    # Also save a categorical-mask diagnostic image.
    # ECMWF does not provide the same categorical precip-type fields as GFS,
    # so this diagnostic is only meaningful for GFS.
    masks_path = None
    if model.upper() == "GFS" and precip_mode.lower() == "rate":
        masks_path = save_precip_mask_diagnostic(
            data_dict=data_dict,
            model=model,
            forecast_cycle=forecast_cycle,
            valid_time=valid_time,
            lead_time=lead_time,
            region=region,
            config=config,
            output_path=Path("output/basic_chart_precip_masks.png"),
        )
    
    print(f"Success! Chart saved to: {output_path}")
    if masks_path is not None:
        print(f"Saved precip mask diagnostic to: {masks_path}")
    else:
        print("Skipped categorical precip-mask diagnostic (not available for ECMWF)")
    print()
    print("The chart displays:")
    print("  - MSLP (mean sea level pressure) contours in black")
    print("  - Precipitation rate in colored filled contours")
    print("  - 1000-500mb thickness as colored contour lines")
    print("  - Surface high/low pressure markers (H/L)")

except DataFetchError as e:
    print(f"Error creating chart (data fetch failed): {e}")
    print()
    print("Common issues:")
    print("  - Network connectivity (data download requires internet)")
    print("  - Forecast cycle too old/new (use a recent cycle)")
    print("  - Model data not available yet (wait ~3 hours after cycle time)")

except RenderError as e:
    print(f"Error creating chart (render failed): {e}")
    print()
    print("Common issues:")
    print("  - Model data variables missing for this cycle/lead time")
    print("  - Check Herbie installation and model availability")

except Exception as e:
    print(f"Error creating chart: {e}")
    print()
    print("Common issues:")
    print("  - Verify 'herbie-data' is installed and up to date")
    print("  - Network connectivity (data download requires internet)")
    print("  - Model data not available yet (wait ~3 hours after cycle time)")


# ============================================================================
# Interactive Display (Optional)
# ============================================================================

# Uncomment the following section to display the chart interactively
# instead of saving to file:

"""
import matplotlib.pyplot as plt

print("\nCreating chart for interactive display...")

try:
    # Create chart without output_path to get figure and axes
    fig, ax = create_chart(
        model="GFS",
        forecast_cycle=get_yesterdays_forecast_cycle(),
        lead_time=18,
        region="CONUS"
    )
    
    print("Displaying chart... (close window to continue)")
    plt.show()
    
except Exception as e:
    print(f"Error: {e}")
"""
