"""
Model data downloader using Herbie for GRIB2 data retrieval.

This module provides the ModelDownloader class for fetching meteorological
variables from numerical weather prediction models.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
from herbie import Herbie

from ..config import Config
from ..constants import MODELS

# Configure logging
logger = logging.getLogger("synoptic_charts.data")


class ModelDownloader:
    """
    Downloads and manages model forecast data using Herbie.
    
    This class provides methods to fetch various meteorological variables
    including MSLP, precipitation types, and geopotential heights from
    GFS and ECMWF models.
    
    Attributes:
        model_name: Name of the model (GFS or ECMWF)
        forecast_cycle: Datetime of model initialization
        lead_time: Forecast hour
        config: Configuration object
        
    Example:
        >>> from datetime import datetime
        >>> from synoptic_charts import Config
        >>> from synoptic_charts.data import ModelDownloader
        >>> 
        >>> config = Config()
        >>> downloader = ModelDownloader(
        ...     model="GFS",
        ...     forecast_cycle=datetime(2024, 1, 15, 0),
        ...     config=config,
        ...     lead_time=24
        ... )
        >>> 
        >>> # Fetch all data at once
        >>> data = downloader.fetch_all_data()
        >>> 
        >>> # Or fetch individual variables
        >>> mslp = downloader.fetch_mslp()
        >>> precip = downloader.fetch_precipitation_rate()
    """
    
    def __init__(
        self,
        model_name: str,
        forecast_cycle: Optional[datetime] = None,
        config: Optional[Config] = None,
        lead_time: Optional[int] = None,
        *,
        init_time: Optional[datetime] = None,
        forecast_hour: Optional[int] = None
    ):
        """
        Initialize ModelDownloader.
        
        Args:
            model_name: Model name (must be in MODELS dict: "GFS" or "ECMWF")
            forecast_cycle: Datetime of model initialization
            config: Configuration object with cache settings
            lead_time: Forecast hour (must be valid for the model)
            init_time: Alias for forecast_cycle (used by synoptic_charts.api)
            forecast_hour: Alias for lead_time (used by synoptic_charts.api)
            
        Raises:
            ValueError: If model is not recognized or lead_time is invalid
        """
        if forecast_cycle is None:
            forecast_cycle = init_time
        if lead_time is None:
            lead_time = forecast_hour
        if forecast_cycle is None or lead_time is None or config is None:
            raise TypeError(
                "ModelDownloader requires 'forecast_cycle'/'init_time', "
                "'lead_time'/'forecast_hour', and 'config'."
            )

        # Validate model
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not recognized. "
                f"Available models: {list(MODELS.keys())}"
            )
        
        self.model_name = model_name
        self._model_config = MODELS[model_name]
        self._herbie_name = self._model_config["herbie_name"]
        
        # Validate lead time
        valid_lead_times = self._model_config["forecast_hours"]
        if lead_time not in valid_lead_times:
            raise ValueError(
                f"Lead time {lead_time}h not valid for {model_name}. "
                f"Valid range: {min(valid_lead_times)}-{max(valid_lead_times)}h"
            )
        
        self.forecast_cycle = forecast_cycle
        self.lead_time = lead_time
        self.config = config
        
        # Ensure cache directory exists
        self.config.ensure_directories()
        
        # Initialize Herbie
        try:
            self._herbie = Herbie(
                forecast_cycle,
                model=self._herbie_name,
                fxx=lead_time,
                save_dir=str(config.cache_dir)
            )
        except TypeError:
            # Older/newer Herbie versions may have slight signature differences.
            self._herbie = Herbie(
                date=forecast_cycle,
                model=self._herbie_name,
                fxx=lead_time,
                save_dir=str(config.cache_dir)
            )
        
        # Statistics tracking
        self._download_stats = {
            "bytes_downloaded": 0,
            "cache_hits": 0,
            "download_time": 0.0,
            "variables_fetched": []
        }
        
        logger.info(
            f"Initialized ModelDownloader: {model_name} "
            f"cycle={forecast_cycle.strftime('%Y-%m-%d %H:%M')} "
            f"fxx={lead_time}h"
        )
    
    @property
    def available_lead_times(self) -> List[int]:
        """Get list of valid forecast hours for this model."""
        return self._model_config["forecast_hours"]
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self._model_config.copy()
    
    @property
    def is_data_available(self) -> bool:
        """
        Check if data is likely available without downloading.
        
        Returns:
            True if data might be available (basic check only)
        """
        # Basic checks: data should exist if cycle is not too far in future
        # and not too far in past (typically kept for ~10 days)
        now = datetime.utcnow()
        age = now - self.forecast_cycle
        
        # Don't try to fetch data from future or older than 10 days
        if not (timedelta(0) <= age <= timedelta(days=10)):
            return False

        # Herbie can often tell us whether the underlying GRIB file exists.
        try:
            exists = getattr(self._herbie, "exists", None)
            if isinstance(exists, bool) and not exists:
                return False
        except Exception:
            # If existence checks fail, fall back to the age-based heuristic.
            pass

        # Attempt a lightweight inventory query for a core variable (MSLP).
        # This provides a stronger signal than age-based heuristics.
        try:
            return self._inventory_has_matches(self._get_search_string("mslp"))
        except Exception:
            return True

    def _inventory_has_matches(self, search_string: str) -> bool:
        """Best-effort check whether the remote file inventory contains a match."""
        if not hasattr(self._herbie, "inventory"):
            return True

        inv = None
        try:
            inv = self._herbie.inventory(search_string)
        except TypeError:
            inv = self._herbie.inventory()
        except Exception:
            return True

        if inv is None:
            return False
        if hasattr(inv, "empty"):
            return not bool(inv.empty)
        try:
            return len(inv) > 0
        except Exception:
            return True
    
    def fetch_mslp(self, use_cache: bool = True) -> Optional[xr.Dataset]:
        """
        Fetch mean sea level pressure data.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            xarray Dataset with MSLP in Pascals, or None if unavailable
        """
        logger.info(f"Fetching MSLP for {self.model_name} fxx={self.lead_time}h")
        
        search_string = self._get_search_string("mslp")
        dataset = self._fetch_with_retry(search_string, use_cache)
        
        if dataset is None:
            logger.warning("MSLP data unavailable")
            return None
        
        # Validate and standardize
        if not self._validate_dataset(dataset):
            logger.error("MSLP dataset validation failed")
            return None
        
        dataset = self._standardize_coordinates(dataset)
        
        # Convert to hPa if in Pascals
        if "PRMSL" in dataset or "msl" in dataset or "prmsl" in dataset:
            # Find the pressure variable
            var_name = None
            for name in ["PRMSL", "msl", "prmsl", "pressure"]:
                if name in dataset:
                    var_name = name
                    break
            
            if var_name and dataset[var_name].max() > 10000:
                # Likely in Pascals, convert to hPa
                dataset[var_name] = dataset[var_name] / 100.0
                dataset[var_name].attrs["units"] = "hPa"
                logger.debug("Converted MSLP from Pa to hPa")
        
        self._download_stats["variables_fetched"].append("mslp")
        logger.info("Successfully fetched MSLP data")
        return dataset
    
    def fetch_precipitation_rate(
        self,
        use_cache: bool = True,
        convert_units: bool = False,
        apply_trace_mask: bool = False
    ) -> Optional[xr.Dataset]:
        """
        Fetch precipitation rate data.
        
        Args:
            use_cache: Whether to use cached data if available
            convert_units: If True, convert from kg/m²/s to inches/hour (default: False)
            apply_trace_mask: If True, apply trace threshold masking (default: False)
            
        Returns:
            xarray Dataset with precipitation rate in native units (kg/m²/s)
            or converted units if convert_units=True, or None if unavailable
        
        Note:
            Raw data is returned by default. Unit conversion and trace masking
            should be performed in the calculations module to avoid double-processing.
        """
        logger.info(f"Fetching precipitation rate for {self.model_name} fxx={self.lead_time}h")
        
        search_string = self._get_search_string("precip_rate")
        dataset = self._fetch_with_retry(search_string, use_cache)
        
        if dataset is None:
            logger.warning("Precipitation rate data unavailable")
            return None
        
        # Validate and standardize
        if not self._validate_dataset(dataset):
            logger.error("Precipitation rate dataset validation failed")
            return None
        
        dataset = self._standardize_coordinates(dataset)

        # ECMWF precipitation commonly arrives as total precipitation accumulation (tp)
        # in meters of water equivalent. For scan-friendly maps and videos, convert
        # to an incremental precipitation rate (inches/hour) by differencing against
        # the previous available forecast hour.
        if (self.model_name == "ECMWF" or self._herbie_name == "ecmwf") and "tp" in dataset:
            try:
                tp_current = dataset["tp"]
                prev_candidates = [lt for lt in self.available_lead_times if lt < self.lead_time]
                prev_lt = max(prev_candidates) if prev_candidates else None

                if prev_lt is None or self.lead_time <= 0:
                    # Nothing to difference against; treat as zero-rate.
                    tp_rate_m_per_hr = tp_current * 0.0
                    hours = 1.0
                else:
                    hours = float(self.lead_time - prev_lt)
                    prev_herbie = Herbie(
                        self.forecast_cycle,
                        model=self._herbie_name,
                        fxx=prev_lt,
                        save_dir=str(self.config.cache_dir),
                    )
                    try:
                        prev_ds = prev_herbie.xarray(self._get_search_string("precip_rate"))
                    except TypeError:
                        prev_ds = prev_herbie.xarray(self._get_search_string("precip_rate"), remove_grib=False)

                    if isinstance(prev_ds, list):
                        prev_ds = [ds for ds in prev_ds if ds is not None]
                        prev_ds = xr.merge(prev_ds, compat="override") if prev_ds else None

                    if prev_ds is not None:
                        prev_ds = self._standardize_coordinates(prev_ds)
                    tp_prev = prev_ds["tp"] if (prev_ds is not None and "tp" in prev_ds) else None

                    if tp_prev is None or tp_prev.shape != tp_current.shape:
                        tp_rate_m_per_hr = tp_current * 0.0
                    else:
                        tp_step_m = tp_current - tp_prev
                        # Guard against small negative diffs from GRIB/float quirks.
                        tp_step_m = xr.where(tp_step_m >= 0, tp_step_m, 0.0)
                        tp_rate_m_per_hr = tp_step_m / hours

                # meters -> inches
                tp_rate_in_per_hr = tp_rate_m_per_hr * 39.3701
                dataset["tp"] = tp_rate_in_per_hr
                dataset["tp"].attrs = tp_current.attrs.copy()
                dataset["tp"].attrs["units"] = "inches/hour"
                dataset["tp"].attrs["long_name"] = "Precipitation Rate"
                dataset["tp"].attrs["description"] = (
                    f"Incremental precipitation rate derived from ECMWF tp accumulation; "
                    f"step={hours:g}h"
                )
            except Exception as e:
                logger.warning(f"Failed to derive ECMWF precip rate from tp: {e}")
        
        # Optional: Convert from kg/m²/s to inches/hour
        # kg/m²/s = mm/s (since 1 kg/m² = 1 mm water)
        # mm/s * 3600 = mm/hr
        # mm/hr * 0.0393701 = inches/hr
        if convert_units and ("PRATE" in dataset or "prate" in dataset or "tp" in dataset):
            var_name = None
            for name in ["PRATE", "prate", "tp"]:
                if name in dataset:
                    var_name = name
                    break
            
            if var_name:
                dataset[var_name] = dataset[var_name] * 3600 * 0.0393701
                dataset[var_name].attrs["units"] = "inches/hour"
                logger.debug("Converted precipitation rate to inches/hour")
                
                # Optional: Apply trace threshold
                if apply_trace_mask:
                    dataset[var_name] = dataset[var_name].where(
                        dataset[var_name] >= self.config.trace_threshold,
                        0.0
                    )
                    logger.debug(f"Applied trace threshold: {self.config.trace_threshold} in/hr")
        
        self._download_stats["variables_fetched"].append("precip_rate")
        logger.info("Successfully fetched precipitation rate data")
        return dataset

    def fetch_precipitation_accumulation(
        self,
        use_cache: bool = True,
    ) -> Optional[xr.Dataset]:
        """Fetch accumulated precipitation (init -> lead time).

        Returns a dataset containing a single variable "precip_accum" in mm.

        Notes:
                        - For GFS, this uses APCP (accumulated precipitation at the surface)
                            which is typically in kg/m^2 (numerically equivalent to mm of water).
                        - For ECMWF, this uses tp (total precipitation) which is typically
                            an accumulation in meters of water equivalent; we convert to mm.
        """

        logger.info(
            f"Fetching precipitation accumulation for {self.model_name} fxx={self.lead_time}h"
        )

        dataset = None
        model_u = (self.model_name or "").upper()

        # Best-effort fast path: some GRIB inventories include a 0-{lead} accumulation record.
        if model_u != "ECMWF" and self.lead_time and self.lead_time > 0:
            specific = f":APCP:surface:0-{int(self.lead_time)} hour acc fcst:"
            dataset = self._fetch_with_retry(specific, use_cache)

        if dataset is not None:
            try:
                if self._validate_dataset(dataset):
                    dataset = self._standardize_coordinates(dataset)
                    accum_var = None
                    for var_name in ["APCP", "apcp", "tp", "precip_accum"]:
                        if var_name in dataset.data_vars:
                            accum_var = dataset[var_name]
                            break
                    if accum_var is None:
                        data_vars = list(dataset.data_vars)
                        if data_vars:
                            accum_var = dataset[data_vars[0]]
                    if accum_var is not None:
                        da_mm = self._to_mm(accum_var)
                        da_mm.attrs["long_name"] = "Accumulated Precipitation"
                        da_mm.attrs["description"] = "Accumulated precipitation from initialization through valid time"
                        out = xr.Dataset({"precip_accum": da_mm})
                        self._download_stats["variables_fetched"].append("precip_accum")
                        return out
            except Exception:
                # Fall back to the robust cumulative computation below.
                pass

        # Robust path: compute init->lead accumulation by summing per-interval increments.
        if not self.lead_time or int(self.lead_time) <= 0:
            # No accumulation at init time.
            return None

        lead_list = [lt for lt in self.available_lead_times if 0 < int(lt) <= int(self.lead_time)]
        lead_list = sorted(set(int(lt) for lt in lead_list))

        cumulative = None
        for lt in lead_list:
            step_downloader = ModelDownloader(
                model_name=self.model_name,
                forecast_cycle=self.forecast_cycle,
                config=self.config,
                lead_time=int(lt),
            )
            step_ds = step_downloader.fetch_precipitation_accumulation_increment(use_cache=use_cache)
            if step_ds is None:
                continue
            step_da = step_ds.get("precip_step") if hasattr(step_ds, "data_vars") else None
            if step_da is None:
                try:
                    data_vars = list(step_ds.data_vars)
                    step_da = step_ds[data_vars[0]] if data_vars else None
                except Exception:
                    step_da = None
            if step_da is None:
                continue

            if cumulative is None:
                cumulative = step_da
            else:
                try:
                    a, b = xr.align(cumulative, step_da, join="inner")
                    cumulative = a + b
                except Exception:
                    cumulative = cumulative + step_da

        if cumulative is None:
            logger.warning("Precipitation accumulation data unavailable")
            return None

        cumulative = cumulative.assign_attrs(cumulative.attrs.copy() if hasattr(cumulative, "attrs") else {})
        cumulative.attrs["units"] = "mm"
        cumulative.attrs["long_name"] = "Accumulated Precipitation"
        cumulative.attrs["description"] = "Accumulated precipitation from initialization through valid time"

        out = xr.Dataset({"precip_accum": cumulative})
        self._download_stats["variables_fetched"].append("precip_accum")
        logger.info("Successfully fetched precipitation accumulation data")
        return out

    def fetch_precipitation_accumulation_increment(
        self,
        use_cache: bool = True,
    ) -> Optional[xr.Dataset]:
        """Fetch per-interval precipitation accumulation ending at this lead time.

        Returns a dataset containing a single variable "precip_step" in mm.
        This is the increment for the interval since the previous available lead.
        """

        model_u = (self.model_name or "").upper()
        logger.info(
            f"Fetching precipitation accumulation increment for {self.model_name} fxx={self.lead_time}h"
        )

        if not self.lead_time or int(self.lead_time) <= 0:
            return None

        search_string = self._get_search_string("precip_accum")
        dataset = self._fetch_with_retry(search_string, use_cache)
        if dataset is None:
            logger.warning("Precipitation increment data unavailable")
            return None
        if not self._validate_dataset(dataset):
            logger.error("Precipitation increment dataset validation failed")
            return None
        dataset = self._standardize_coordinates(dataset)

        # Extract variable
        var = None
        for var_name in ["APCP", "apcp", "tp", "precip_accum"]:
            if var_name in dataset.data_vars:
                var = dataset[var_name]
                break
        if var is None:
            data_vars = list(dataset.data_vars)
            var = dataset[data_vars[0]] if data_vars else None
        if var is None:
            logger.warning("Could not extract precipitation increment variable")
            return None

        if model_u == "ECMWF" or self._herbie_name == "ecmwf":
            # ECMWF tp is commonly an accumulation since init; difference to get per-step.
            prev_candidates = [lt for lt in self.available_lead_times if int(lt) < int(self.lead_time)]
            prev_lt = max(prev_candidates) if prev_candidates else None
            if prev_lt is None:
                step = var * 0.0
            else:
                prev_herbie = Herbie(
                    self.forecast_cycle,
                    model=self._herbie_name,
                    fxx=int(prev_lt),
                    save_dir=str(self.config.cache_dir),
                )
                try:
                    prev_ds = prev_herbie.xarray(search_string)
                except TypeError:
                    prev_ds = prev_herbie.xarray(search_string, remove_grib=False)

                if isinstance(prev_ds, list):
                    prev_ds = [ds for ds in prev_ds if ds is not None]
                    prev_ds = xr.merge(prev_ds, compat="override") if prev_ds else None
                if prev_ds is not None:
                    prev_ds = self._standardize_coordinates(prev_ds)
                prev_var = None
                if prev_ds is not None and hasattr(prev_ds, "data_vars"):
                    for vn in ["tp", "TP", "precip_accum"]:
                        if vn in prev_ds.data_vars:
                            prev_var = prev_ds[vn]
                            break
                    if prev_var is None:
                        dvs = list(prev_ds.data_vars)
                        prev_var = prev_ds[dvs[0]] if dvs else None

                if prev_var is None or prev_var.shape != var.shape:
                    step = var * 0.0
                else:
                    step = var - prev_var
                    step = xr.where(step >= 0, step, 0.0)
        else:
            # GFS APCP is typically a period accumulation (increment).
            step = var

        step_mm = self._to_mm(step)
        step_mm.attrs["long_name"] = "Precipitation Increment"
        step_mm.attrs["description"] = "Per-interval precipitation accumulation ending at valid time"

        self._download_stats["variables_fetched"].append("precip_step")
        return xr.Dataset({"precip_step": step_mm})

    def _to_mm(self, da: xr.DataArray) -> xr.DataArray:
        """Convert precipitation-like fields to mm (best-effort)."""

        units = str(getattr(da, "attrs", {}).get("units", "") or "").strip().lower()
        model_u = (self.model_name or "").upper()

        da_mm = da
        if model_u == "ECMWF" or self._herbie_name == "ecmwf":
            # ECMWF tp is commonly meters of water equivalent.
            if units in {"m", "meter", "meters", "metre", "metres"} or units.startswith("m ") or units == "":
                da_mm = da * 1000.0
        else:
            # GFS APCP is commonly kg m-2 or mm; both are numerically mm.
            da_mm = da

        da_mm = da_mm.assign_attrs(da.attrs.copy() if hasattr(da, "attrs") else {})
        da_mm.attrs["units"] = "mm"
        return da_mm
    
    def fetch_categorical_precipitation(
        self, use_cache: bool = True
    ) -> Optional[Dict[str, xr.DataArray]]:
        """
        Fetch categorical precipitation types (rain, snow, freezing rain, sleet).
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with keys 'rain', 'snow', 'frzr', 'sleet' mapping to
            binary DataArrays (0 or 1), or None if data unavailable
        """
        # ECMWF does not provide the same categorical precip-type flags as GFS.
        # Previously we mapped all types to total precipitation (tp), which
        # caused the renderer's priority logic to label all precip as freezing
        # rain. Treat ECMWF as "no categorical precip available".
        if self.model_name == "ECMWF" or self._herbie_name == "ecmwf":
            logger.info(
                "Skipping categorical precipitation fetch for ECMWF "
                "(model does not provide categorical precip-type flags)"
            )
            return None

        logger.info(
            f"Fetching categorical precipitation for {self.model_name} "
            f"fxx={self.lead_time}h"
        )
        
        precip_types = {}
        type_map = {
            "rain": "CRAIN",
            "snow": "CSNOW",
            "frzr": "CFRZR",
            "sleet": "CICEP"
        }
        
        for precip_type, var_name in type_map.items():
            search_string = self._get_search_string(precip_type)
            dataset = self._fetch_with_retry(search_string, use_cache)
            
            if dataset is not None:
                dataset = self._standardize_coordinates(dataset)
                
                # Extract the data array
                if var_name in dataset:
                    precip_types[precip_type] = dataset[var_name]
                elif var_name.lower() in dataset:
                    precip_types[precip_type] = dataset[var_name.lower()]
                else:
                    # Try to find any variable in the dataset
                    data_vars = list(dataset.data_vars)
                    if data_vars:
                        precip_types[precip_type] = dataset[data_vars[0]]
                
                logger.debug(f"Fetched {precip_type} data")
            else:
                logger.warning(f"Categorical {precip_type} data unavailable")
        
        if not precip_types:
            logger.warning("No categorical precipitation data available")
            return None
        
        # Validate all arrays have consistent grids
        if len(precip_types) > 1:
            ref_shape = list(precip_types.values())[0].shape
            for ptype, data in precip_types.items():
                if data.shape != ref_shape:
                    logger.warning(
                        f"Inconsistent grid shape for {ptype}: "
                        f"{data.shape} vs {ref_shape}"
                    )
        
        self._download_stats["variables_fetched"].append("categorical_precip")
        logger.info(
            f"Successfully fetched {len(precip_types)} categorical "
            f"precipitation types"
        )
        return precip_types
    
    def fetch_geopotential_heights(
        self, use_cache: bool = True
    ) -> Optional[Dict[str, xr.DataArray]]:
        """
        Fetch geopotential heights at 500mb and 1000mb.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with keys '500mb' and '1000mb' mapping to DataArrays
            with heights in geopotential meters, or None if unavailable
        """
        logger.info(
            f"Fetching geopotential heights for {self.model_name} "
            f"fxx={self.lead_time}h"
        )
        
        heights = {}
        
        # Fetch 500mb height
        search_500 = self._get_search_string("height_500")
        dataset_500 = self._fetch_with_retry(search_500, use_cache)
        
        if dataset_500 is not None:
            dataset_500 = self._standardize_coordinates(dataset_500)
            
            # Find height variable
            for var_name in ["HGT", "gh", "z", "height"]:
                if var_name in dataset_500:
                    heights["500mb"] = dataset_500[var_name]
                    logger.debug("Fetched 500mb height")
                    break
        else:
            logger.error("500mb height data unavailable (critical)")
            return None
        
        # Fetch 1000mb height
        search_1000 = self._get_search_string("height_1000")
        dataset_1000 = self._fetch_with_retry(search_1000, use_cache)
        
        if dataset_1000 is not None:
            dataset_1000 = self._standardize_coordinates(dataset_1000)
            
            # Find height variable
            for var_name in ["HGT", "gh", "z", "height"]:
                if var_name in dataset_1000:
                    heights["1000mb"] = dataset_1000[var_name]
                    logger.debug("Fetched 1000mb height")
                    break
        else:
            logger.warning("1000mb height data unavailable (may not exist over terrain)")
            # Explicitly set to None to indicate missing data
            heights["1000mb"] = None
        
        if "500mb" not in heights:
            logger.error("Critical: 500mb height missing")
            return None
        
        # Validate grids match (only if 1000mb exists)
        if "1000mb" in heights and heights["1000mb"] is not None:
            if heights["500mb"].shape != heights["1000mb"].shape:
                logger.warning(
                    f"Inconsistent grid shapes: 500mb={heights['500mb'].shape}, "
                    f"1000mb={heights['1000mb'].shape}"
                )
        
        self._download_stats["variables_fetched"].append("geopotential_heights")
        logger.info("Successfully fetched geopotential heights")
        return heights
    
    def fetch_all_data(self, use_cache: bool = True, *, include_precip: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch all meteorological variables.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with keys:
                - 'mslp': xarray Dataset with MSLP
                - 'precip_rate': xarray Dataset with precipitation rate
                - 'precip_categorical': Dict of precipitation type DataArrays
                - 'geopotential_heights': Dict of height DataArrays
            Returns None if critical variables are missing
        """
        logger.info(
            f"Fetching all data for {self.model_name} "
            f"cycle={self.forecast_cycle.strftime('%Y-%m-%d %H:%M')} "
            f"fxx={self.lead_time}h"
        )
        
        start_time = time.time()
        
        # Fetch core variables
        mslp = self.fetch_mslp(use_cache)
        geopotential_heights = self.fetch_geopotential_heights(use_cache)

        # Fetch precipitation based on the configured mode.
        precip_mode = str(getattr(self.config, "precip_mode", "rate") or "rate").strip().lower()
        precip_rate = None
        precip_categorical = None
        precip_accum = None

        if include_precip:
            if precip_mode == "accumulated":
                precip_accum = self.fetch_precipitation_accumulation(use_cache)
            else:
                precip_rate = self.fetch_precipitation_rate(use_cache)
                precip_categorical = self.fetch_categorical_precipitation(use_cache)
        
        # Check critical variables
        if mslp is None or geopotential_heights is None:
            logger.error(
                "Cannot create complete dataset: missing critical variables "
                "(MSLP or geopotential heights)"
            )
            return None
        
        # Check that both height levels exist (1000mb may be None over terrain)
        if geopotential_heights.get("1000mb") is None:
            logger.error(
                "Cannot create complete dataset: 1000mb height unavailable "
                "(required for thickness calculations)"
            )
            return None
        
        # Build result dictionary
        result = {
            "mslp": mslp,
            "precip_rate": precip_rate,
            "precip_categorical": precip_categorical,
            "precip_accum": precip_accum,
            "geopotential_heights": geopotential_heights,
        }
        
        # Update statistics
        elapsed = time.time() - start_time
        self._download_stats["download_time"] = elapsed
        
        # Log summary
        available_vars = [k for k, v in result.items() if v is not None]
        logger.info(
            f"Fetch complete: {len(available_vars)}/4 variable groups available "
            f"(time: {elapsed:.2f}s)"
        )
        logger.info(f"Variables fetched: {self._download_stats['variables_fetched']}")
        
        return result
    
    def _fetch_with_retry(
        self,
        search_string: str,
        use_cache: bool = True,
        max_retries: int = 3
    ) -> Optional[xr.Dataset]:
        """
        Fetch data with exponential backoff retry logic.
        
        Args:
            search_string: GRIB2 search string for Herbie
            use_cache: Whether to use cached data (Note: Herbie controls caching
                      internally via save_dir. This parameter is kept for API
                      consistency but may not fully bypass cache. To force fresh
                      downloads, clear the cache directory or use a different
                      save_dir.)
            max_retries: Maximum number of retry attempts
            
        Returns:
            xarray Dataset or None if all retries fail
        """
        delays = [2, 4, 8]  # Exponential backoff delays in seconds

        # Best-effort preflight: check inventory for this search string.
        try:
            if not self._inventory_has_matches(search_string):
                logger.warning(
                    f"No inventory match for search '{search_string}' "
                    f"({self.model_name} cycle={self.forecast_cycle.strftime('%Y-%m-%d %H:%M')} fxx={self.lead_time}h)"
                )
                return None
        except Exception:
            # Inventory lookup is not guaranteed; proceed with direct fetch.
            pass
        
        # Note: Herbie's xarray() method automatically uses cached files from save_dir.
        # The use_cache parameter is documented but cannot be fully honored without
        # modifying Herbie's internal behavior or clearing the cache directory.
        if not use_cache:
            logger.debug(
                "use_cache=False requested, but Herbie may still use cached files. "
                "Consider clearing cache or using clear_cache() method."
            )
        
        for attempt in range(max_retries):
            try:
                # Try to fetch data
                try:
                    dataset = self._herbie.xarray(search_string)
                except TypeError:
                    # Herbie versions differ in xarray() kwargs.
                    dataset = self._herbie.xarray(search_string, remove_grib=False)

                # cfgrib may open multiple hypercubes; Herbie can return a list.
                bytes_downloaded = 0
                if isinstance(dataset, list):
                    datasets = [ds for ds in dataset if ds is not None]
                    if not datasets:
                        dataset = None
                    else:
                        try:
                            dataset = xr.merge(datasets, compat="override")
                        except Exception:
                            dataset = datasets[0]
                        try:
                            bytes_downloaded = sum(getattr(ds, "nbytes", 0) for ds in datasets)
                        except Exception:
                            bytes_downloaded = 0
                
                if dataset is not None:
                    if bytes_downloaded:
                        self._download_stats["bytes_downloaded"] += bytes_downloaded
                    else:
                        self._download_stats["bytes_downloaded"] += getattr(dataset, "nbytes", 0)
                    # Increment cache hits if file was likely from cache
                    # (Herbie downloads to save_dir, subsequent calls use cached file)
                    return dataset
                else:
                    logger.warning(f"No data returned for search: {search_string}")
                    
            except (ConnectionError, TimeoutError) as e:
                logger.warning(
                    f"Network error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    delay = delays[attempt]
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    
            except FileNotFoundError as e:
                logger.warning(f"Data file not found: {e}")
                return None  # No point retrying if file doesn't exist

            except (ValueError, KeyError) as e:
                # Often indicates a search string mismatch / inventory miss.
                logger.error(
                    f"Herbie failed to open dataset for search '{search_string}': {e}"
                )
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error fetching data: {e}")
                if attempt < max_retries - 1:
                    delay = delays[attempt]
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
        
        logger.error(
            f"Failed to fetch data after {max_retries} attempts: {search_string}"
        )
        return None
    
    def _validate_dataset(self, dataset: xr.Dataset) -> bool:
        """
        Validate that dataset has required structure and reasonable values.
        
        Args:
            dataset: xarray Dataset to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        if dataset is None or len(dataset.data_vars) == 0:
            logger.error("Dataset is None or empty")
            return False
        
        # Check for coordinates
        required_coords = {"latitude", "longitude", "lat", "lon", "x", "y"}
        available_coords = set(dataset.coords.keys())
        
        if not any(coord in available_coords for coord in required_coords):
            logger.error(f"Missing required coordinates. Available: {available_coords}")
            return False
        
        # Check for excessive NaN values
        for var_name in dataset.data_vars:
            var = dataset[var_name]
            if var.size > 0:
                nan_fraction = np.isnan(var.values).sum() / var.size
                if nan_fraction > 0.5:
                    logger.warning(
                        f"Variable {var_name} has {nan_fraction*100:.1f}% NaN values"
                    )
        
        # Validate coordinate ranges
        lat_names = [c for c in dataset.coords if c in {"latitude", "lat", "y"}]
        lon_names = [c for c in dataset.coords if c in {"longitude", "lon", "x"}]
        
        if lat_names:
            lat_values = dataset[lat_names[0]].values
            if np.min(lat_values) < -90 or np.max(lat_values) > 90:
                logger.error(
                    f"Invalid latitude range: [{np.min(lat_values)}, {np.max(lat_values)}]"
                )
                return False
        
        if lon_names:
            lon_values = dataset[lon_names[0]].values
            if np.min(lon_values) < -180 or np.max(lon_values) > 360:
                logger.error(
                    f"Invalid longitude range: [{np.min(lon_values)}, {np.max(lon_values)}]"
                )
                return False
        
        return True
    
    def _get_search_string(self, variable: str) -> str:
        """
        Get GRIB2 search string for a variable.
        
        Args:
            variable: Variable identifier (mslp, precip_rate, rain, etc.)
            
        Returns:
            GRIB2 search string for Herbie
        """
        # GFS search strings
        gfs_search_strings = {
            "mslp": ":PRMSL:mean sea level:",
            "precip_rate": ":PRATE:surface:",
            "precip_accum": ":APCP:surface:",
            "rain": ":CRAIN:surface:",
            "snow": ":CSNOW:surface:",
            "frzr": ":CFRZR:surface:",
            "sleet": ":CICEP:surface:",
            "height_500": ":HGT:500 mb:",
            "height_1000": ":HGT:1000 mb:"
        }
        
        # ECMWF search strings (different variable names)
        ecmwf_search_strings = {
            "mslp": ":msl:",  # mean sea level pressure
            "precip_rate": ":tp:",  # total precipitation
            "precip_accum": ":tp:",
            "rain": ":tp:",  # ECMWF doesn't have categorical precip types
            "snow": ":tp:",
            "frzr": ":tp:",
            "sleet": ":tp:",
            "height_500": ":gh:500:",  # geopotential height at 500mb
            "height_1000": ":gh:1000:"  # geopotential height at 1000mb
        }
        
        # Select search strings based on model
        if self.model_name == "ECMWF" or self._herbie_name == "ecmwf":
            search_strings = ecmwf_search_strings
        else:
            search_strings = gfs_search_strings
        
        if variable not in search_strings:
            logger.warning(f"Unknown variable: {variable}")
            return f":{variable}:"
        
        return search_strings[variable]
    
    def _standardize_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Standardize coordinate names and values.
        
        Args:
            dataset: Input xarray Dataset
            
        Returns:
            Dataset with standardized coordinates
        """
        # Rename coordinates to standard names
        rename_map = {}
        
        # Find and standardize latitude
        for old_name in ["latitude", "y", "lat"]:
            if old_name in dataset.coords and old_name != "lat":
                rename_map[old_name] = "lat"
                break
        
        # Find and standardize longitude
        for old_name in ["longitude", "x", "lon"]:
            if old_name in dataset.coords and old_name != "lon":
                rename_map[old_name] = "lon"
                break
        
        if rename_map:
            dataset = dataset.rename(rename_map)
            logger.debug(f"Renamed coordinates: {rename_map}")
        
        # Convert longitude from 0-360 to -180-180 if needed
        if "lon" in dataset.coords:
            lon_values = dataset["lon"].values
            if np.max(lon_values) > 180:
                dataset = dataset.assign_coords(
                    lon=(dataset["lon"] + 180) % 360 - 180
                )
                # Sort by longitude if needed
                dataset = dataset.sortby("lon")
                logger.debug("Converted longitude to -180 to 180 range")
        
        # Add coordinate attributes
        if "lat" in dataset.coords:
            dataset["lat"].attrs.update({
                "units": "degrees_north",
                "standard_name": "latitude",
                "long_name": "Latitude"
            })
        
        if "lon" in dataset.coords:
            dataset["lon"].attrs.update({
                "units": "degrees_east",
                "standard_name": "longitude",
                "long_name": "Longitude"
            })
        
        return dataset
    
    def clear_cache(self, max_age_days: int = 7) -> int:
        """
        Remove old cached files.
        
        Args:
            max_age_days: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        cache_dir = Path(self.config.cache_dir)
        if not cache_dir.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        for file_path in cache_dir.glob("**/*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old cache file: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info(f"Cleared {removed_count} old cache files (>{max_age_days} days)")
        return removed_count
    
    def get_download_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of download operations.
        
        Returns:
            Dictionary with download statistics
        """
        return {
            "model": self.model_name,
            "forecast_cycle": self.forecast_cycle.isoformat(),
            "lead_time": self.lead_time,
            "bytes_downloaded": self._download_stats["bytes_downloaded"],
            "cache_hits": self._download_stats["cache_hits"],
            "download_time_seconds": self._download_stats["download_time"],
            "variables_fetched": self._download_stats["variables_fetched"]
        }
