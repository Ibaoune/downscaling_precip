"""
==========================================================
 Script: data_loading.py
 Author: M. El Aabaribaoune
 Description:
     Unified data loader for downscaling experiments.

     - Loads predictors (ERA5 or LMDZ)
     - Loads precipitation targets (MSWEP or LMDZ35)
     - Applies spatial masking
     - Handles daily / sub-daily temporal resolution
     - Returns tensors and coordinate metadata

 Notes:
     - Model-agnostic (ViT, CNN, UNet, RF...)
     - GPU-ready but does NOT force GPU allocation
     - Logging via vprint only (HPC friendly)
==========================================================
"""

import os
import numpy as np
import xarray as xr
import torch

import utils as use
from utils import vprint

# -------------------------------------
#              Helpers
# -------------------------------------

def _is_daily_time_index(time_index):
    """
    Returns True if the time_index spacing is (approximately) 1 day.
    Uses median timestep for robustness to missing values.
    """
    if len(time_index) < 2:
        return True

    deltas = np.diff(time_index.values.astype("datetime64[ns]"))
    median_days = np.median(deltas).astype("timedelta64[ns]") / np.timedelta64(1, "D")
    return np.isclose(median_days, 1.0, rtol=1e-3, atol=1e-6)


def _print_basic_stats(arr, name, vprint_fn):
    """
    Print mean / min / max statistics for an xarray DataArray.
    """
    try:
        vals = arr.values
        if not np.issubdtype(vals.dtype, np.number):
            vals = vals.astype("float32")

        flat = vals.ravel()
        if flat.size == 0:
            vprint_fn(f"   [STATS] {name}: EMPTY array")
            return

        mean = np.nanmean(flat)
        vmin = np.nanmin(flat)
        vmax = np.nanmax(flat)
        units = arr.attrs.get("units", "")
        unit_str = f" {units}" if units else ""

        vprint_fn(
            f"   [STATS] {name}: mean={mean:.6g}{unit_str}, "
            f"min={vmin:.6g}, max={vmax:.6g}"
        )

    except Exception as e:
        vprint_fn(f"   [STATS] {name}: unable to compute stats ({e})")

# -------------------------------------
#              Main loader
# -------------------------------------

def load_datasets(cfg):
    """
    Unified loader for ERA5 → MSWEP and LMDZ → LMDZ35.

    Returns:
        X           : xarray.Dataset (full predictors)
        y_train     : torch.Tensor
        y_test      : torch.Tensor
        lon_in      : np.ndarray
        lat_in      : np.ndarray
        lon_out     : np.ndarray
        lat_out     : np.ndarray
        time_train  : np.ndarray
        time_test   : np.ndarray
    """

    vprint(f"=== Loading datasets (src={cfg.src}, target={cfg.target}) ===")

    # ==========================================================
    # 1) Load precipitation (target)
    # ==========================================================
    if cfg.target == "lmdz35":
        pr_file = os.path.join(cfg.data_path, "precip-1979-2100.nc")
        precip_var = "precip"
        time_dim = "time_counter"

    elif cfg.target == "mswep":
        pr_file = os.path.join(cfg.data_path, "mswep/all_Mor/mswep_1979_2020.nc")
        precip_var = "precipitation"
        time_dim = "time"

    else:
        raise ValueError("cfg.target must be 'lmdz35' or 'mswep'")

    vprint(f"Loading precipitation file: {pr_file}")
    ds_pr = xr.open_dataset(pr_file)

    ds_pr = use.mask_dataset(
        ds_pr,
        slice(cfg.lon_min, cfg.lon_max),
        slice(cfg.lat_min, cfg.lat_max),
    )

    if precip_var not in ds_pr:
        raise KeyError(f"Precip variable '{precip_var}' not found")

    y_train_x = ds_pr[precip_var].sel(
        {time_dim: slice(cfg.start_date_train, cfg.end_date_train)}
    )
    y_test_x = ds_pr[precip_var].sel(
        {time_dim: slice(cfg.start_date_test, cfg.end_date_test)}
    )

    vprint(f"Precip shapes → train={y_train_x.shape}, test={y_test_x.shape}")
    _print_basic_stats(y_train_x, "precip (train)", vprint)
    _print_basic_stats(y_test_x, "precip (test)", vprint)

    # Keep tensors on CPU (GPU later in training loop)
    y_train = torch.tensor(y_train_x.values.astype("float32"))
    y_test  = torch.tensor(y_test_x.values.astype("float32"))

    # ==========================================================
    # 2) Load predictor variables
    # ==========================================================
    variables = ["z", "q", "u", "v", "t"]
    levels = ["1000", "850", "700", "500"]
    data_arrays = []

    if cfg.src == "lmdz":
        file_pattern = "{var}_{lev}_1979-2100.nc"
    else:
        file_pattern = "era5/all_Mor/{var}_{lev}_1979-2020.nc"

    for var in variables:
        for lev in levels:
            filename = file_pattern.format(var=var.lower(), lev=lev)
            file_path = os.path.join(cfg.data_path, filename)
            vprint(f"Loading {filename}...")

            if not os.path.exists(file_path):
                vprint(f"  → FILE NOT FOUND")
                continue

            try:
                ds = xr.open_dataset(file_path).squeeze()
                ds = use.mask_dataset(
                    ds,
                    slice(cfg.lon_min, cfg.lon_max),
                    slice(cfg.lat_min, cfg.lat_max),
                )

                arr = ds[list(ds.data_vars)[0]].squeeze()

                # ERA5 cleanup
                if cfg.src == "era5":
                    rename = {}
                    if "latitude" in arr.dims:
                        rename["latitude"] = "lat"
                    if "longitude" in arr.dims:
                        rename["longitude"] = "lon"
                    if rename:
                        arr = arr.rename(rename)

                    if "level" in arr.dims:
                        arr = arr.isel(level=0, drop=True)
                    if "level" in arr.coords:
                        arr = arr.drop_vars("level", errors="ignore")

                    for c in ["number", "expver", "surface", "valid_time"]:
                        if c in arr.coords:
                            arr = arr.drop_vars(c, errors="ignore")

                arr = arr.transpose("time", "lat", "lon")
                _print_basic_stats(arr, f"{var}_{lev}", vprint)

                arr = arr.expand_dims({"level": [f"{var}_{lev}"]})
                data_arrays.append(arr)

            except Exception as e:
                vprint(f"  ERROR: {e}")

    if not data_arrays:
        raise ValueError("No predictor files loaded")

    X = xr.concat(data_arrays, dim="level", coords="minimal")
    X = X.transpose("time", "level", "lat", "lon")

    if cfg.src == "era5":
        vprint("Checking time resolution...")
        if not _is_daily_time_index(X.time):
            X = X.resample(time="1D").mean()
        if not _is_daily_time_index(y_train_x):
            y_train_x = y_train_x.resample(time="1D").mean()
        if not _is_daily_time_index(y_test_x):
            y_test_x = y_test_x.resample(time="1D").mean()

    x_train = X.sel(time=slice(cfg.start_date_train, cfg.end_date_train))
    x_test  = X.sel(time=slice(cfg.start_date_test, cfg.end_date_test))

    x_train_t = torch.tensor(x_train.values.astype("float32"))
    x_test_t  = torch.tensor(x_test.values.astype("float32"))

    vprint("=== Finished loading datasets ===")

    return (
        X,
        y_train,
        y_test,
        X.lon.values,
        X.lat.values,
        ds_pr.lon.values,
        ds_pr.lat.values,
        y_train_x[time_dim].values,
        y_test_x[time_dim].values,
    )

