#!/usr/bin/env python3
# ==========================================================
# Precipitation Comparison Script
# Spatial + Monthly plots
# ==========================================================

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
try:
    import regionmask
except ImportError:
    pass

COLORS = [
        (1.0, 1.0, 1.0),
        (0.8, 0.9, 1.0),
        (0.2, 0.4, 0.8),
        (0.1, 0.2, 0.5),
        (0.0, 0.5, 0.0),
        (0.3, 0.7, 0.3),
        (0.6, 0.9, 0.4),
        (1.0, 1.0, 0.3),
        (1.0, 0.7, 0.0),
        (1.0, 0.0, 0.0),
    ]
# ----------------------------------------------------------
# Utility print (verbose)
# ----------------------------------------------------------
def vprint(msg):
    print(msg)


# ==========================================================
# Spatial Comparison Plot
# ==========================================================
def spatial_comparaison_plot(
    y_test_ds,
    output_ds,
    lon_values,
    lat_values,
    model_name,
    filename,
    title_suffix="",
    y_name="Truth",
    y_var="precipitation",
    model_var="precipitation",
):
    """
    Plot spatial comparison between reference dataset and model output.

    Parameters
    ----------
    y_test_ds : xarray.Dataset
        Reference (truth) dataset
    output_ds : xarray.Dataset
        Model output dataset
    lon_values : array
        Longitude values
    lat_values : array
        Latitude values
    model_name : str
        Name of the model
    filename : str
        Output image filename
    """

    vprint("→ Generating spatial comparison plot")

    # Remove negative precipitation
    y_test_clean = y_test_ds[y_var].where(y_test_ds[y_var] >= 0)
    model_out_clean = output_ds[model_var].where(output_ds[model_var] >= 0)

    # Time mean
    daily_precip_y_test = y_test_clean.mean(dim="time")
    daily_precip_model = model_out_clean.mean(dim="time")

    # Statistics
    min_y, max_y, mean_y = (
        daily_precip_y_test.min().item(),
        daily_precip_y_test.max().item(),
        daily_precip_y_test.mean().item(),
    )
    min_m, max_m, mean_m = (
        daily_precip_model.min().item(),
        daily_precip_model.max().item(),
        daily_precip_model.mean().item(),
    )

    vprint(f"  Truth   → min={min_y:.2f}, max={max_y:.2f}, mean={mean_y:.2f}")
    vprint(f"  Model   → min={min_m:.2f}, max={max_m:.2f}, mean={mean_m:.2f}")

    # Colormap
    levels = np.linspace(0, 5, 11)
    colors = COLORS

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # Figure
    fig, axs = plt.subplots(
        1, 2,
        figsize=(12, 5),
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        dpi=300,
    )

    for i, (data, title, stats) in enumerate([
        (daily_precip_y_test, y_name, (min_y, max_y, mean_y)),
        (daily_precip_model, model_name, (min_m, max_m, mean_m)),
    ]):
        img = axs[i].pcolormesh(
            lon_values,
            lat_values,
            data,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )
        axs[i].coastlines()
        axs[i].set_extent([
            lon_values.min(), lon_values.max(),
            lat_values.min(), lat_values.max(),
        ])
        axs[i].set_title(
            f"{title}\n"
            f"Max={stats[1]:.2f}, Min={stats[0]:.2f}, Mean={stats[2]:.2f}"
        )

    fig.suptitle(
        "Spatial Distribution of Daily Mean Precipitation (mm/day)",
        fontsize=14,
        weight="bold",
    )

    if title_suffix:
        fig.text(0.5, 0.95, title_suffix, ha="center", fontsize=9, style="italic")

    cbar = plt.colorbar(img, ax=axs, shrink=0.8, pad=0.08)
    cbar.set_label("Precipitation (mm/day)")
    cbar.set_ticks(levels)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    vprint(f"  ✔ Spatial plot saved: {filename}")


# ==========================================================
# High-Resolution Prediction vs Target Plot
# ==========================================================
def spatial_comparison_per_epoch(
    y_true_data,
    y_pred_data,
    vmin=None,
    vmax=None,
    extent=None,
):
    # Remove negative values
    y_true_data[y_true_data < 0] = 0
    y_pred_data[y_pred_data < 0] = 0

    # Compute statistics
    true_min, true_max, true_mean = y_true_data.min(), y_true_data.max(), y_true_data.mean()
    pred_min, pred_max, pred_mean = y_pred_data.min(), y_pred_data.max(), y_pred_data.mean()

    # Set colorbar range
    vmin = true_min if vmin is None else vmin
    vmax = true_max if vmax is None else vmax

    # Use precipitation colormap
    levels = np.linspace(vmin, vmax, 11)
    cmap = mcolors.ListedColormap(COLORS)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # Create figure
    fig, axs = plt.subplots(
        1, 2, figsize=(11, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    for i, (data, title, stats) in enumerate([
        (y_true_data, "Ground Truth High-Res", (true_min, true_max, true_mean)),
        (y_pred_data, "Predicted High-Res", (pred_min, pred_max, pred_mean)),
    ]):
        img = axs[i].imshow(
            data,
            cmap=cmap,
            norm=norm,
            extent=extent,
            transform=ccrs.PlateCarree(),
        )
        axs[i].coastlines()
        axs[i].set_extent(extent)
        axs[i].set_title(
            f"{title}\n"
            f"Max={stats[1]:.2f}, Min={stats[0]:.2f}, Mean={stats[2]:.2f}"
        )

    fig.suptitle(
        "High-Resolution Precipitation Comparison (mm/day)",
        fontsize=14,
        weight="bold",
    )

    cbar = plt.colorbar(img, ax=axs, shrink=0.8, pad=0.08)
    cbar.set_label("Precipitation (mm/day)")

    plt.close()

    return fig


# ==========================================================
# Monthly Precipitation Comparison
# ==========================================================
def monthly_precip_comparaison_plot(
    output_model,
    y_test_ds,
    model_name,
    filename,
    y_name="Truth",
    y_var="precipitation",
    model_var="precipitation",
    title="Monthly Average Precipitation (mm/day)",
    title_suffix="",
):
    """
    Plot monthly averaged precipitation and compute Bias / RMSE
    """

    vprint("→ Generating monthly precipitation comparison plot")

    # Land mask
    region = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = region.mask(output_model)

    model_masked = output_model.where(mask.notnull())
    truth_masked = y_test_ds.where(mask.notnull())

    # Monthly means
    model_m = model_masked.resample(time="1ME").mean()
    truth_m = truth_masked.resample(time="1ME").mean()

    model_monthly = (
        model_m.groupby("time.month")
        .mean(dim="time")
        .mean(dim=["lat", "lon"])[model_var]
        .values
    )

    truth_monthly = (
        truth_m.groupby("time.month")
        .mean(dim="time")
        .mean(dim=["lat", "lon"])[y_var]
        .values
    )

    months = np.arange(1, 13)

    bias = (truth_monthly - model_monthly).mean()
    rmse = np.sqrt(((truth_monthly - model_monthly) ** 2).mean())

    vprint(f"  Bias  = {bias:.4f} mm/day")
    vprint(f"  RMSE  = {rmse:.4f} mm/day")

    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(months, model_monthly, "-o",
             label=f"{model_name} (Bias={bias:.2g}, RMSE={rmse:.2g})")
    plt.plot(months, truth_monthly, "-o", label=y_name)

    plt.xticks(months,
               ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    plt.xlabel("Month")
    plt.ylabel("Precipitation (mm/day)")
    plt.grid(True)
    plt.legend()

    if title:
        plt.title(f"{title}\n{y_name} vs {model_name}", weight="bold")

    if title_suffix:
        plt.suptitle(title_suffix, fontsize=8, style="italic")

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    vprint(f"  ✔ Monthly plot saved: {filename}")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    vprint("==================================================")
    vprint(" Precipitation Comparison (Spatial + Monthly)")
    vprint("==================================================")

    # ---- Paths ----
    pred_path = "results/evaluation/config_nll_vit/inference_pred_2011-2014.nc"
    truth_path = "results/evaluation/config_nll_vit/inference_truth_2011-2014.nc"

    # ---- Load datasets ----
    vprint("→ Loading NetCDF files")
    ds_pred = xr.open_dataset(pred_path)
    ds_truth = xr.open_dataset(truth_path)

    lon = ds_pred["lon"].values
    lat = ds_pred["lat"].values

    vprint("  Files loaded successfully")
    vprint(f"  Time steps: {ds_pred.dims['time']}")
    vprint(f"  Grid: {len(lat)} x {len(lon)}")

    # ---- Output directory ----
    os.makedirs("figures", exist_ok=True)

    # ---- Spatial plot ----
    spatial_comparaison_plot(
        y_test_ds=ds_truth,
        output_ds=ds_pred,
        lon_values=lon,
        lat_values=lat,
        model_name="ViT",
        filename="figures/spatial_precip_2011_2014.png",
        title_suffix="2011–2014",
    )

    # ---- Monthly plot ----
    monthly_precip_comparaison_plot(
        output_model=ds_pred,
        y_test_ds=ds_truth,
        model_name="ViT",
        filename="figures/monthly_precip_2011_2014.png",
        title_suffix="2011–2014",
    )

    vprint("✔ All plots generated successfully")

