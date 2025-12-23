"""
==========================================================
 Script: evaluation.py
 Author: M. El Aabaribaoune
 Description:
     Model evaluation and diagnostics.

     - Loads trained model
     - Runs inference on test data
     - Saves predictions to NetCDF
     - Generates diagnostic plots

 Notes:
     - GPU-safe (chunked inference)
     - Model-agnostic (ViT by default)
==========================================================
"""

import os
import torch
import xarray as xr

import utils as use
from utils import vprint, load_model
from vit_arch import DownscalingViT

def _build_model(cfg, x_test, y_test):
    if cfg.model_type == "vit":
        return DownscalingViT(
            in_channels=x_test.shape[1],
            emb_size=cfg.emb_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            output_channels=1,
            n_lat_out=y_test.shape[1],
            n_lon_out=y_test.shape[2],
        )
    else:
        raise NotImplementedError(f"Model {cfg.model_type} not supported")


def evaluate_and_save(cfg, x_test, y_test, lon, lat, time):
    vprint("=== Starting evaluation ===")

    device = cfg.device
    vprint(f"Using device: {device}")

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------
    model_arch = _build_model(cfg, x_test, y_test).to(device)
    model, train_losses, val_losses = load_model(cfg, model_arch)
    model.eval()

    vprint("Model loaded successfully.")

    # --------------------------------------------------
    # Experiment output path (same structure as training)
    # --------------------------------------------------
    exp_path = use.build_experiment_path(cfg)
    os.makedirs(exp_path, exist_ok=True)

    # --------------------------------------------------
    # Inference (chunked for memory safety)
    # --------------------------------------------------
    preds = []
    chunk_size = 500

    vprint("Running inference...")
    with torch.no_grad():
        for xb in torch.split(x_test, chunk_size):
            xb = xb.to(device)
            out = model(xb)[:, 0, :, :]
            preds.append(out.cpu())

    preds = torch.cat(preds, dim=0).numpy()

    # --------------------------------------------------
    # Save predictions to NetCDF
    # --------------------------------------------------
    ds_pred = xr.Dataset(
        {"precipitation": (["time", "lat", "lon"], preds)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    path_out_data = os.path.join(exp_path, "output_data")
    os.makedirs(path_out_data, exist_ok=True)

    out_nc = os.path.join(
        path_out_data, f"{cfg.model_type}_predictions_era5_to_{cfg.target}.nc"
    )
    ds_pred.to_netcdf(out_nc)
    vprint(f"Predictions saved at: {out_nc}")

    # --------------------------------------------------
    # Ground truth dataset
    # --------------------------------------------------
    y_test_np = y_test.numpy()
    y_test_ds = xr.Dataset(
        {"precip": (["time", "lat", "lon"], y_test_np)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    # --------------------------------------------------
    # Diagnostics / plots
    # --------------------------------------------------
    path_out_figs = os.path.join(exp_path, "output_figs")
    os.makedirs(path_out_figs, exist_ok=True)

    vprint("Generating plots...")

    use.plot_losses(
        train_losses,
        val_losses,
        model_name=cfg.model_type.upper(),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        filename=os.path.join(path_out_figs, "losses.png"),
    )

    use.spatial_comparaison_plot(
        y_test_ds,
        ds_pred,
        lon,
        lat,
        model_name=cfg.model_type.upper(),
        filename=os.path.join(path_out_figs, "spatial_distribution.png"),
        title_suffix=" (ERA5→MSWEP)",
        y_name="MSWEP",
        y_var="precip",
        model_var="precipitation",
    )

    use.monthly_precip_comparaison_plot(
        ds_pred,
        y_test_ds,
        model_name=cfg.model_type.upper(),
        filename=os.path.join(path_out_figs, "monthly_means.png"),
        y_name="MSWEP",
        y_var="precip",
        model_var="precipitation",
        title="Monthly Average Precipitation (mm/day)",
        title_suffix=" (ERA5→MSWEP)",
    )

    vprint(f"All evaluation outputs saved in: {exp_path}")
    vprint("=== Evaluation complete ===")

