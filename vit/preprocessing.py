"""
==========================================================
 Script: preprocessing.py
 Author: M. El Aabaribaoune
 Description:
     Data preprocessing utilities:
     - Normalization of predictors
     - Precipitation unit conversion
     - Conversion to PyTorch tensors

 Notes:
     - CPU-only preprocessing (GPU handled in training loop)
     - Model-agnostic
==========================================================
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from utils import vprint

def _print_stats(label, arr, units=None):
    if arr is None:
        vprint(f"{label}: None")
        return

    u = f" [{units}]" if units else ""
    vprint(
        f"{label}{u} → min={np.nanmin(arr):.4g}, "
        f"max={np.nanmax(arr):.4g}, mean={np.nanmean(arr):.4g}"
    )

def _detect_units_from_source(cfg):
    """
    Detect precipitation units based on dataset SOURCE.
    """
    src = cfg.target.lower()

    if src in ["lmdz", "era5"]:
        return "kg/m2/s"
    elif src in ["mswep", "ter"]:
        return "mm/day"
    elif src in ["imerg"]:
        return "mm/hr"
    else:
        vprint(f"Warning: unknown target '{cfg.target}', units unclear")
        return None


def _convert_precip(arr, units, label):
    vprint(f"--- Processing precipitation ({label}) ---")
    _print_stats(f"{label} raw", arr, units)

    if arr is None:
        return None, units

    if units is None:
        vprint("  Units unknown → skipping conversion")
        return arr, None

    if units in ["kg/m2/s", "kg/m^2/s", "kg/m²/s"]:
        vprint("  Converting kg/m²/s → mm/day")
        arr = arr * 86400.0
        units = "mm/day"

    elif units == "mm/hr":
        vprint("  Converting mm/hr → mm/day")
        arr = arr * 24.0
        units = "mm/day"

    _print_stats(f"{label} converted", arr, units)
    return arr, units

def preprocess_data(cfg, X, y_train, y_test):
    vprint("=== Preprocessing data ===")

    if X is None:
        raise ValueError("Input dataset X is missing.")

    # --------------------------------------------------
    # Extract predictor arrays
    # --------------------------------------------------
    x_train_np = X.sel(
        time=slice(cfg.start_date_train, cfg.end_date_train)
    ).values

    x_test_np = X.sel(
        time=slice(cfg.start_date_test, cfg.end_date_test)
    ).values

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    vprint(f"Applying normalization mode: {cfg.norm_mode}")

    if cfg.norm_mode == "global":
        mean = x_train_np.mean()
        std  = x_train_np.std()
        x_train_np = (x_train_np - mean) / (std + 1e-8)
        x_test_np  = (x_test_np  - mean) / (std + 1e-8)

    elif cfg.norm_mode == "channel":
        mean = x_train_np.mean(axis=(0, 2, 3), keepdims=True)
        std  = x_train_np.std (axis=(0, 2, 3), keepdims=True)
        x_train_np = (x_train_np - mean) / (std + 1e-8)
        x_test_np  = (x_test_np  - mean) / (std + 1e-8)

    elif cfg.norm_mode == "gridbox":
        t, c, h, w = x_train_np.shape
        x_train_std = np.zeros_like(x_train_np)
        x_test_std  = np.zeros_like(x_test_np)

        for lvl in range(c):
            for i in range(h):
                for j in range(w):
                    scaler = StandardScaler()
                    x_train_std[:, lvl, i, j] = scaler.fit_transform(
                        x_train_np[:, lvl, i, j].reshape(-1, 1)
                    ).ravel()
                    x_test_std[:, lvl, i, j] = scaler.transform(
                        x_test_np[:, lvl, i, j].reshape(-1, 1)
                    ).ravel()

        x_train_np = x_train_std
        x_test_np  = x_test_std

    elif cfg.norm_mode == "flatten":
        scaler = StandardScaler()
        x_train_flat = x_train_np.reshape(x_train_np.shape[0], -1)
        x_test_flat  = x_test_np.reshape(x_test_np.shape[0], -1)
        x_train_np = scaler.fit_transform(x_train_flat).reshape(x_train_np.shape)
        x_test_np  = scaler.transform(x_test_flat).reshape(x_test_np.shape)

    else:
        raise ValueError(f"Unknown norm_mode '{cfg.norm_mode}'")

    vprint("Normalization done.")

    # --------------------------------------------------
    # Target preprocessing (numpy → convert → tensor)
    # --------------------------------------------------
    y_train_np = y_train.numpy()
    y_test_np  = y_test.numpy()

    units = _detect_units_from_source(cfg)
    y_train_np, _ = _convert_precip(y_train_np, units, "train")
    y_test_np , _ = _convert_precip(y_test_np , units, "test")

    # --------------------------------------------------
    # Convert to CPU tensors
    # --------------------------------------------------
    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
    x_test_tensor  = torch.tensor(x_test_np , dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test_np , dtype=torch.float32)

    vprint("=== Preprocessing complete ===")

    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor

