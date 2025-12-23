"""
==========================================================
 Script: train.py
 Author: M. El Aabaribaoune
 Description:
     Main entry point for training the downscaling model.
     - Loads configuration
     - Loads and preprocesses data
     - Trains the model
     - Saves trained weights and losses

 Notes:
     - Compatible CPU / GPU
     - Model-agnostic (ViT by default)
==========================================================
"""

import torch

from config import load_config
from data_loading import load_datasets
from preprocessing import preprocess_data
from training import train_model
from utils import vprint, save_model

from utils import set_verbose

cfg = load_config("config.yaml")
set_verbose(cfg.verbose)

def main():
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    cfg = load_config(train_mode=True)
    vprint(f"Using device: {cfg.device}")

    vprint("=== Starting training process ===")

    # ------------------------------------------------------------------
    # Load and preprocess data
    # ------------------------------------------------------------------
    X, y_train, y_test, lon_in, lat_in, lon_out, lat_out, time_train, time_test = load_datasets(cfg)

    x_train_tensor, _, y_train_tensor, _ = preprocess_data(
        cfg, X, y_train, y_test
    )

    # Ensure tensors are on correct device
    x_train_tensor = x_train_tensor.to(cfg.device)
    y_train_tensor = y_train_tensor.to(cfg.device)

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    model, train_losses, val_losses = train_model(
        cfg, x_train_tensor, y_train_tensor
    )

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = save_model(cfg, model, train_losses, val_losses)
    vprint(f"Model saved successfully at: {model_path}")

    vprint("=== Training completed ===")


if __name__ == "__main__":
    main()
