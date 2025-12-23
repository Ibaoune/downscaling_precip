"""
==========================================================
 Script: training.py
 Author: M. El Aabaribaoune
 Description:
     Defines the training loop, loss computation, and
     early stopping for the downscaling model.

 Design:
     - Model-agnostic (ViT, CNN, UNet, etc.)
     - GPU / CPU compatible
==========================================================
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from vit_arch import DownscalingViT
from utils import vprint


def _build_model(cfg, x_train, y_train):
    """
    Factory function for model instantiation.
    Can be extended to support CNN / UNet / RF.
    """
    if cfg.model_type == "vit":
        return DownscalingViT(
            in_channels=x_train.shape[1],
            emb_size=cfg.emb_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            output_channels=1,
            n_lat_out=y_train.shape[1],
            n_lon_out=y_train.shape[2],
        )
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} not supported yet")


def train_model(cfg, x_train, y_train):
    vprint("Initializing model for training...")

    model = _build_model(cfg, x_train, y_train).to(cfg.device)

    # ------------------------------------------------------------------
    # Loss & optimizer
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(cfg.device.type == "cuda"),
    )

    train_losses = []
    val_losses = []  # kept for compatibility
    best_loss = float("inf")
    patience = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(xb)[:, 0, :, :]
            loss = criterion(outputs, yb.squeeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)

        vprint(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {epoch_loss:.4f}")

        # ------------------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------------------
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_max:
                vprint("Early stopping triggered.")
                break

    return model, train_losses, val_losses
