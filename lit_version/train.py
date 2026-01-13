import pandas as pd
from lit_module import LitDataModule, LitModule
from models.vit import DownscalingViT
from models.unet import UNet
from models.losses import BernoulliGammaLoss
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import torch
import os
import yaml
import numpy as np
import xarray as xr
from datetime import datetime

def inference(model_module, data_loader=None, train_loader=None, output_dir='results'):
    model_module.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_module.to(device)
    # check criterion instance, if nll-bg, compute expected value
    is_nll = isinstance(model_module.criterion, BernoulliGammaLoss)
    with torch.no_grad():
        all_y_pred = []
        all_y_true = []
        all_dates = []

        for batch in data_loader:
            x, y, forcings, date = batch
            x = x.to(device)
            y_true = y.to(device)
            y_pred = model_module(x, forcings.to(device))
            if is_nll:
                # Expected value of Bernoulli-Gamma
                y_pred = y_pred[:, 0:1, :, :] * y_pred[:, 1:2, :, :] * y_pred[:, 2:3, :, :]
            # Denormalize if needed (inludes inverse log-transform)
            y_pred_denorm = train_loader.dataset.denormalize(y_pred, data_type="y")
            y_true_denorm = train_loader.dataset.denormalize(y_true, data_type="y")
            
            # Gather predictions
            all_y_pred.append(y_pred_denorm.cpu().numpy())
            all_y_true.append(y_true_denorm.cpu().numpy())
            all_dates.extend(date)
        
        # Concatenate all batches
        y_pred_all = np.concatenate(all_y_pred, axis=0)
        y_true_all = np.concatenate(all_y_true, axis=0)
        all_dates = pd.to_datetime(all_dates)
        # Create xarray Dataset
        ds = xr.Dataset({
            'precipitation': (['time', 'lat', 'lon'], y_pred_all.squeeze())}, 
            coords={'time': all_dates,
                    'lat': data_loader.dataset.lat,
                    'lon': data_loader.dataset.lon,
            }
        )
        start_year = all_dates[0].year
        end_year = all_dates[-1].year
        # Save to NetCDF
        output_path = os.path.join(output_dir, f"inference_pred_{start_year}-{end_year}.nc")
        os.makedirs(output_dir, exist_ok=True)
        ds.to_netcdf(output_path)#, encoding={'complevel': 5})
        print(f"Saved predictions to {output_path}")

        # Also save ground truth
        ds_true = xr.Dataset({
            'precipitation': (['time', 'lat', 'lon'], y_true_all.squeeze())}, 
            coords={'time': all_dates,
                    'lat': data_loader.dataset.lat,
                    'lon': data_loader.dataset.lon,
            }
        )
        output_true_path = os.path.join(output_dir, f"inference_truth_{start_year}-{end_year}.nc")
        ds_true.to_netcdf(output_true_path)#, encoding={'complevel': 5})
        print(f"Saved ground truth to {output_true_path}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a downscaling model using PyTorch Lightning")
    # example usage: python train.py --config config_mse.yaml --model unet
    parser.add_argument("--config", type=str, required=True, help="The configuration YAML file")
    parser.add_argument("--model", type=str, required=True, help="Model to use: 'unet', 'vit'...")

    parser.add_argument("--inference", action="store_true", help="Run in inference mode")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    with open(os.path.join("configs", args.config), 'r') as f:
        config = yaml.safe_load(f)

    # Define loss and optimizer
    loss_name = config['training']['loss_type']
    if loss_name == 'mse':
        criterion = torch.nn.MSELoss()
        out_ch = 1
    elif loss_name == 'mae':
        criterion = torch.nn.L1Loss()
        out_ch = 1
    elif loss_name == 'nll':
        criterion = BernoulliGammaLoss()
        out_ch = 3  # ocurrence, shape_parameter, scale_parameter
        # check if target_normalize is None in config
        if config['data']['common_kwargs']['target_normalize'] is not None:
            raise ValueError("Target normalisation is not suitable when using NLL loss.")
    else:
        raise NotImplementedError(f"Loss type {loss_name} not implemented")
    
    # Data Module
    data_module = LitDataModule(
        config=config['data'].copy(),
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 0)
    )
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    if args.inference:
        test_loader = data_module.test_dataloader() 

    # Build model
    model_name = args.model
    model_kwargs = config['models'][args.model].copy()
    # add data specific parameters
    model_kwargs['output_shape'] = train_loader.dataset.output_shape
    model_kwargs['in_channels'] = train_loader.dataset.n_channels
    model_kwargs['out_channels'] = out_ch     
    
    if args.model == 'vit':
        model = DownscalingViT(**model_kwargs)
    elif args.model == 'unet':
        model = UNet(**model_kwargs)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    # Initialize model module
    model_module = LitModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        learning_rate=config['training']['learning_rate'],

    )
    # Setup logging and checkpointing
    exp_name = args.config.replace('.yaml', '') + f"_{args.model}"
    # if not inference or resume remove existing logs
    if not args.inference and not args.resume:
        log_dir = os.path.join("logs", exp_name)
        if os.path.exists(log_dir):
            import shutil
            print(f"Removing existing log directory {log_dir}")
            shutil.rmtree(log_dir)
    logger = TensorBoardLogger(save_dir="logs", name=exp_name, version="")
    logger.log_hyperparams(config)

    weights_path = os.path.join(config['training'].get('weights_dir', 'checkpoints/'), exp_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_path,
        filename='best',
        save_top_k=1,
        monitor='val_loss',
        save_last=True,
        mode='min'
    ) 
    last_checkpoint_path = os.path.join(
        checkpoint_callback.dirpath,
        "last.ckpt"
    )
    best_checkpoint_path = os.path.join(
        checkpoint_callback.dirpath,
        f"best.ckpt")

    # other callback could be added here, ex, early stopping, learning rate monitor, etc.
    torch.set_float32_matmul_precision('medium')
    if args.inference:  
        ckpt = best_checkpoint_path if os.path.exists(best_checkpoint_path) else last_checkpoint_path      
        if os.path.exists(ckpt):
            print("Loading model from checkpoint for inference...")
            model_module = LitModule.load_from_checkpoint(
                checkpoint_path=ckpt,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                learning_rate=config['training']['learning_rate'],
            )
            # implement inference logic here
            inference(model_module, 
                      test_loader, 
                      train_loader, 
                      os.path.join(config['training']['eval_out_dir'], exp_name))
        else:
            raise FileNotFoundError(f"No checkpoint found at {last_checkpoint_path} for inference.")
    else:
        if not args.resume:
            # remove ckpt files if exist
            if os.path.exists(last_checkpoint_path):
                print(f"Removing existing checkpoint {last_checkpoint_path}")
                os.remove(last_checkpoint_path)
            if os.path.exists(best_checkpoint_path):
                print(f"Removing existing checkpoint {best_checkpoint_path}")
                os.remove(best_checkpoint_path)
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=config['training']['epochs'],
            callbacks=[checkpoint_callback], # add more callbacks to the list if needed
            logger=logger,
            log_every_n_steps=1,
            devices='auto',
            accelerator='auto',
            #precision='16-mixed', # '16' or '32'
            accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        )

        # Start training
        trainer.fit(model_module, train_loader, val_loader, 
            ckpt_path=last_checkpoint_path if args.resume else None
        )