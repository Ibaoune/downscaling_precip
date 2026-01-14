#!/bin/bash
# example usage: bash inference.sh vit
# make sure not to remove "--inference" flag else it will run training 
# instead of inference and will overwrite the model weights
python train.py --config config_mse_global_x.yaml --model $1 --inference
python train.py --config config_mse_global_y.yaml --model $1 --inference
python train.py --config config_mse_log.yaml --model $1 --inference
python train.py --config config_mse_no_norm.yaml --model $1 --inference
python train.py --config config_mse_per_channel.yaml --model $1 --inference
python train.py --config config_mse_per_day.yaml --model $1 --inference
python train.py --config config_mse_log_global_y.yaml --model $1 --inference
python train.py --config config_nll_global_x.yaml --model $1 --inference
python train.py --config config_nll_log.yaml --model $1 --inference
python train.py --config config_nll_no_norm.yaml --model $1 --inference
python train.py --config config_nll_per_channel.yaml --model $1 --inference
python train.py --config config_nll_per_day.yaml --model $1 --inference