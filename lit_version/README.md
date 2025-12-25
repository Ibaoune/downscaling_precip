# Organisation of the repository
This repository is organized as follows:

#### Organisation of the repository
This repository is organized as follows:
- `train.py`: Main training script, uses `lit_module`, `models`, and `BernoulliGammaLoss`.
- `lit_module.py`: Defines the PyTorch Lightning module for training and validation, contains two classes:
  - `LitDataModule`: Data loaders setup, calls `dataset.DownscalingDataset`.
  - `LitModule`: Model training and validation logic.
- `models/`: Directory containing model architectures: U-Net, ViT, also contains `losses.py` for loss functions.
- `dataset.py`: Defines `DownscalingDataset` class for loading and preprocessing data.

#### Usage, training:
To train a model, run the `train.py` script with the desired configuration and model type. For example:
```bash
python train.py --config config_mse.yaml --model unet
``` 
This will train a U-Net model using the MSE loss function as specified in `config_mse.yaml`. Weights and logs will be saved in the `lit_version/weights` and `lit_version/logs` directories, respectively.

Make sure `TensorBoard` is installed to monitor training progress:
```bash
pip install tensorboard
```
To launch TensorBoard, run:
```bash
tensorboard --logdir=logs
```

#### Usage, inference:
After training, the model can be used for inference. The predictions and ground truth will be saved in a directory as specified in the configuration file. Ground truth and predictions are saved in NetCDF format for later analysis.
```bash
python train.py --config config_mse.yaml --model unet --inference
```