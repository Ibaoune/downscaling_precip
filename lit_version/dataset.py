from logging import warning
import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd

class DownscalingDataset(Dataset):
    def __init__(
        self,
        mode="train",
        variables=("z", "q", "u", "v", "t"),
        levels=("1000", "850", "700", "500"),
        input_normalize=None,
        target_normalize=None,
        return_date=False,
        extent: list = [-10, 0, 29, 36],
        data_path: list = None,
        start_date: str = None,
        end_date: str = None,
        stats_path: str = "data_stats/",
        input_ds_name: str = "era5",
        target_ds_name: str = "mswp",
        log_transform: bool = False,
    ):
        print(f"[{mode.capitalize()} Dataset]")
        self.extent = extent
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.stats_path = stats_path
        os.makedirs(self.stats_path, exist_ok=True)
        self.input_normalize = input_normalize
        self.target_normalize = target_normalize
        self.log_transform = log_transform
        self.return_date = return_date
        self.input_ds_name = input_ds_name.lower()
        self.target_ds_name = target_ds_name.lower()
        #
        if self.target_ds_name in ["lmdz", "era5"]:
            self.target_unit = "kg/m2/s"
        elif self.target_ds_name in ["mswp", "ter"]:
            self.target_unit = "mm/day"
        elif self.target_ds_name in ["imerg"]:
            self.target_unit = "mm/hr"
        else:
            raise ValueError(f"Unknown target dataset name: {self.target_ds_name}")
        # get input data
        self.x_data = self._get_inputs(variables, levels)
        self.x_mean, self.x_std = self._compute_stats(
            self.x_data, data_type="x", 
            normalization=self.input_normalize,
            ds_name=self.input_ds_name
            )
        self.n_channels = len(variables) * len(levels)
        # get target data
        self.y_data = self._get_targets()
        self.y_mean, self.y_std = self._compute_stats(
            self.y_data, data_type="y", 
            normalization=self.target_normalize,
            ds_name=self.target_ds_name
            )
        self.output_shape = (len(self.lat), len(self.lon))

        assert self.x_data.sizes["time"] == self.y_data.sizes["time"], \
            "Input and target time dimensions do not match"
        
    def _get_inputs(self, variables, levels):
        ds = xr.open_mfdataset(
            self.data_path['input'],
            combine="by_coords",
            preprocess=rename_valid_time,
        )
        ds = self.slice_ds(ds)
        # select levels
        ds = ds.sel(level=levels)
        # select variables
        x_data = ds[variables]
        # check time frequency, should be daily
        try:
            freq = xr.infer_freq(x_data.time.to_index())
        except ValueError:
            print("Could not infer time frequency, trying with first 100 timestamps.")
            freq = xr.infer_freq(x_data.time[:100].to_index())
            if freq is None:
                print("Still could not infer frequency, assuming daily frequency.")
                freq = 'D'
        if freq != 'D':
            if not freq.endswith('h'):
                raise ValueError(f"Input data time frequency is {freq}, expected 'D'")
            else:
                # resample to daily
                x_data = x_data.resample(time='1D').mean()
        else:
            print(f"Input data time frequency: {freq}")
        # load() all data into memory, remove in case of GPU memory issues
        return x_data.load()

    def _get_targets(self):
        ds = xr.open_mfdataset(
            self.data_path["target"],
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine="nested",
            concat_dim="time",
            preprocess=rename_valid_time,
        )
        # ds contains only one variable name, pr/precip/precipitation, get it
        var = list(ds.data_vars.keys())[0]
        ds = self.slice_ds(ds)
        # check time frequency, should be daily
        freq = xr.infer_freq(ds.time.to_index())
        if freq != 'D':
            if not freq.endswith('h'):
                raise ValueError(f"Target data time frequency is {freq}, expected 'D'")
            else:
                # resample to daily
                print(f"Resampling target data from {freq} to daily.")
                ds = ds.resample(time='1D').sum() # mean or sum ?
        else:
            print(f"Target data time frequency: {freq}")
        # get precipitation unit
        self.precip_unit = ds[var].attrs.get("units", None)
        if self.target_unit != self.precip_unit:
            warning(f"Target dataset unit ({self.precip_unit}) does not match expected unit ({self.target_unit}), Please check.")
        else:
            print(f"Target data units: {self.precip_unit}")
        # select variable
        y_data = ds[var]
        # convert to mm/day
        y_data = self._precip_unit_conversion(y_data)
        if self.log_transform:
            print("Applying log(1 + x) transform to target data")
            y_data = np.log1p(y_data)
        self.time = y_data.time.values
        self.lon = y_data.lon.values
        self.lat = y_data.lat.values
        # load() all data into memory, remove in case of GPU memory issues
        return y_data.load()

    def slice_ds(self, ds):
        # standardise coordinates
        ds = ds.rename(
            {k: v for k, v in {
                "latitude": "lat",
                "longitude": "lon"
            }.items() if k in ds.dims}
        )
        # sort by time in case of concatenation issues
        ds = ds.sortby("time")
        # Ensure descending latitude
        if ds.lat[0] < ds.lat[-1]:
            ds = ds.reindex(lat=list(reversed(ds.lat)))
        # Adjust longitude coordinates to [-180, 180] if needed
        lon = ds.lon.values
        if not np.any(lon < 0) or np.any(lon > 180):
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
            ds = ds.sortby('lon')

        # sel extent
        ds = ds.sel(
            lon=slice(self.extent[0], self.extent[1]),
            lat=slice(self.extent[3], self.extent[2]),
        )
        # check level dimension order
        print(f"Dataset dimensions: {ds.dims}")
        return ds.sel(time=slice(self.start_date, self.end_date))
    
    def _precip_unit_conversion(self, ds):
        if self.target_unit in ["kg/m2/s", "kg/m^2/s", "kg/m²/s"]:
            ds = ds * 86400.0
        elif self.target_unit in ["mm/hr", "mm h-1"]:
            ds = ds * 24.0
        elif self.target_unit in ["mm/day", "mm d-1"]:
            return ds
        else:
            raise ValueError(f"Unknown precipitation unit: {self.target_unit}")
        print(f"  Converting {self.target_unit} -> mm/day")
        return ds
    
    def _compute_stats(self, ds, data_type="x", normalization=None, ds_name=None):
        if normalization is None or normalization == "per_day":
            return None, None
        
        suffix = "_log" if self.log_transform and data_type == "y" else ""
        stats_file = os.path.join(self.stats_path, 
                                  f"{data_type}_stats_{normalization}_{ds_name}{suffix}.npz")
        if self.mode == "train":
            if normalization == "per_channel":
                # stats per var and level 
                mean_ds = np.concatenate(
                    [ds[var].mean(dim=("time", "lat", "lon")).values for var in ds.data_vars], 
                    axis=0)[:, np.newaxis, np.newaxis]
                std_ds = np.concatenate(
                    [ds[var].std(dim=("time", "lat", "lon")).values for var in ds.data_vars], 
                    axis=0)[:, np.newaxis, np.newaxis]
            elif normalization == "global":
                if data_type == "x":
                    mean_ds = np.mean([ds[var].mean().values for var in ds.data_vars])
                    std_ds = np.mean([ds[var].std().values for var in ds.data_vars])
                else:
                    mean_ds = ds.mean().values
                    std_ds = ds.std().values
            else:
                raise ValueError(f"Unknown normalization type: {normalization}")
            # save stats as npy
            np.savez(stats_file, mean=mean_ds, std=std_ds)
            print(f"Saved training data stats to {self.stats_path}")
            return mean_ds, std_ds
        else:
            # load stats
            if not os.path.exists(stats_file):
                raise FileNotFoundError(f"Stats file not found: {stats_file}. "
                "Make sure to run on training mode first to compute and save the stats.")
            print(f"Loading data stats from {self.stats_path}")
            stats = np.load(stats_file, allow_pickle=True)
            return stats['mean'], stats['std']

    def denormalize(self, data, data_type="x"):
        if data_type == "x":
            if self.input_normalize == "per_day":
                raise NotImplementedError("Error, remove this exception, "
                "if day stats are passed in the getitem function, " \
                "then use them here for denormalization.")
            mean = self.x_mean
            std = self.x_std
        else:
            mean = self.y_mean
            std = self.y_std
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float()
        if data_type == "x" and self.input_normalize is None:
            denorm_data = data
        elif data_type == "y" and self.target_normalize is None:
            denorm_data = data
        else:
            denorm_data = data * (std + 1e-8) + mean
        # inverse log transform if applied
        if self.log_transform and data_type == "y":
            denorm_data = torch.expm1(denorm_data)
        return denorm_data


    def __len__(self):
        return self.x_data.sizes["time"]

    def __getitem__(self, idx):
        # get input data
        date = self.time[idx]
        dayofyear = pd.to_datetime(date).dayofyear
        x_data_idx = self.x_data.isel(time=idx)
        # Stack variables into a single array: (variable x level, lat, lon)
        x_arr = np.concatenate(
            [x_data_idx[var].values for var in x_data_idx.data_vars], 
            axis=0)
        x = torch.from_numpy(x_arr)        
        y = torch.from_numpy(self.y_data.isel(time=idx).values)

        if self.input_normalize is not None:
            if self.input_normalize == "per_day":
                # per day per channel normalization
                mean = x.mean(dim=(1,2), keepdim=True)
                std  = x.std(dim=(1,2), keepdim=True)
                x = (x - mean) / (std + 1e-8)
            else:
                x = (x - self.x_mean) / (self.x_std + 1e-8)

        if self.target_normalize is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)
        x, y, = x.float(), y.float()
        # seasonal forcing
        cosin_time = np.cos(2 * np.pi * dayofyear / 365)
        sin_time = np.sin(2 * np.pi * dayofyear / 365)
        seasonal_forcing = torch.tensor([cosin_time, sin_time], dtype=torch.float32)
        if self.return_date:
            return x, y, seasonal_forcing, str(date)
        return x, y, seasonal_forcing

def rename_valid_time(ds):
    if "valid_time" in ds.dims:
        return ds.rename({"valid_time": "time"})
    elif "counter_time" in ds.dims:
        return ds.rename({"counter_time": "time"})
    return ds

if __name__ == "__main__":
    # Example usage
    mode = "train"
    with open('configs/config_nll_global_x.yaml', 'r') as f:        
        config = yaml.safe_load(f)
    kwargs = config['data']['common_kwargs']
    kwargs.update(config['data'][mode])
    dataset = DownscalingDataset(**kwargs)

    print(f"Dataset length: {len(dataset)}")
    x, y, _ = dataset[10]
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    # plot 
    plt.subplot(1, 2, 1)
    plt.imshow(x[-5].numpy(), cmap='viridis', origin='upper')
    plt.title('Input Variable (Channel 0)')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(y.numpy(), cmap='viridis', origin='upper')
    plt.title('Target Variable')
    plt.colorbar()
    plt.savefig('sample_plot.png')