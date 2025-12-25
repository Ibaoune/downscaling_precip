import os
import time
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import yaml

class DownscalingDataset(Dataset):
    def __init__(
        self,
        mode="train",
        variables=("z", "q", "u", "v", "t"),
        levels=("1000", "850", "700", "500"),
        input_normalize=None,
        target_normalize=None,
        return_metadata=False,
        extent: list = [-10, 0, 29, 36],
        data_path: list = None,
        start_date: str = None,
        end_date: str = None,
        stats_path: str = "data_stats/",
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
        self.return_metadata = return_metadata
        # get input data
        self.x_data = self._get_inputs(variables, levels)
        self.x_mean, self.x_std = self._compute_stats(
            self.x_data, data_type="x", 
            normalization=self.input_normalize
            )
        self.n_channels = len(variables) * len(levels)
        # get target data
        self.y_data = self._get_targets()
        self.y_mean, self.y_std = self._compute_stats(
            self.y_data, data_type="y", 
            normalization=self.target_normalize
            )
        self.output_shape = (self.y_data.sizes["lat"], self.y_data.sizes["lon"])

        assert self.x_data.sizes["time"] == self.y_data.sizes["time"], \
            "Input and target time dimensions do not match"

    def _compute_stats(self, ds, data_type="x", normalization=None):
        if normalization is None:
            return 0, 1
        if self.mode == "train":
            if normalization == "per_channel":
                # stats per var and level 
                mean_ds = np.concatenate(
                    [ds[var].mean(dim=("time", "lat", "lon")).values for var in ds.data_vars], 
                    axis=0)[:, np.newaxis, np.newaxis]
                std_ds = np.concatenate(
                    [ds[var].std(dim=("time", "lat", "lon")).values for var in ds.data_vars], 
                    axis=0)[:, np.newaxis, np.newaxis]
            else:
                mean_ds, std_ds = ds.mean().values, ds.std().values
            # save stats as npy
            np.savez(
                os.path.join(self.stats_path, f"{data_type}_stats_{normalization}.npz"),
                mean=mean_ds, std=std_ds)
            print(f"Saved training data stats to {self.stats_path}")
            return mean_ds, std_ds
        else:
            # load stats
            stats = np.load(os.path.join(self.stats_path, f"{data_type}_stats_{normalization}.npz"), allow_pickle=True)
            print(f"Loading data stats from {self.stats_path}")
            return stats['mean'], stats['std']

    def _denormalize(self, data, data_type="x"):
        if data_type == "x":
            mean = self.x_mean
            std = self.x_std
        else:
            mean = self.y_mean
            std = self.y_std
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float()
        return data * (std + 1e-8) + mean

    def _get_inputs(self, variables, levels):
        ds = xr.open_mfdataset(
            self.data_path['input'],
            combine="by_coords"
        )
        ds = self.slice_ds(ds)
        # select levels
        ds = ds.sel(level=levels)
        # select variables
        x_data = ds[variables]
        # check time frequency, should be daily
        freq = xr.infer_freq(x_data.time.to_index())
        if freq != 'D':
            if not freq.endswith('h'):
                raise ValueError(f"Input data time frequency is {freq}, expected 'D'")
            else:
                # resample to daily
                x_data = x_data.resample(time='1D').mean()
        else:
            print(f"Input data time frequency: {freq}")
        self.time = x_data.time.values
        self.lon = x_data.lon.values
        self.lat = x_data.lat.values
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
        print(f"Target variable units: {self.precip_unit}")
        y_data = ds[var]
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
        return ds.sel(time=slice(self.start_date, self.end_date))
    
    def __len__(self):
        return self.x_data.sizes["time"]

    def __getitem__(self, idx):
        # get input data
        x_data_idx = self.x_data.isel(time=idx)
        # Stack variables into a single array: (variable x level, lat, lon)
        x_arr = np.concatenate(
            [x_data_idx[var].values for var in x_data_idx.data_vars], 
            axis=0)
        x = torch.from_numpy(x_arr).float()
        
        y = torch.from_numpy(
            self.y_data.isel(time=idx).values
        ).float()

        if self.input_normalize == "per_day":
            # per day per channel normalization
            mean = x.mean(dim=(1,2), keepdim=True)
            std  = x.std(dim=(1,2), keepdim=True)
            x = (x - mean) / (std + 1e-8)
        else:
            x = (x - self.x_mean) / (self.x_std + 1e-8)
        
        y = (y - self.y_mean) / (self.y_std + 1e-8)

        if not self.return_metadata:
            return x, y

        return x, y, {
            "time": self.time[idx],
            "lon": self.lon,
            "lat": self.lat,
        }



def rename_valid_time(ds):
    if "valid_time" in ds.dims:
        return ds.rename({"valid_time": "time"})
    return ds

if __name__ == "__main__":
    # Example usage
    # dataset = DownscalingDataset(
    #     variables=["z", "q", "u", "v", "t"],
    #     levels=[1000, 850, 700, 500],
    #     extent=[-10, 0, 29, 36],
    #     data_path={
    #         "input": "data/predictors/er5_*.nc",
    #         #"target": "data/predictands/MSWP_1979_2014.nc"
    #         "target": "data/predictors/era5_pr/ERA_1979_2014.nc"
    #     },
    #     start_date="2000-01-01",
    #     end_date="2010-12-31",
    #     return_metadata=False,
    # )
    # or using config file
    mode = "train"
    with open('configs/config_1.yaml', 'r') as f:        
        config = yaml.safe_load(f)
    kwargs = config['data']['common_kwargs']
    kwargs.update(config['data'][mode])
    dataset = DownscalingDataset(**kwargs)

    print(f"Dataset length: {len(dataset)}")
    x, y = dataset[10]
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