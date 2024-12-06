"""
poem.emulator.training_data
===========================

Provides dataset interfaces for loading emulator training data.
"""
from math import ceil
from pathlib import Path
import struct
from typing import Union

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import xarray as xr

from pytorch_retrieve.modules.input import StandardizationLayer

def read_profile_file(
        path: Path
):
    """
    Read simulation input data file.

    Args:
        path: A path object pointing to the file to read.

    Return:
        An xarray.Dataset containing the data.
    """
    with open(path, "rb") as stream:
        n_profiles, n_heights = struct.unpack('ii', stream.read(8))
        n_layers = n_heights - 1
        size = n_profiles * n_heights
        layer_size = n_profiles * (n_heights - 1)
        latitude = np.fromfile(stream, dtype="f", count=n_profiles)
        longitude = np.fromfile(stream, dtype="f", count=n_profiles)
        height = np.fromfile(stream, dtype="f", count=size).reshape(n_heights, n_profiles)
        temperature = np.fromfile(stream, dtype="f", count=size).reshape(n_heights, n_profiles)
        pressure = np.fromfile(stream, dtype="f", count=size).reshape(n_heights, n_profiles)

        humidity = np.fromfile(stream, dtype="f", count=layer_size).reshape(n_layers, n_profiles)
        clwc = np.fromfile(stream, dtype="f", count=layer_size).reshape(n_layers, n_profiles)
        plwc = np.fromfile(stream, dtype="f", count=layer_size).reshape(n_layers, n_profiles)
        tiwc = np.fromfile(stream, dtype="f", count=layer_size).reshape(n_layers, n_profiles)

        humidityg = np.zeros_like(pressure)
        clwcg = np.zeros_like(pressure)
        plwcg = np.zeros_like(pressure)
        tiwcg = np.zeros_like(pressure)

        humidityg[1:-1] = 0.5 * (humidity[1:] + humidity[:-1])
        humidityg[0] = humidity[0]
        humidityg[-1] = humidity[-1]

        clwcg[1:-1] = 0.5 * (clwc[1:] + clwc[:-1])
        clwcg[0] = clwc[0]
        clwcg[-1] = clwc[-1]

        plwcg[1: -1] = 0.5 * (plwc[1:] + plwc[:-1])
        plwcg[0] = plwc[0]
        plwcg[-1] = plwc[-1]

        tiwcg[1: -1] = 0.5 * (tiwc[1:] + tiwc[:-1])
        tiwcg[0] = tiwc[0]
        tiwcg[-1] = tiwc[-1]

        sst = np.fromfile(stream, dtype="f", count=n_profiles)
        t2m = np.fromfile(stream, dtype="f", count=n_profiles)
        wsp = np.fromfile(stream, dtype="f", count=n_profiles)
        n0_rain = np.fromfile(stream, dtype="f", count=n_profiles)
        mu_rain = np.fromfile(stream, dtype="f", count=n_profiles)
        n0_snow = np.fromfile(stream, dtype="f", count=n_profiles)
        rho_snow = np.fromfile(stream, dtype="f", count=n_profiles)
        eia = np.fromfile(stream, dtype="f", count=n_profiles)

    return xr.Dataset({
        "latitude": (("profiles",), latitude),
        "longitude": (("profiles",), longitude),
        "height": (("levels", "profiles"), height),
        "temperature": (("levels", "profiles"), temperature),
        "pressure": (("levels", "profiles"), pressure),
        "humidity": (("levels", "profiles"), humidityg),
        "clwc": (("levels", "profiles"), clwcg),
        "plwc": (("levels", "profiles"), plwcg),
        "tiwc": (("levels", "profiles"), tiwcg),
        "sea_surface_temperature": (("profiles",), sst),
        "two_meter_temperature": (("profiles",), t2m),
        "wind_speed": (("profiles",), wsp),
        "n0_rain": (("profiles",), n0_rain),
        "mu_rain": (("profiles",), mu_rain),
        "n0_snow": (("profiles",), n0_snow),
        "rho_snow": (("profiles",), rho_snow),
        "earth_incidence_angle": (("profiles",), eia),
    })


def read_sim_file(
        path: Path
):
    """
    Read simulation output file.

    Args:
        path: A path object pointing to a simluation output file to read.

    Return:

        An xarray.Dataset containing the simulation results.
    """
    path = Path(path)
    with open(path, 'rb') as stream:
        n_profiles = np.fromfile(stream, sep='', count=1, dtype='i')[0]
        n_channels  = np.fromfile(stream, sep='', count=1, dtype='i')[0]
        n_layers = np.fromfile(stream, sep='', count=1, dtype='i')[0]
        brightness_temperatures = np.fromfile(
            stream,
            count=n_profiles*n_channels,
            dtype='f'
        ).reshape(n_channels, n_profiles)
        earth_incidence_angle = np.fromfile(
            stream,
            count=n_profiles,
            dtype='f'
        )

    return xr.Dataset({
        "brightness_temperatures": (("channels", "profiles"), brightness_temperatures),
        "earth_incidence_angle": (("profiles",), earth_incidence_angle)
    })




class TrainingData(Dataset):
    """
    Loads training data from AMSR2 simulations.
    """

    def __init__(
            self,
            path: Union[str, Path],
            batch_size: int = 256
    ):
        path = Path(path)
        if path.is_dir():
            self.training_files = np.array(sorted(list(path.glob("poem_emulator*.nc"))))
        else:
            self.training_files = np.array([path])
        self._training_data = None
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed=42)

    @property
    def training_data(self):
        if self._training_data is None:
            self._training_data = xr.concat(
                [xr.load_dataset(training_file) for training_file in self.training_files],
                dim="profiles"
            ).transpose("profiles", "levels", "channels")
        return self._training_data

    def init_worker_fn(self, w_id: int) -> None:
        """
        Initialize worker for multi-process data loading.
        """
        worker_info = torch.utils.data.get_worker_info()
        n_workers = worker_info.num_workers
        self.training_files = self.training_files[w_id::n_workers]

    def __len__(self) -> int:
        """
        The number of samples in the training dataset.
        """
        return ceil(self.training_data.profiles.size / self.batch_size)

    def __getitem__(self, index: int):
        """
        Load batch of training data.
        """
        batch_start = self.batch_size * index
        batch_end = batch_start + self.batch_size

        batch = self.training_data[{"profiles": slice(batch_start, batch_end)}]
        n_channels = batch.channels.size
        channels = self.rng.integers(0, n_channels, size=batch.profiles.size)
        batch = batch[{"channels": xr.DataArray(channels, dims={"profiles": batch.profiles})}]

        humidity = torch.tensor(batch.humidity.data)
        clwc = torch.tensor(batch.clwc.data)
        plwc = torch.tensor(batch.plwc.data)
        tiwc = torch.tensor(batch.tiwc.data)

        sst = torch.tensor(batch.sea_surface_temperature.data[..., None])
        t2m = torch.tensor(batch.two_meter_temperature.data[..., None])
        wsp = torch.tensor(batch.wind_speed.data[..., None])

        n0_rain = torch.tensor(batch.n0_rain.data[..., None])
        mu_rain = torch.tensor(batch.mu_rain.data[..., None])
        n0_snow = torch.tensor(batch.n0_snow.data[..., None])
        rho_snow = torch.tensor(batch.rho_snow.data[..., None])
        eia = torch.tensor(batch.earth_incidence_angle.data[..., None])

        channel = nn.functional.one_hot(torch.tensor(channels), num_classes=n_channels)

        input_features = torch.cat(
            [humidity, clwc, plwc, tiwc, sst, t2m, wsp, n0_rain, mu_rain, n0_snow, rho_snow, eia, channel],
            1
        )
        brightness_temperatures = torch.tensor(batch.brightness_temperatures.data)
        return input_features, brightness_temperatures


class ScoreModel(Dataset):
    """
    Loads training data from AMSR2 simulations.
    """

    def __init__(
            self,
            path: Union[str, Path],
            stats_file: Path,
            batch_size: int = 256
    ):
        path = Path(path)
        if path.is_dir():
            self.training_files = np.array(sorted(list(path.glob("poem_emulator*.nc"))))
        else:
            self.training_files = np.array([path])
        self._training_data = None
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed=42)

        stats_file = Path(stats_file)
        print(stats_file.parent.parent)
        self.n_layer = StandardizationLayer(
            "input",
            162,
            stats_path=stats_file.parent.parent.parent
        )
        self.n_layer.load_stats(stats_file)

    @property
    def training_data(self):
        if self._training_data is None:
            self._training_data = xr.concat(
                [xr.load_dataset(training_file) for training_file in self.training_files],
                dim="profiles"
            ).transpose("profiles", "levels", "channels")
        return self._training_data

    def init_worker_fn(self, w_id: int) -> None:
        """
        Initialize worker for multi-process data loading.
        """
        worker_info = torch.utils.data.get_worker_info()
        n_workers = worker_info.num_workers
        self.training_files = self.training_files[w_id::n_workers]

    def __len__(self) -> int:
        """
        The number of samples in the training dataset.
        """
        return ceil(self.training_data.profiles.size / self.batch_size)

    def __getitem__(self, index: int):
        """
        Load batch of training data.
        """
        batch_start = self.batch_size * index
        batch_end = batch_start + self.batch_size

        batch = self.training_data[{"profiles": slice(batch_start, batch_end)}]
        n_channels = batch.channels.size
        channels = self.rng.integers(0, n_channels, size=batch.profiles.size)
        batch = batch[{"channels": xr.DataArray(channels, dims={"profiles": batch.profiles})}]

        humidity = torch.tensor(batch.humidity.data)
        clwc = torch.tensor(batch.clwc.data)
        plwc = torch.tensor(batch.plwc.data)
        tiwc = torch.tensor(batch.tiwc.data)

        sst = torch.tensor(batch.sea_surface_temperature.data[..., None])
        t2m = torch.tensor(batch.two_meter_temperature.data[..., None])
        wsp = torch.tensor(batch.wind_speed.data[..., None])

        n0_rain = torch.tensor(batch.n0_rain.data[..., None])
        mu_rain = torch.tensor(batch.mu_rain.data[..., None])
        n0_snow = torch.tensor(batch.n0_snow.data[..., None])
        rho_snow = torch.tensor(batch.rho_snow.data[..., None])
        eia = torch.tensor(batch.earth_incidence_angle.data[..., None])

        channel = nn.functional.one_hot(torch.tensor(channels), num_classes=n_channels)

        input_features = torch.cat(
            [humidity, clwc, plwc, tiwc, sst, t2m, wsp, n0_rain, mu_rain, n0_snow, rho_snow, eia, channel],
            1
        )
        x_n = self.n_layer(input_features)
        noise_level = torch.rand(1)
        corrupted = (
            x_n + noise_level * (1.0 - torch.rand(x_n.shape))
        )
        return corrupted[:, :147], x_n[:, :147]
