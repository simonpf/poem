"""
poem.data.era5
==============

Provides functionality for downloading ERA5 data for poem simulations.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from pansat import TimeRange
from pansat.products.reanalysis.era5 import ERA5Hourly



def download_atmosphere_data(
        time: datetime,
        roi: Optional[Tuple[float, float, float, float]] = None
) -> List[Path]:
    variables = [
        "geopotential",
        "temperature",
        "specific_humidity",
        "specific_cloud_ice_water_content",
        "specific_snow_water_content",
        "specific_cloud_liquid_water_content",
        "specific_rain_water_content"
    ]
    prod = ERA5Hourly(
        "pressure_levels",
        variables=variables,
        domain=roi
    )
    recs = prod.get(TimeRange(time))
    return recs


def download_surface_data(
        time: datetime,
        roi: Optional[Tuple[float, float, float, float]] = None
) -> List[Path]:
    variables = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "sea_surface_temperature",
    ]
    prod = ERA5Hourly(
        "surface",
        variables=variables,
        domain=roi
    )
    recs = prod.get(TimeRange(time))
    return recs
