from typing import List

import xskillscore as xs
import xarray as xr 


def _chunks(avg_dim: List[str]) -> dict:
    """
    Create a dictionary of chunks for xskillscore functions.
    (chunking not possible along avg_dim, needs to be investigated further)
    """
    return {dim: -1 for dim in avg_dim}

def rmse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    return xs.rmse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def mse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    return xs.mse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def bias(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    return xs.me(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)
