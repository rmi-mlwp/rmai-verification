from typing import List
import logging

import xskillscore as xs
import xarray as xr 

LOG = logging.getLogger(__name__)

def _chunks(avg_dim: List[str]) -> dict:
    """
    Create a dictionary of chunks for xskillscore functions.
    (chunking not possible along avg_dim, needs to be investigated further)
    """
    return {dim: -1 for dim in avg_dim}

def rmse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    LOG.info("Using chunks: %s for RMSE calculation.", chunks)
    return xs.rmse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def mse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    LOG.info("Using chunks: %s for MSE calculation.", chunks)
    return xs.mse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def bias(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    LOG.info("Using chunks: %s for bias calculation.", chunks)
    return xs.me(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)
