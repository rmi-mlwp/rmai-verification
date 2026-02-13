from typing import List

import xskillscore as xs
import xarray as xr 


def _chunks(avg_dim: List[str]) -> dict:
    """
    Create a dictionary of chunks for xskillscore functions.
    (chunking along avg_dim does not work,)
    """
    return {dim: -1 for dim in avg_dim}

def rmse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    return xs.rmse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def mse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    print("Using xskillscore mse function")
    print("fcst:", fcst)
    print()
    print("Size of fcst", fcst.nbytes / 1024 / 1024 / 1024, "GiB")
    print()
    print("obs", obs)
    print()
    print("Size of obs", obs.nbytes / 1024 / 1024 / 1024, "GiB")
    print()
    print("avg_dim:", avg_dim)
    print()
    print("skipna:", skipna)
    print()
    chunks = _chunks(avg_dim)
    print("chunks:", chunks)
    return xs.mse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def bias(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    return xs.me(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

