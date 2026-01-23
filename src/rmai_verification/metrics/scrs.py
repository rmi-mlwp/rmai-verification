from typing import List
import scores
import xarray as xr

def rmse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return scores.continuous.rmse(fcst, obs, reduce_dims=avg_dim)

def mse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return scores.continuous.mse(fcst, obs, reduce_dims=avg_dim)

def bias(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return scores.continuous.additive_bias(fcst, obs, reduce_dims=avg_dim)

def crps(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    print("Calculating CRPS")
    print("fcst: ", fcst)
    print("obs: ", obs)
    return scores.probability.crps_for_ensemble(fcst, obs, ensemble_member_dim="ensemble_member", reduce_dims=avg_dim)

