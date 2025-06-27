from typing import List
from scores.continuous import mse, mean_error 
from scores.continuous.correlation import pearsonr
import xarray as xr
import numpy as np

def act(fcst: xr.Dataset | xr.DataArray, clim: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return np.sqrt(mse(fcst, clim, reduce_dims = avg_dim) - mean_error(fcst, clim, reduce_dims=avg_dim)**2)

def nfa(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, clim: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return act(fcst, clim, avg_dim)/act(obs, clim, avg_dim)

def acc(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, clim: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    fcst_anomaly = fcst - clim
    obs_anomaly = obs - clim
    # scores.continuous.correlation.pearsonr only accepts DataArrays
    assert type(fcst_anomaly) == type(obs_anomaly), "Observations and forecast need to be of same type." 
    is_dataset = False
    if isinstance(fcst_anomaly, xr.Dataset):
        is_dataset = True
        fcst_anomaly = fcst_anomaly.to_dataarray(dim='tmp_dim')
        obs_anomaly = obs_anomaly.to_dataarray(dim='tmp_dim')
    pr = pearsonr(fcst_anomaly, obs_anomaly, reduce_dims=avg_dim)
    if is_dataset:
        pr = pr.to_dataset(dim='tmp_dim')
    return pr
