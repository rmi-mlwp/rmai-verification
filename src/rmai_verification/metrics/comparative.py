import logging

import numpy as np
import xarray as xr
import xskillscore as xs

from typing import List
from scipy.interpolate import griddata
from .diagnostic import spread

LOG = logging.getLogger(__name__)

def _chunks(avg_dim: List[str]) -> dict:
    """
    Create a dictionary of chunks for xskillscore functions.
    (chunking not possible along avg_dim, needs to be investigated further)
    """
    return {dim: -1 for dim in avg_dim}

def rmse(obs: xr.Dataset | xr.DataArray, fcst: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    LOG.info("Using chunks: %s for RMSE calculation.", chunks)
    return xs.rmse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def mse(obs: xr.Dataset | xr.DataArray, fcst: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    LOG.info("Using chunks: %s for MSE calculation.", chunks)
    return xs.mse(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def bias(obs: xr.Dataset | xr.DataArray, fcst: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    chunks = _chunks(avg_dim)
    LOG.info("Using chunks: %s for bias calculation.", chunks)
    return xs.me(fcst.chunk(chunks), obs.chunk(chunks), avg_dim, skipna=skipna)

def crps(obs: xr.Dataset | xr.DataArray, fcst: xr.Dataset | xr.DataArray, avg_dim: List[str], member_dim: str = "ensemble", skipna: bool = True) -> xr.Dataset | xr.DataArray:
    N = fcst.sizes[member_dim]
    assert N > 1, "Ensemble size must be greater than 1."

    # MAE term
    mae = np.abs(fcst - obs)
    if skipna:
        mae_mean = mae.mean(dim=member_dim, skipna=True)
    else:
        mae_mean = mae.mean(dim=member_dim)

    # Spread term
    xi = fcst
    xj = fcst.rename({member_dim: f"{member_dim}_"})
    pairwise = np.abs(xi - xj)  # shape: (..., N, N)
    if skipna:
        pair_sum = pairwise.sum(dim=[member_dim, f"{member_dim}_"], skipna=True)
    else:
        pair_sum = pairwise.sum(dim=[member_dim, f"{member_dim}_"])
    spread = pair_sum / (N * (N - 1))

    # Calculate point-wise CRPS
    point_crps = mae_mean - 0.5 * spread

    LOG.info("Computing fair CRPS (alpha=1), averaging over dims: %s", avg_dim)

    if skipna:
        return point_crps.mean(dim=avg_dim, skipna=True)
    else:
        return point_crps.mean(dim=avg_dim)


def skill(obs: xr.Dataset | xr.DataArray, fcst: xr.Dataset | xr.DataArray, avg_dim: List[str], member_dim: str = "ensemble", skipna: bool = True) -> xr.Dataset | xr.DataArray:
    """
    RMSE of the ensemble mean vs observations.
    """
    ens_mean = fcst.mean(dim=member_dim, skipna=skipna) if skipna else fcst.mean(dim=member_dim)
    return rmse(obs, ens_mean, avg_dim, skipna=skipna)

def SSR(obs: xr.Dataset | xr.DataArray, fcst: xr.Dataset | xr.DataArray, avg_dim: List[str], member_dim: str = "ensemble", skipna: bool = True) -> xr.Dataset | xr.DataArray: 
    """
    Spread Skill Ratio (SSR = Spread/Skill) gives an estimate of whether an ensemble is over (SSR > 1) or under (SSR < 1) dispersed.
    """
    fcst_skill = skill(obs, fcst, avg_dim, member_dim=member_dim, skipna=skipna)
    fcst_spread = spread(fcst, avg_dim, member_dim=member_dim, skipna=skipna)
    return fcst_spread / fcst_skill.where(fcst_skill > 0)       # replace inf with NaN (Can sometimes happen if the forecast is to good :) ) 
