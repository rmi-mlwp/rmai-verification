import xarray as xr
import cfgrib
import dask
import numpy as np
import logging

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

from .base import GridDataStore, FcstDataStore

LOG = logging.getLogger(__name__)

DROP_VARS = ["surface"]
             
class IfsForecast(GridDataStore, FcstDataStore):
    def __init__(self,
                 files: List[str],
                 variables: Union[List[str], Tuple[str], set] = None,
                 mf_kwargs: Dict[str,str] = dict(),
                 stacked: bool = True,
                 shift_longitude: bool = True,
                 subgrid_idx: str = None,
                 max_steps: int = None,
                 is_surface: bool = True
                ):
        LOG.info("Initializing IfsForecast datastore")

        self._files = files
        self._mf_kwargs = mf_kwargs
        self._stacked = stacked
        self._idx = subgrid_idx

        data = xr.open_mfdataset(
            self._files,
            combine="nested",
            concat_dim="time",
            chunks={
                "time" : 1,
                "step": -1,
                "values": -1
            },
            **self._mf_kwargs
        )
        if shift_longitude:
            data.coords["longitude"] = (data.coords["longitude"] + 180.) % 360. -180.

        self._data = data.rename_dims(
            time="reference_time",
            step="lead_time",
            values="grid_index"
        ).rename_vars(
            time="reference_time",
            step="lead_time"
        )
        
        if is_surface:
            self._data = self._data.drop_vars(
            ["number", "surface"]
        )

        else:
            vrs = list(self._data.drop_vars(["number"]))
            levels = self._data.isobaricInhPa.values
            levels = [int(l) for l in levels if l not in [600]]
            new_data = xr.Dataset()
            for v in vrs:
                for l in levels:
                    new_data[f"{v}_{l}"] = self._data[v].sel(isobaricInhPa = l)
            self._data = new_data

        if variables:
            self.select_variables(variables)

        if max_steps:
            lts = [np.timedelta64(6*i, 'h') for i in range(max_steps + 1)]
            self._data = self._data.sel(lead_time = lts) 

        if self._idx:
            self.sub_grid(self._idx)
        
        LOG.info("Finished initializing IfsForecast datastore")
        
    def unstack(self):
        LOG.warning("Unstacking of IfsForecast datastores not supported yet")
        pass





def _preprocess(ds: xr.Dataset) -> xr.Dataset:
    ds_pruned = ds.rename_dims(
        time="reference_time",
        step="lead_time",
        values="grid_index"
    ).rename_vars(
        time="reference_time",
        step="lead_time"
    ).drop_vars(
        ["number","surface"]
    )
    return ds_pruned