import xarray as xr
import pandas as pd
import numpy as np
import logging

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

from .base import GridDataStore, FcstDataStore
from ..grids.grid_mapping import add_xy

LOG = logging.getLogger(__name__)


#although all data will be loaded from a single file, we will leverage the
# ability to generate filenames to store info on reference_time, lead_time_max and lead_time_step 
# e.g for climatology stored in zarr_file.zarr we would expect the list of filenames
# ["zarr_file.zarr@yyyymmdd_hh@lt_max:144h@lt_step:6h"]
class ClimatologicalForecast(GridDataStore,FcstDataStore):
    def __init__(self, 
                 files: List[str], 
                 variables: Union[List[str],Tuple[str],set] = None,
                 mapping: Union[Dict[str,str],str] = None, 
                 ):
        LOG.info("Initializing ClimatologicalForecast datastore")
        # process 'files'
        self.file=''
        self.lt_max = ''
        self.lt_step = ''
        self._reference_times = []
        for file in files:
            parts = file.split("@")
            for part in parts:
                if '.zarr' in part:
                    if self.file == '':
                        self.file = part
                    assert self.file == part, "Clim dataset should be single zarr file"
                elif 'lt_max:' in part:
                    lt_max = part.split(':')[1]
                    if self.lt_max == '':
                        self.lt_max = lt_max
                    assert self.lt_max == lt_max, "A single maximal lead time should be specified."
                elif 'lt_step:' in part:
                    lt_step = part.split(':')[1]
                    if self.lt_step == '':
                        self.lt_step = lt_step
                    assert self.lt_step == lt_step, "A single lead time step should be specified."
                else:
                    self._reference_times.append(part)
            assert self.file != '' and self.lt_max != '' and self.lt_step != '', \
                "Climatological forecast file description should contain a single zarr file, a maximal lead time and a lead time step."
        self._reference_times = [pd.to_datetime(rt,format="%Y%m%d_%H").to_datetime64() for rt in self._reference_times]    
        self._lead_times = [lt.to_timedelta64() for lt in pd.timedelta_range(start="0ns", end=self.lt_max, freq=self.lt_step)]
        self._mapping: Union[Dict[str],str] = mapping
        self._stacked: bool = True
        self._variables = variables
        self._data = self._open()
        
        
        if self._mapping:
            self._data = add_xy(self._data,self._mapping)

        LOG.info("Finished initializing ClimatologicalForecast datastore")

    def _open(self):
        """

        Returns:
            xarray.Dataset: The processed dataset with assigned coordinates and
            attributes.
        """
        ds = xr.open_zarr(self.file)
        if self._variables:
            ds = ds[self._variables]
        return self._postprocess(ds)

    def unstack(self,mapping: Union[str, Dict[str,str]] = None):
        """Unstacks the dataset from a stacked format to a grid format.

        This method checks if the dataset is currently stacked and if so, it
        unstacks it. If a mapping is provided, it will be used to add x and y
        coordinates to the dataset. If the dataset is already unstacked it
        will do nothing.

        Args:
            mapping (Union[Dict[str,str],str], optional): A mapping to add x and y
                coordinates to the dataset. Defaults to None.

        Returns:
            None
        """
        LOG.debug(f"Start unstacking, stacked state is currently: {self._stacked}")
        if not self._stacked:
            pass

        if mapping != None:
            if self._mapping != None:
                LOG.error("Dataset already contains a mapping")
                raise ValueError
            else:
                self._mapping = mapping
                self._data = add_xy(self._data,mapping)
        elif self._mapping == None:
            LOG.error("No grid mapping found!")
            raise ValueError
        ds_unstacked = self._data.unstack()
        ds_transposed = ds_unstacked.transpose(
            "reference_time",
            "lead_time",
            "y",
            "x"
        )
        self._data = ds_transposed
        self._stacked = False

    def _postprocess(self, dataset : xr.Dataset) -> xr.Dataset:
        """Post-process the dataset to add coordinates and drop unused variables.

        Args:
            dataset (xr.Dataset): The input dataset to be processed.

        Returns:
            xr.Dataset: The processed dataset with assigned coordinates and
                attributes.
        """
        
        # we are going to make a forecast dataset from a climatological dataset
        # this might be overkill, but lets see if this simple way works
        climate_year = dataset.climate_time.dt.year.values[0]
        dss=[] 
        for reference_time in self._reference_times:
            reference_year=pd.to_datetime(reference_time).year
            time_shift = np.datetime64(f"{climate_year}-01-01",'ns')-np.datetime64(f"{reference_year}-01-01",'ns')
            climate_times = [reference_time + lt + time_shift for lt in self._lead_times]
            ds_rt = dataset.sel(climate_time=climate_times).expand_dims({'reference_time': [reference_time]})
            ds_rt = ds_rt.rename({'climate_time': 'lead_time'})
            ds_rt = ds_rt.assign_coords(
                {
                    "lead_time" : ("lead_time", self._lead_times),
                    "valid_time": (
                    ["reference_time", "lead_time"],
                    ds_rt.reference_time.values[:,np.newaxis] + \
                        np.array(self._lead_times)[np.newaxis,:])
                }
            )
            dss.append(ds_rt)    
        ds_pp = xr.concat(dss, dim="reference_time")
        grid_indices = ds_pp.cell.values
        ds_pp = ds_pp.rename({'cell':'grid_index'})
        ds_pp = ds_pp.assign_coords({'grid_index' : ('grid_index', grid_indices)})
        ds_pp.attrs["is_observation"] = False
        for key in ("latitude", "longitude"):
            vals=ds_pp[key].values.astype(np.float32)
            ds_pp = ds_pp.assign_coords({key: ('grid_index', vals)})
        ds_pp = ds_pp.transpose('reference_time', 'lead_time', 'grid_index', ...)
        return ds_pp