import xarray as xr
import numpy as np
import logging

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

from .base import GridDataStore, FcstDataStore
from ..grids.grid_mapping import add_xy

LOG = logging.getLogger(__name__)

DROP_VARS = []
#[
#     "latitude",
#     "longitude",
#     "time",
#     "cos_julian_day",
#     "cos_latitude",
#     "cos_local_time",
#     "cos_longitude",
#     "insolation",
#     "sin_julian_day",
#     "sin_latitude",
#     "sin_local_time",
#     "sin_longitude",
# ]

MF_KWARGS = {
    "engine":"h5netcdf",
    "parallel":True,
}


class AnemoiInference(GridDataStore,FcstDataStore):
    def __init__(self, 
                 files: List[str], 
                 variables: Union[List[str],Tuple[str],set] = None,
                 mapping: Union[Dict[str,str],str] = None,
                 mf_kwargs: Dict[str,str] = dict(),
                 ens_size: int = 1
                 ):
        LOG.info("Initializing AnemoiInference datastore")
        # Add the files to the class
        self._files = files #FIXME should we handle file-globbing here?
        print("Number of files provided: ", len(self._files))
        print("Filenames: ", self._files)
        self._mapping: Union[Dict[str],str] = mapping
        self._stacked: bool = True
        self.ens_size = ens_size

        # Add the xr.open_mfdataset kwargs
        self._mf_kwargs = dict()
        for key, value in MF_KWARGS.items():
            self._mf_kwargs[key]=mf_kwargs.get(key,value)
        for key, value in mf_kwargs.items():
            if key not in self._mf_kwargs.keys():
                self._mf_kwargs[key] = value
        
    
        # open a single dataset to infer some properties
        ds = xr.open_dataset(self._files[0],engine=self._mf_kwargs["engine"])

        # Get the longitudes and latitude 
        self._longitudes = ds["longitude"].data
        self._latitudes = ds["latitude"].data

        # Get the lead times
        self._lead_times = _calc_lead_times(ds)
        print("Lead times: ", self._lead_times)

        ds.close()


        self._data = self._open()
        if variables:
            self.select_variables(variables)
        
        if self._mapping:
            self._data = add_xy(self._data,self._mapping)
        
        print("Finished initializing AnemoiInference datastore")
        LOG.info("Finished initializing AnemoiInference datastore")

    def _open(self):
        """
        Opens and processes multiple NetCDF datasets into an xarray Dataset with
        assigned coordinates and attributes.
        This method uses `xarray.open_mfdataset` to open multiple NetCDF files,
        preprocesses them, and assigns additional coordinates such as lead time,
        grid index, valid time.

        Returns:
            xarray.Dataset: The processed dataset with assigned coordinates and
            attributes.
        """
        if self.ens_size == 1:
            ds = xr.open_mfdataset(
                self._files,
                preprocess=_preprocess,
                chunks={
                    "reference_time" : 1,
                    "time": -1,
                    "values": -1
                },
                combine="by_coords",
                **self._mf_kwargs,
            )
        else:
            ds = xr.open_mfdataset(
                self._files,
                preprocess=_preprocess,
                chunks={
                    "reference_time" : 1,
                    "time": -1,
                    "values": -1,
                    "ensemble": -1
                },
                concat_dim="ensemble",
                combine="nested",
                **self._mf_kwargs,
            )   
        print("Dataset opened", ds)
        ds_coords = ds.assign_coords(
            {
                "lead_time": ("lead_time", self._lead_times),
                "grid_index": ("grid_index", np.arange(ds.sizes["grid_index"])),
                "valid_time": (
                    ["reference_time", "lead_time"],
                    ds["reference_time"].data[:,np.newaxis] + \
                        self._lead_times[np.newaxis,:]
                ),
                "longitude" : ("grid_index", self._longitudes),
                "latitude": ("grid_index", self._latitudes),
            }
        )
        ds_coords.attrs["is_observation"] = False
        return ds_coords

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
            

def _calc_lead_times(ds: xr.Dataset | xr.DataArray) -> NDArray[np.timedelta64]:
        """
        Calculate the lead times from a dataset.

        This function computes the lead times by subtracting the first time value 
        in the dataset from all other time values. The result is returned as a 
        data array.

        Args:
            ds (xarray.Dataset): The input dataset containing a "time" coordinate.

        Returns:
            numpy.ndarray: An array of lead times relative to the first time value.
        """
        return (ds["time"]- ds["time"][0]).data
        # print(ds)
        # print("Calculating lead times:")
        # print("Reference time: ", ds.coords["reference_time"].data)
        # print("Time values: ", ds["time"].data)
        # print("First time value: ", ds["time"][0].data)
        # print("Reference lead times calculated: ", (ds.coords["reference_time"]- ds["time"][0]).data)
        # print("Lead times calculated: ", (ds["time"]- ds["time"][0]).data)
        # print(type((ds.coords["reference_time"]- ds["time"][0]).data))
        # return (ds.coords["reference_time"]- ds["time"][0]).data + (ds["time"]- ds["time"][0]).data

def _preprocess(ds: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """
    Preprocess the dataset by dropping unnecessary variables and renaming dimensions.
    This function drops specified variables from the dataset and renames dimensions
    to standard names. It also expands the reference time dimension and assigns
    attributes to the reference time variable.

    Args:
        ds (xarray.Dataset): The input dataset to preprocess.

    Returns:
        xarray.Dataset: The preprocessed dataset with dropped variables and renamed dimensions.
    """

    reference_time = ds["time"].data[0]
    
    ds_pruned = ds.drop_vars(DROP_VARS)
    ds_reftime = ds_pruned.expand_dims(
        reference_time=[reference_time]
    )
    ds_reftime[
        "reference_time"
    ].attrs["standard_name"] = "forecast_reference_time"

    ds_renamed = ds_reftime.rename_dims(
        {
            "values":"grid_index",
            "time":"lead_time"
        }
    )
    return ds_renamed
