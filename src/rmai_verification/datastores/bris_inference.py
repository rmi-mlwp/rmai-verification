import xarray as xr
import numpy as np
import logging

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

from .base import GridDataStore, FcstDataStore
from ..grids.grid_mapping import add_xy

LOG = logging.getLogger(__name__)

DROP_DIMS = [
    "height1",
    "height_above_msl",
    "height",
    "height2",
]

MF_KWARGS = {
    "engine":"h5netcdf",
    "combine":"by_coords",
    "parallel":True,
    "concat_dim": None,
    "data_vars":"minimal"
}


class BrisInference(GridDataStore,FcstDataStore):
    def __init__(self, 
                 files: List[str], 
                 variables: Union[List[str],Tuple[str],set] = None,
                 mapping: Union[Dict[str,str],str] = None,
                 mf_kwargs: Dict[str,str] = dict()
                 ):
        LOG.info("Initializing BrisInference datastore")
        # Add the files to the class
        self._files = files #FIXME should we handle file-globbing here?
        self._mapping: Union[Dict[str],str] = mapping
        self._stacked: bool = False

        # Add the xr.open_mfdataset kwargs
        self._mf_kwargs = dict()
        for key, value in MF_KWARGS.items():
            self._mf_kwargs[key]=mf_kwargs.get(key,value)
        for key, value in mf_kwargs.items():
            if key not in self._mf_kwargs.keys():
                self._mf_kwargs[key] = value
        
    
        # open a single dataset to infer some properties
        ds = xr.open_dataset(self._files[0],engine=self._mf_kwargs["engine"])
        ds["longitude"] = (ds["longitude"]  % 360)
        # Get the longitudes and latitude 
        self._longitudes = ds["longitude"].data
        self._latitudes = ds["latitude"].data

        # Get the lead times
        self._lead_times = _calc_lead_times(ds)

        ds.close()


        self._data = self._open()
        if variables:
            self.select_variables(variables)
        
        if self._mapping:
            self._data = add_xy(self._data,self._mapping)

        LOG.info("Finished initializing BrisInference datastore")
    
    @property
    def longitudes(self) -> NDArray:
        """Returns the longitudes of the datastore
        
        Returns
            ndarray: 
        """
        return self._data["longitude"].values

    @property
    def latitudes(self) -> NDArray:
        """Returns the latitudes of the datastore
        
        Returns
            ndarray: 
        """
        return self._data["latitude"].values
    
    @latitudes.setter
    def latitudes(self, latitudes: NDArray[np.float64]):
        """
        Setter for the latitudes attribute.
        This method assigns the provided latitudes to the datastore's latitude coordinate.
        
        Args:
            latitudes (NDArray[np.float64]): An array of latitude values to set.
        """
        self._latitudes = latitudes

    @longitudes.setter
    def longitudes(self, longitudes: NDArray[np.float64]): 
        """
        Setter for the longitudes attribute.
        This method assigns the provided longitudes to the datastore's longitude coordinate.
        
        Args:
            longitudes (NDArray[np.float64]): An array of longitude values to set.
        """
        self._longitudes = longitudes

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
        ds = xr.open_mfdataset(
            self._files,
            preprocess=_preprocess,
            chunks={
                "reference_time" : 1,
                "time": 1,
                "x": 200,
                "y": 200
            },
            **self._mf_kwargs,
        )
        ds_coords = ds.assign_coords(
            {
                "lead_time": ("lead_time", self._lead_times),
                "x": ("x", ds["x"].data),
                "y": ("y", ds["y"].data),
                "valid_time": (
                    ["reference_time", "lead_time"],
                    ds["reference_time"].data[:,np.newaxis] + \
                        self._lead_times[np.newaxis,:]
                ),
                "longitude" : (["y", "x"], self._longitudes),
                "latitude": (["y", "x"], self._latitudes),
            }
        )
        ds_coords.attrs["is_observation"] = False
        return ds_coords

    def stack(self):
        """Stacks the dataset from a grid format to a stacked format.

        This method checks if the dataset is currently unstacked and if so, it
        stacks it. If the dataset is already stacked it will do nothing.

        Returns:
            None
        """
        LOG.debug(f"Start stacking, stacked state is currently: {self._stacked}")
        if self._stacked:
            pass
        ds_stacked = self._data.stack(
            {
                "grid_index": ["y", "x"]
            }
        )
        ds_transposed = ds_stacked.transpose(
            "reference_time",
            "lead_time",
            "grid_index"
        )
        self._data = ds_transposed
        self._stacked = True


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
    for var in DROP_DIMS:
        if var in ds.coords:
            ds = ds.squeeze(var)
            ds = ds.drop(var)
    ds_reftime = ds.expand_dims(
        reference_time=[reference_time]
    )
    ds_reftime[
        "reference_time"
    ].attrs["standard_name"] = "forecast_reference_time"
    ds_reftime["time"] = np.array(range(len(ds_reftime["time"])))
    ds_renamed = ds_reftime.rename_dims(
        {
            "time":"lead_time"
        }
    )
    return ds_renamed
