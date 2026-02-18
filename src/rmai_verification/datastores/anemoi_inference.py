import xarray as xr
import numpy as np
import zarr
import logging
import os
import shutil

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Sequence, Iterator, Optional, Any

from .base import GridDataStore, FcstDataStore
from ..grids.grid_mapping import add_xy

LOG = logging.getLogger(__name__)

DROP_VARS = [
    "latitude",
    "longitude",
    "time",
    "cos_julian_day",
    "cos_latitude",
    "cos_local_time",
    "cos_longitude",
    "insolation",
    "sin_julian_day",
    "sin_latitude",
    "sin_local_time",
    "sin_longitude",
]

MF_KWARGS = {
    "engine": "h5netcdf",
    "combine": "by_coords",
    "parallel": False,  # Let dask handle parallelization to avoid overhead
    "concat_dim": None,
    "data_vars": "minimal",
    "decode_times": True,  # Decode times for proper temporal handling
    # Note: lock parameter not set - uses xarray's default SerializableLock
    # which is required for proper synchronization in distributed environments
}

# Default chunking strategy for lazy loading
DEFAULT_CHUNKS = {
    "reference_time": 1,
    "time": -1,
    "values": -1
}


# def _chunked(seq: Sequence[str], n: int) -> Iterator[List[str]]:
#     for i in range(0, len(seq), n):
#         yield list(seq[i : i + n])


class AnemoiInference(GridDataStore,FcstDataStore):
    def __init__(
            self, 
            files: List[str], 
            variables: Union[List[str],Tuple[str],set] = None,
            mapping: Union[Dict[str,str],str] = None,
            mf_kwargs: Optional[Dict[str, Any]] = None,
            chunks: Optional[Dict[str, Union[int, str, Tuple]]] = None,
            # *,
            # zarr_path: Optional[str] = None,
            # use_zarr_if_available: bool = True, TO DO? Don't create zarr if it already exists
        ):

        LOG.info("Initializing AnemoiInference datastore")

        # Add the files to the class
        self._files = files  #FIXME should we handle file-globbing here?
        self._mapping: Union[Dict[str],str] = mapping
        self._stacked: bool = True

        # Add the xr.open_mfdataset kwargs
        # Use mutable default handling to avoid shared state
        if mf_kwargs is None:
            mf_kwargs = {}
        
        self._mf_kwargs = dict()
        for key, value in MF_KWARGS.items():
            self._mf_kwargs[key]=mf_kwargs.get(key,value)
        for key, value in mf_kwargs.items():
            if key not in self._mf_kwargs.keys():
                self._mf_kwargs[key] = value
        
        # Store custom chunks if provided, otherwise use defaults
        # Use copy() to avoid modifying the global DEFAULT_CHUNKS
        self._chunks = chunks if chunks is not None else DEFAULT_CHUNKS.copy()
        
    
        # If user passed a zarr_path and wants to use it, open it directly (fast path)
        # self._zarr_path = zarr_path

        # TO DO: Get dir from zarr_path and check if it exists, if not, create it. This way we can use zarr even if the user doesn't provide a path, as long as they want to use zarr and have a default directory for it.
        # if zarr_path:
        #     zarr_dir = os.path.dirname(zarr_path)
        #     os.makedirs(zarr_dir, exist_ok=True)

        # open a single dataset to infer some properties
        ds = xr.open_dataset(self._files[0],engine=self._mf_kwargs["engine"])

        # Get the longitudes and latitude 
        self._longitudes = ds["longitude"].data
        self._latitudes = ds["latitude"].data

        # Get the lead times
        self._lead_times = _calc_lead_times(ds)

        ds.close()

        # if zarr_path and os.path.exists(zarr_path):
        #     # fast path: open existing zarr
        #     self._data = xr.open_zarr(zarr_path, consolidated=True)
        # elif zarr_path:
        #     # build it once, then open
        #     self._to_zarr(zarr_path, batch_size=200, overwrite=False, consolidated=True)
        #     self._data = xr.open_zarr(zarr_path, consolidated=True)
        # else:
        #     self._data = self._open_netcdf()
        self._data = self._open_netcdf()

        if variables:
            self.select_variables(variables)
        
        if self._mapping:
            self._data = add_xy(self._data,self._mapping)

        LOG.info("Finished initializing AnemoiInference datastore")

    def _open_netcdf(self):
        """
        Opens and processes multiple NetCDF datasets into an xarray Dataset with
        assigned coordinates and attributes.
        This method uses `xarray.open_mfdataset` to open multiple NetCDF files,
        preprocesses them, and assigns additional coordinates such as lead time,
        grid index, valid time.
        
        Performance optimizations:
        - Uses dask chunking for lazy loading
        - Disables parallel mode to let dask handle parallelization
        - Uses default SerializableLock for thread-safe file access
        - Assigns coordinates without triggering computation

        Returns:
            xarray.Dataset: The processed dataset with assigned coordinates and
            attributes.
        """
        ds = xr.open_mfdataset(
            self._files,
            preprocess=_preprocess,
            chunks=self._chunks,
            **self._mf_kwargs,
        )
        
        # Build coordinate arrays without triggering computation
        # Use lazy operations where possible
        grid_size = ds.sizes["grid_index"]
        
        ds_coords = ds.assign_coords(
            {
                "lead_time": ("lead_time", self._lead_times),
                "grid_index": ("grid_index", np.arange(grid_size)),
                # Compute valid_time lazily using coordinates instead of .data
                "valid_time": (
                    ["reference_time", "lead_time"],
                    ds["reference_time"].values[:,np.newaxis] +
                        self._lead_times[np.newaxis,:]
                ),
                "longitude" : ("grid_index", self._longitudes),
                "latitude": ("grid_index", self._latitudes),
            }
        )
        ds_coords.attrs["is_observation"] = False
        return ds_coords

    
    # def _to_zarr(
    #     self,
    #     zarr_path: str,
    #     *,
    #     batch_size: int = 200,
    #     variables: Union[List[str], Tuple[str, ...], set, None] = None,
    #     # chunking: Optional[Dict[str, int]] = None,
    #     overwrite: bool = False,
    #     consolidated: bool = True,
    #     ):
    #     """
    #     One-time ETL: convert many NetCDF forecast files to a single Zarr store.

    #     - Writes in batches and appends along 'reference_time'
    #     - Optionally subsets variables before writing
    #     - Rechunks for good Zarr read performance

    #     Returns the zarr_path.
    #     """

    #     if os.path.exists(zarr_path):
    #         if overwrite:
    #             shutil.rmtree(zarr_path)
    #         else:
    #             return zarr_path

    #     first = True
    #     for bi, batch in enumerate(_chunked(self._files, batch_size), start=1):
    #         LOG.info(f"Zarr conversion batch {bi}: {len(batch)} files -> {zarr_path}")

    #         ds = xr.open_mfdataset(
    #             batch,
    #             preprocess=_preprocess,
    #             # reading chunking: keep tasks manageable (esp grid_index)
    #             chunks={"reference_time": 1, "time": -1, "values": -1},
    #             **self._mf_kwargs,
    #         )

    #         ds_coords = ds.assign_coords(
    #             {
    #                 "lead_time": ("lead_time", self._lead_times),
    #                 "grid_index": ("grid_index", np.arange(ds.sizes["grid_index"])),
    #                 "valid_time": (
    #                     ["reference_time", "lead_time"],
    #                     ds["reference_time"].data[:,np.newaxis] + \
    #                         self._lead_times[np.newaxis,:]
    #                 ),
    #                 "longitude" : ("grid_index", self._longitudes),
    #                 "latitude": ("grid_index", self._latitudes),
    #             }
    #         )
    #         ds_coords.attrs["is_observation"] = False

    #         if first:
    #             ds_coords.to_zarr(zarr_path, mode="w", consolidated=False)
    #             first = False
    #         else:
    #             ds_coords.to_zarr(zarr_path, mode="a", append_dim="reference_time", consolidated=False)

    #         ds.close()

    #     if consolidated:
    #         zarr.consolidate_metadata(zarr_path)

    #     return zarr_path


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
    
    ds_pruned = ds.drop_vars(DROP_VARS, errors="ignore")
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
