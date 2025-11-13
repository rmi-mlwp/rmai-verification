import xarray as xr
import numpy as np
import logging

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

from .base import GridDataStore, FcstDataStore
from ..grids.grid_mapping import add_xy

LOG = logging.getLogger(__name__)

COORDS = dict(
    longitude="longitude",
    latitude="latitude",
    valid_time="valid_time"
)

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



class XarrayZarr(GridDataStore,FcstDataStore):
    """Datastore-class to represent zarr-files"""
    def __init__(self,
                 files: str, 
                 variables: Union[List[str], Tuple[str], set] = None, 
                 mapping: Union[Dict[str,str], str] = None
                 ) -> None:
        """Initialize the Xarray-Zarr datastore.

        This constructor sets up an instance by loading data files and optionally
        mapping coordinates and selecting variables.

        Args:
            files (str): file path to load data from.
            variables (Union[List[str], Tuple[str], set], optional): Variables to select from the dataset.
                If None, all variables are loaded. Defaults to None.
            mapping (Union[Dict[str,str], str], optional): Mapping configuration for adding x,y coordinates.
                Can be either a dictionary mapping variable names or a string specifying the mapping type.
                Defaults to None.

        Returns:
            None
        """
        LOG.info("Initializing Xarray-Zarr datastore")
        self._files: Union[str,List[str]] = files
        self._mapping: Union[Dict[str,str], str]  = mapping
        self._stacked: bool = True

        # Open the dataset
        self._data = self._open()
        if self._mapping:
            # Print or log the dataset after add_xy
            LOG.info("Dataset before add_xy(): %s", self._data)
            print(self._data)  # This will print the entire dataset to the console

            self._data = add_xy(self._data,self._mapping)

            # Print or log the dataset after add_xy
            print("Dataset after add_xy():")
            print(self._data)  # This will print the entire dataset to the console

            # Alternatively, use logging for better control
            LOG.info("Dataset after add_xy(): %s", self._data)
        # Force the chunk structure and dtype
        self._data = self._data.chunk({
            'reference_time': 1,
            'lead_time': 11,
            'grid_index': 3071581
        }).astype('float32')
        
        LOG.info("Dataset after forcing the chynks: %s", self._data)

        LOG.info("Finished initializing Xarray-Zarr datastore")

    def _open(self, variables: List[str] = None):
        """Open the dataset and apply post-processing.
        This method loads the dataset from the specified file path and applies
        post-processing to add coordinates and drop unused variables.

        Args:
            variables (List[str], optional): Variables to select from the dataset.
                If None, all variables are loaded. Defaults to None.

        Returns:
            xr.Dataset: The processed dataset with coordinates and selected variables.
        """
        if isinstance(self._files,list):
            dss = [xr.open_zarr(file, consolidated=False) 
                for file in self._files]
            dss_postproc = [_postprocess(ds) for ds in dss]
            ds_postproc = xr.concat(dss_postproc, dim="valid_time")
        else:
            ds = xr.open_zarr(self._files,consolidated=False)
            ds_postproc = _postprocess(ds)
            
        if variables:
            ds_selected = ds_postproc.sel(variable=variables)
        else:
            ds_selected = ds_postproc
            # if len(ds_selected["variable"]) > 10:
            #     LOG.warning(f"Transforming anemoi-datasets xr.DataArray with {len(ds_postproc['variable'])} variables to xr.Dataset, this might take some time. Consider selecting the relevant variables during initialization")
        #LAZY FIX
        # --- Add this block for time selection ---
        ds_selected = ds_selected.sel(
            reference_time=slice('2022-10-01', '2023-09-27')
        )
        # ds_selected["grid_index"]=1
        ds_selected.attrs["is_observation"] = False

        # ds_selected = ds_selected.chunk({"grid_index": -1})
        return ds_selected #.to_dataset(dim="variable")
      
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
        
        # # Save the transposed dataset to Zarr
        # zarr_path = "/pfs/lustrep4/scratch/project_465000527/francois/Anemoi/rmai_verification_4uwcwestforecast_v1/rmai-verification/debug_XarrayZarr.zarr"
        # ds_transposed.to_zarr(zarr_path, mode="w")
        self._stacked = False


def _postprocess(dataset : xr.Dataset) -> xr.Dataset:
    """Post-process the dataset to add coordinates and drop unused variables.

    Args:
        dataset (xr.Dataset): The input dataset to be processed.

    Returns:
        xr.Dataset: The processed dataset with assigned coordinates and
            attributes.
    """
    
    # Add coordinates
    coords = {key: dataset[value].astype("datetime64[ns]").load() if key == "valid_time" else dataset[value].load() for key, value in COORDS.items()}
    for key in ("latitude","longitude"):
        coords[key] = coords[key].astype(np.float32)
    # coords["variable"] = dataset.attrs["variables"]
    coords["valid_time"] = coords["valid_time"].astype("datetime64[ns]")
    ds_coords = dataset.assign_coords(coords)

    # Drop unused variables and remove ensemble dimension
    #drop_vars = [var for var in DROP_VARS if var in coords["variable"]]
    
    ds_pruned=ds_coords #["data"]
    # ds_pruned = ds_coords["data"].isel(
    #     ensemble=0
    # ).drop_sel(
    #     variable=drop_vars
    # ).swap_dims(
    #     {"time":"valid_time"}
    # ).rename(
    #     {"cell":"grid_index"}
    # )
    return ds_pruned

