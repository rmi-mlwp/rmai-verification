import abc
import xarray as xr
from numpy.typing import NDArray
import numpy as np
from typing import Dict, List, Any, Union


class BaseDataStore(abc.ABC):
    """
    Base DataStore class
    """
    @property
    def dims(self) -> Dict[str, int]:
        """Returns the names and sizes of the dimensions of the DataStore
        
        Returns:
            Dict: the dimension names and sisze of the data
        """
        return dict(self._data.sizes)

    @property
    def vars(self) -> List[str]:
        """Returns the variables in the datastore
        
        Returns:
            List: Variables in the datastore
        """
        return list(self._data.keys())
    
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

    @property
    def valid_times(self) -> xr.DataArray:
        """Returns the valid times of the datastore
        
        Returns
            xr.DataArray: for ObsDatastore a 1D data-array, 
            for FcstDataStore a 2D data-array 
        """
        return self._data["valid_time"]


    @property
    def is_observation(self) -> bool:
        """Returns True if the datastore contains observations, 
        False if it contains forecasts

        Returns:
            bool: True for observation datastore, False for forecast datastore
        """
        return self._is_observation

    @property
    def is_point(self) -> bool:
        """Returns True if the datastore contains point-based data, 
        False if it contains grid-based data

        Returns:
            bool: True for point-based datastore, False for grid-based datastore
        """
        return self._is_point

    @property
    def data(self) -> xr.Dataset | xr.DataArray:
        """Returns the xr.Dataset or xr.DataArray in a prediscribed format

        Returns:
            xr.Dataset or xr.DataArray
        """
        return self._data

    def select_variables(self,variables: List[str]) -> None:
        """Selects variables from the data

        Returns: 
            None
        """
        print(f"Variables in the dataset before selection: {list(self._data.keys())}")
        new_data = self._data[variables]
        self._data = new_data
    
    def transform(self,transformation): #FIXME: define a transformation class
        """
        Applies a transformation to the current data and updates it.

        Args:
            transformation: An transformation object that provides an `execute` method, 
                            which takes the current data as input and returns the transformed data.

        Returns:
            None
        """
        new_data = transformation.execute(self._data)
        self._data = new_data

    
class GridDataStore(BaseDataStore):
    """Class for DataStores that contain gridded (structured or unstructured) data"""
    _is_point: bool = False

    @property
    def is_stacked(self) -> bool:
        """Returns true if the data is stacked (1-D)
        
        Returns:
            Boolean: True if data in the GridDataStore is stacked (1-D)
        """
        return self._stacked
    
    
    @abc.abstractmethod
    def unstack(self) -> None:
        """Unstack the 1D spatial dimension to 2D

        Returns:
            None
        """
        pass


class PointDataStore(BaseDataStore):
    """Class for DataStores that contain point data"""
    _is_point: bool = True

class ObsDataStore(BaseDataStore):
    "Class for Datastores that contain observations or (re)analysis"
    _is_observation: bool = True

    @property
    def valid_times(self) -> NDArray[np.datetime64]:
        """Returns the valid times from the datastore.

        Returns:
            NDArray[np.datetime64]: An array of valid times represented as numpy datetime64 objects.
        """
        return self._data["valid_time"].values

    def select_valid_times(self,
                           valid_times: Union[List[np.datetime64], NDArray[np.datetime64], xr.DataArray],
                           chunks: Union[Dict[str, Any], None] = None) -> None:
        """Subsets the data in the datastore to only contain
        the selected valid_times

        Args:
            valid_times: the valid times to select, typically the 2D
                (reference_time, lead_time) array of the forecasts.
            chunks: optional dask chunks of the forecasts along the dims of
                ``valid_times``, e.g. {"reference_time": (186,),
                "lead_time": (1,)*72}. When given, the selection is built in
                exactly these chunks (see ``_select_valid_times_windowed``).

        Returns:
            None
        """
        new_data = self._data.sel(valid_time=valid_times)
        if chunks is not None and isinstance(valid_times, xr.DataArray):
            new_data = _select_valid_times_windowed(self._data, valid_times, chunks, new_data)
        self._data = new_data

class FcstDataStore(BaseDataStore):
    _is_observation: bool = False

    @property
    def reference_times(self) -> NDArray[np.datetime64]:
        """Returns the reference times from the datastore.

        Returns:
            NDArray[np.datetime64]: An array of reference times.
        """
        return self._data["reference_time"].values

    @property
    def lead_times(self) -> NDArray[np.timedelta64]:
        """
        Retrieve the lead times from the datastore.

        Returns:
            NDArray[np.timedelta64]: An array of lead times represented as numpy timedelta64 objects.
        """
        return self._data["lead_time"].values
    
    def select_reference_times(self,reference_times: Union[List[np.datetime64], NDArray[np.datetime64], xr.DataArray]) -> None:
        """Subsets the data in the datastore to only contain the selected reference times

        Returns:
            None
        """
        new_data = self._data.sel(reference_time=reference_times)
        self._data = new_data    

    
    def select_lead_times(self,lead_times: Union[List[np.timedelta64], NDArray[np.timedelta64], xr.DataArray]) -> None:
        """Subsets the data in the datastore to only contain the selected lead times

        Returns:
            None
        """
        new_data = self._data.sel(lead_time=lead_times)
        self._data = new_data


class PointObservations(PointDataStore, ObsDataStore):
    def __init__(self, files):
        self._data = xr.open_dataset(files)

class PointForcasts(PointDataStore, FcstDataStore):
    def __init__(self, files):
        self._data = xr.open_dataset(files)

class GriddedObservations(GridDataStore, ObsDataStore):
    def __init__(self, files):
        self._data = xr.open_dataset(files)

class GriddedForecasts(GridDataStore, FcstDataStore):
    def __init__(self, files):
        self._data = xr.open_dataset(files)



# Largest contiguous valid-time window (bytes per variable) that
# _select_valid_times_windowed holds as a single chunk; above it the plain
# label selection is kept.
WINDOW_MAX_BYTES = 2 * 1024**3


def _select_valid_times_windowed(data: xr.Dataset,
                                 valid_times: xr.DataArray,
                                 chunks: Dict[str, Any],
                                 selected: xr.Dataset) -> xr.Dataset:
    """Same values as ``data.sel(valid_time=valid_times)``, laid out in the
    forecasts' chunks.

    ``data.sel(valid_time=<2D array>)`` on dask data gathers point by point and
    keeps the source's time chunking, so its (reference_time, lead_time) chunks
    cross those of the forecast (e.g. 1 x 24 against 186 x 1). Every
    elementwise op between the two then splits into 1 x 1 pieces: ~600k tasks
    for 31 inits x 72 leads and 12 thresholds, which spent minutes and tens of
    GB in the dask scheduler (ETS pilot, job 23477403; profile 23485113).

    Here the contiguous valid-time span the batch needs (inits + lead range,
    e.g. 186 + 72 hours) becomes one chunk along valid_time, read once, and
    each forecast-shaped block is cut from it with ``np.take``: one task per
    forecast block. ``selected`` is the plain lazy selection; only its data is
    replaced, so coordinates, attributes and dim order stay identical to it.
    Falls back to ``selected`` when a valid time is missing or the window
    would be too large.
    """
    axis = data["valid_time"].values
    want = valid_times.values
    pos = np.searchsorted(axis, want)
    if (pos >= axis.size).any() or not (axis[np.minimum(pos, axis.size - 1)] == want).all():
        return selected
    lo, hi = int(pos.min()), int(pos.max()) + 1
    window = data.isel(valid_time=slice(lo, hi))
    pos = pos - lo
    vt_dims = list(valid_times.dims)

    import dask.array as da

    out = selected.copy()
    for name, var in window.data_vars.items():
        if "valid_time" not in var.dims or not hasattr(var.data, "dask"):
            continue
        rest = [d for d in var.dims if d != "valid_time"]
        if np.prod([var.sizes[d] for d in rest]) * (hi - lo) * var.dtype.itemsize > WINDOW_MAX_BYTES:
            return selected
        src = var.transpose("valid_time", *rest).data.rechunk({0: -1})
        idx = da.from_array(pos, chunks=tuple(chunks.get(d, -1) for d in vt_dims))
        n_idx = idx.ndim
        rest_ind = tuple(range(n_idx + 1, n_idx + 1 + len(rest)))
        arr = da.blockwise(
            _take_window, tuple(range(n_idx)) + rest_ind,
            idx, tuple(range(n_idx)),
            src, (n_idx,) + rest_ind,
            concatenate=True, dtype=var.dtype, meta=np.empty((0,) * (n_idx + len(rest)), var.dtype),
        )
        target_dims = selected[name].dims
        arr_dims = vt_dims + rest
        # Set the data on the (shallow-copied) variable itself: assigning a
        # DataArray would re-merge its coordinates, including the grid
        # MultiIndex, for nothing.
        out.variables[name].data = arr.transpose([arr_dims.index(d) for d in target_dims])
    return out


def _take_window(idx: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Blocks of the windowed selection: src[idx[...], ...]."""
    return np.take(src, idx, axis=0)
