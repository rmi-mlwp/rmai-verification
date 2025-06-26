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

    def sub_grid(self, index_file, dim = 'grid_index'):
        index_info = np.load(index_file)
        full_grid_lats = index_info['full_grid_lats']
        full_grid_lons = index_info['full_grid_lons']
        sub_grid_lats = index_info['sub_grid_lats']
        sub_grid_lons = index_info['sub_grid_lons']
        idx = index_info['index']
        assert np.allclose(full_grid_lats, self.latitudes) and np.allclose(full_grid_lons, self.longitudes), 'incompatible index'
        self._data = self._data.isel({dim  : idx}).assign_coords({dim : list(range(len(idx)))})
        assert np.allclose(sub_grid_lats, self.latitudes) and np.allclose(sub_grid_lons, self.longitudes), 'incompatible index'
        
        



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

    def select_valid_times(self,valid_times: Union[List[np.datetime64], NDArray[np.datetime64], xr.DataArray]) -> None:
        """Subsets the data in the datastore to only contain 
        the selected valid_times

        Returns:
            None
        """
        new_data = self._data.sel(valid_time=valid_times)
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

