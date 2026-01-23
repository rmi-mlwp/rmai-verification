from typing import Dict, List
import logging
import xarray as xr
import numpy as np

from ..interpolation import METHODS
from ..datastores import BaseDataStore

LOG = logging.getLogger(__name__)

POINT_COORDS = ["latitude", "longitude"]

# Tolerance in degrees that the coordinates of two grids can differ while still being interpreted as the same grid.
# 0.0001 degrees ~ 10m at 45 deg latitude
COORD_TOLERANCE = 0.001 

def align_spatial(datastores : Dict[str, BaseDataStore], reference_datastore : str, transformation_kwargs : Dict[str, str] = dict()) -> Dict[str, xr.Dataset]:
    """Align spatial coordinates of multiple datastores to a reference datastore.

    This function handles spatial alignment between different datastores, supporting both point-based 
    and grid-based data. For point-based reference datastores, it either selects matching points 
    or interpolates grid data to points. Grid-based reference datastores are not currently supported.

    Parameters
    ----------
    datastores : Dict[str, BaseDataStore]
        Dictionary of datastores to align, where keys are datastore names and values are BaseDataStore objects
    reference_datastore : str
        Name of the reference datastore (must be a key in datastores dict)
    transformation_kwargs : Dict[str, str], optional
        Additional keyword arguments for transformation methods, by default empty dict
    
    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary containing the aligned data for each datastore
   
    Raises
    ------
    NotImplementedError
        If the reference datastore is grid-based (regridding not yet supported)

    """

    #FIXME: Datastores should have a .coords() classmethod
    _datastores = datastores.copy()
    ref_store = _datastores.pop(reference_datastore)
    common_data = dict()
    if ref_store.is_point:
        # The reference datastore is a PointDataStore.
        LOG.info(f"reference datastore {reference_datastore} is an ObsDataStore")
        common_data[reference_datastore] = ref_store.data
        interpolator = METHODS["interpolate"]
        interpolation = interpolator(
            ref_store.data,
            transformation_kwargs
        )
        for name, store in _datastores.items():
            # If the reference datastore is a PointDataStore there are two options:
            if store.is_point:
                # 1. The current datastore is also a PointDataStore:
                # Only select those datapoints that are in the reference datastore.
                # TODO: What if there are points in the reference datastore that are not 
                # in the current datastore?
                LOG.info(f"Selecting all reference points for datastore: {name}")
                _data = store.data.sel(
                    code=ref_store.data["code"].values
                )
            else:
                # 2. The current datastore is a GridDataStore: Interpolate the grid to the points
                # using the above defined interpolator
                for coord in POINT_COORDS:
                    assert coord in list(ref_store.data.coords.keys()), f"Coordinate {coord} missing from the reference datastore {reference_datastore}"
                LOG.info(f"Interpolating to reference points for datastore {name}")
                if not store.is_stacked:
                    store.unstack()
                _data = interpolation.execute(store.data)
            common_data[name] = _data

    else:
        # The reference datastore is a GridDataStore
        # Options:
        # 1. Same projection: Take subgrid
        # 2. Different projection: Regrid
        LOG.info(f"reference datastore {reference_datastore} is a GridDataStore")
        if ref_store._mapping:
            ref_store.unstack()
        common_data[reference_datastore] = ref_store.data
        for name, store in _datastores.items():
            if store.is_point:
                LOG.error(f"Cannot transform PointDataStore {name} to a grid.")
                raise ValueError
            else:
                if store._mapping:
                    store.unstack()
                # Check if shape of the coordinates is equal 
                if (ref_store.latitudes.shape == store.latitudes.shape) and \
                    (ref_store.longitudes.shape == store.longitudes.shape):
                    #TODO: We don't necessarily need to unstack here. But then the scores are also multiindexed.
                    # And saving multiindexed data is not yet supported by the xarray backend.
                    if (ref_store.latitudes == store.latitudes).all() and (ref_store.longitudes == store.longitudes).all():
                        LOG.info(f"The grid coordinates of datastore {name} are identical to the reference datastore")
                        common_data[name] = store.data
                    elif np.isclose(ref_store.latitudes, store.latitudes, atol=COORD_TOLERANCE).all() and \
                        np.isclose(ref_store.longitudes, store.longitudes, atol=COORD_TOLERANCE).all():
                        LOG.warning(f"Some lat-lon coordinates of datastore {name} and reference datastore {reference_datastore} differ.\n" + 
                                    f"But the difference is less then {COORD_TOLERANCE} degrees, considering both grids as equal")
                        store = store.data.assign_coords(
                                    {
                                        "x": ("x", ref_store.data["x"].data),
                                        "y": ("y", ref_store.data["y"].data),
                                        "longitude" : (["y", "x"], ref_store.longitudes),
                                        "latitude": (["y", "x"], ref_store.latitudes),
                                    }
                                )
                        common_data[name] = store

                    else:
                        raise NotImplementedError("Regridding is not yet supported")
                else:
                    raise NotImplementedError("Regridding is not yet supported")

    return common_data