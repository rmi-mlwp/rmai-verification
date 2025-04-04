from typing import Dict, List
import logging
import xarray as xr
import numpy as np

from interpolation import METHODS

from datastores.base import BaseDataStore

LOG = logging.getLogger(__name__)

POINT_COORDS = ["latitude", "longitude"]


def common_grid_or_points(datastores : Dict[str, BaseDataStore], reference_datastore : str, transformation_kwargs : Dict[str, str] = dict()) -> Dict[str, xr.Dataset]:
    #FIXME: Datastores should have a .coords() classmethod
    common_data = dict()
    if datastores[reference_datastore].is_point:
        LOG.info(f"reference datastore {reference_datastore} is a PointDatastore")
        transformer = METHODS["interpolate"]
        transformation = transformer(
            datastores[reference_datastore].data,
            transformation_kwargs
        )
        for name, store in datastores.items():
            if store.is_point:
                LOG.info(f"Selecting all reference points for datastore: {name}")
                #TODO: Check if all reference codes are in Datastore
                _data = store.data.sel(
                    code=datastores[reference_datastore].data["code"].values
                )
            
            else:
                for coord in POINT_COORDS:
                    assert coord in list(datastores[reference_datastore].data.coords.keys()), f"Coordinate {coord} missing from the reference datastore {reference_datastore}"
                LOG.info(f"Interpolating to reference points for datastore {name}")
                if store.is_stacked:
                    store.unstack()
                _data = transformation.execute(store.data)
            common_data[name] = _data

    else:
        LOG.info(f"reference datastore {reference_datastore} is a GridDatastore")
        LOG.warning("Regridding not supported yet")
        for name, store in datastores.items():
            if store.is_point:
                LOG.error(f"Cannot transform a point datastore {name} to a grid.")
                raise ValueError
            else:
                # for coord in POINT_COORDS:
                #     assert coord in list(datastores[reference_datastore].data.coords.keys()), f"Coordinate {coord} missing from the reference datastore {reference_datastore}"
                LOG.info(f"Fake regridding to reference points for datastore {name}")
                if store.is_stacked:
                    LOG.info(f"Unstacking {name}")
                    store.unstack()
                _data = store.data
            common_data[name] = _data


        #raise NotImplementedError
        #transformer = get_transformation("regrid")
        #key = "regridding"         
    return common_data