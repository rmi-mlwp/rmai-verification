import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import logging
from typing import List, Union, Dict
from . import GRIDS, PROJECTIONS

LOG = logging.getLogger(__name__)

def create_cartopy_crs(projection : str ,projection_kws: Dict[str, str]) -> ccrs.Projection:
    """Create a Cartopy coordinate reference system (CRS) based on the specified projection.

    This function creates a Cartopy projection object using the provided projection name
    and associated keyword arguments.
    
    Parameters
    ----------
    projection : str
        Name of the projection to create. Must be one of the supported projections
        defined in PROJECTIONS.
    projection_kws : Dict[str, str]
        Dictionary of keyword arguments to pass to the projection constructor.
        Can include 'globe' parameters which will be used to create a ccrs.Globe object.
    
    Returns
    -------
    ccrs.Projection
        The created Cartopy projection object.
    Raises
    ------
    AssertionError
        If the specified projection is not supported (not in PROJECTIONS).
    
    Examples
    --------
    >>> projection_kws = {'central_longitude': 0, 'globe': {'ellipse': 'WGS84'}}
    >>> crs = create_cartopy_crs('latlon', projection_kws)
    """

    assert projection in PROJECTIONS, f"Projection {projection} not supported (yet)."
    
    # - Get the cartopy projection (crs)
    projection = PROJECTIONS[projection]
    kwargs = projection_kws.copy()

    # - Move globe keywords to different dictionary
    globe = kwargs.pop("globe", None)
    if globe:
        globe = ccrs.Globe(**globe)
    
    crs = projection(globe=globe, **kwargs)
    return crs


def create_multiindex(ds: xr.Dataset | xr.DataArray, x : str = "x", y : str = "y", dim_to_multiindex : str ="grid_index", **kwargs) -> pd.MultiIndex:
    """Create a multiindex for the x and y coordinates of a dataset.
    The multiindex is created based on the specified grid dimensions and the
    coordinates of the lower left and upper right corners of the grid.
    The multiindex is created using the specified dimension name and the x and y
    coordinates.
    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        The dataset or data array for which to create the multiindex.
    x : str, optional
        The name of the x coordinate. Default is "x".
    y : str, optional
        The name of the y coordinate. Default is "y".
    dim_to_multiindex : str, optional
        The name of the dimension to create the multiindex for. Default is "grid_index".
    **kwargs : dict
        Additional keyword arguments for the grid dimensions and coordinates.
        Must include "lower_left" and "upper_right" coordinates and "nx" and "ny", the number of x and y gridpoints.
    Returns
    -------
    pd.MultiIndex
        The created multiindex for the x and y coordinates.
    """
    # - Get the grid dimensions  
    nx = kwargs.get("nx")
    ny = kwargs.get("ny")

    # - Get the coordinates of the lower left and upper right corners
    #   of the grid
    lon_ll, lat_ll = kwargs.get("lower_left")
    lon_ur, lat_ur = kwargs.get("upper_right")
    crs = ds.attrs["crs"]

    assert dim_to_multiindex in ds.dims, f"Dimension {dim_to_multiindex} not found in the dataset"
    assert ds.sizes[dim_to_multiindex] == nx * ny, f"Proposed grid dimensions ({ny}, {nx}) do not match the length of {dim_to_multiindex}: {ds.sizes[dim_to_multiindex]}"

    # Transform the coordinates of the lower left and upper right corners
    # of the grid to the projection of the dataset
    x_ll, y_ll = crs.transform_point(
        x=lon_ll,
        y=lat_ll,
        src_crs=ccrs.PlateCarree()
    )
    x_ur, y_ur = crs.transform_point(
        x=lon_ur,
        y=lat_ur,
        src_crs=ccrs.PlateCarree()
    )
    
    # Allow for thinning
    if "thinning" in ds.attrs:
        thinning_factor = ds.attrs["thinning"]
    else:
        thinning_factor = 1

    x_values = np.linspace(x_ll,x_ur,nx,endpoint=True)[::thinning_factor]
    y_values = np.linspace(y_ll,y_ur,ny,endpoint=True)[::thinning_factor]
    
    # Create the multiindex
    mindex = pd.MultiIndex.from_product(
        [y_values, x_values],
        names=[y,x]
    )

    return mindex

def add_xy(ds: xr.Dataset | xr.DataArray , grid: Union[str, Dict[str,str]]) -> xr.Dataset | xr.DataArray:
    """Add x and y coordinates to a xarray Dataset or DataArray based on specified grid mapping.

    This function adds cartesian coordinates (x, y) to a dataset based on a predefined or custom grid
    specification. It creates a Coordinate Reference System (CRS) and calculates the corresponding
    x, y coordinates for the given grid.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input dataset or data array to which coordinates will be added
    grid : Union[str, Dict[str,str]]
        Either a string specifying a predefined grid name from GRIDS, or a dictionary containing:
        - 'projection': str, projection name
        - 'projection_kws': dict, projection parameters
        - 'grid_kws': dict, containing at minimum 'lower_left' and 'upper_right' corner coordinates
    
    Returns
    -------
    xr.Dataset | xr.DataArray
        Copy of input dataset with added x, y coordinates and CRS information in attributes
    Notes
    -----
    The grid_kwargs dictionary must contain 'lower_left' and 'upper_right' specifications
    defining the extent of the grid.
   """
    
    # If native domain is a string, get the specifications from the pre-defined mappings
    if isinstance(grid, str):
        assert grid in GRIDS, f"Grid {grid} not supported, please provide a dictionary with the specifications"
        grid = GRIDS[grid]

    # - Get the cartopy projection (crs)
    projection = grid["projection"]

    # - Get the crs-keywords
    projection_kwargs = grid["projection_kws"].copy()

    # - Get the grid-keywords
    grid_kwargs = grid["grid_kws"].copy()
    
    # Create Coordinate Reference Systems (crs)
    crs = create_cartopy_crs(
        projection,
        projection_kwargs
    )
    ds_new = ds.copy()
    ds_new.attrs["crs"] = crs
 
    assert "lower_left" in grid_kwargs and "upper_right" in grid_kwargs, f"Longitudes and latitudes of lower left and/or upper right conrer are missing"    
    
    LOG.info("Calculating x and y values from the extent")
    multi_index = create_multiindex(ds_new,**grid_kwargs) 
    multi_coordinates = xr.Coordinates.from_pandas_multiindex(
        multi_index, "grid_index")
    ds_xy = ds_new.assign_coords(multi_coordinates)
    return ds_xy

