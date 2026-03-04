import xarray as xr
import logging
import os

LOG = logging.getLogger(__name__)


def save_as_netcdf(ds : xr.Dataset | xr.DataArray, path : str, **kwargs) -> None:
    """Save an xarray dataset or array as a netCDF file.
    Args:
        ds (xarray.Dataset | xarray.DataArray): The xarray dataset or array to save
        path (str): Path where the netCDF file will be saved
        **kwargs: Additional keyword arguments passed to xarray's to_netcdf() method
    Returns:
        None
    Examples:
        >>> ds = xarray.Dataset(...)
        >>> save_as_netcdf(ds, "output.nc")
    """

    LOG.info(f"Saving data as netCDF-file at {path}")
    ds.to_netcdf(path, **kwargs)

def save_as_zarr(ds: xr.Dataset | xr.DataArray, path: str, **kwargs) -> None:
    """
    Save an xarray dataset or array as a zarr store.
    Supports appending along a dimension via append_dim=...
    """
    append_dim = kwargs.pop("append_dim", None)

    if append_dim is not None:
        mode = kwargs.pop("mode", None)
        if mode is None:
            mode = "a" if os.path.exists(path) else "w"

        LOG.info(f"Saving data as zarr at {path} (mode={mode}, append_dim={append_dim})")
        if mode == "a":
            ds.to_zarr(path, mode=mode, append_dim=append_dim, zarr_format=2, **kwargs)
        elif mode == "w":
            ds.to_zarr(path, mode=mode, zarr_format=2, **kwargs)
        else:
            LOG.error(f"Unsupported mode '{mode}' for zarr saving. Use 'a' for append or 'w' for write.")
            raise ValueError(f"Unsupported mode '{mode}' for zarr saving. Use 'a' for append or 'w' for write.")
    else:
        LOG.info(f"Saving data as zarr-file at {path}")
        ds.to_zarr(path, zarr_format=2, **kwargs)

def save_as_verif(ds : xr.Dataset | xr.DataArray, path : str) -> None:
    """Save an xarray dataset or array in the verif format.
    Args:
        ds (xarray.Dataset | xarray.DataArray): The xarray dataset or array to save
        path (str): Path where the verif file will be saved
    Returns:
        None
    Examples:
        >>> ds = xarray.Dataset(...)
        >>> save_as_verif(ds, "output.verif")
    """ 
    LOG.error("Saving to verif-format not yet supported")
    raise NotImplementedError