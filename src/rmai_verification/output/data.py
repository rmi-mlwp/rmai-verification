import numpy as np
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

    # Encoding keys sitting in a variable's *attrs* (e.g. _FillValue on a
    # datetime coordinate, carried over from a source zarr) are rejected by
    # xarray when it encodes for an append: the first batch writes, every later
    # one fails. They describe the source's serialisation, not this output's.
    ds = ds.copy()
    for var in ds.variables.values():
        for key in ("_FillValue", "missing_value", "scale_factor", "add_offset", "dtype"):
            var.attrs.pop(key, None)

    if append_dim is not None:
        mode = kwargs.pop("mode", None)
        if mode is None:
            mode = "a" if os.path.exists(path) else "w"

        LOG.info(f"Saving data as zarr at {path} (mode={mode}, append_dim={append_dim})")
        if mode == "a":
            ds.to_zarr(path, mode=mode, append_dim=append_dim, zarr_format=2, **kwargs)
        elif mode == "w":
            # Fix the encoding of a datetime append coordinate. Left to xarray,
            # it is inferred from the FIRST batch: a single init gives
            # "days since ..." as int64, and every later hourly init appended
            # is then rounded to whole days (2021-01-31T23 was stored as
            # 2021-02-01T22). Seconds since the epoch are exact for any
            # whole-second times. Appends reuse the store's encoding.
            if append_dim in ds.coords and np.issubdtype(ds[append_dim].dtype, np.datetime64) \
                    and "units" not in ds[append_dim].encoding:
                ds[append_dim].encoding.update(units="seconds since 1970-01-01 00:00:00",
                                               calendar="proleptic_gregorian", dtype="int64")
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