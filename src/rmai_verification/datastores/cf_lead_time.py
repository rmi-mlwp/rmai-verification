"""
Forecasts stored as CF files, one file per (init month, lead time).

This is the layout readFA's ``anemoi_to_CF_month`` writes (used by the
AI-extreme-precipitation export): for every lead time L there is one file per
month holding the fields of all inits of that month at lead L, on a 2D
``(time, y, x)`` grid, with ``time`` the VALID time (init + L).

The datastore turns that back into the layout every other forecast store
uses -- ``(reference_time, lead_time, grid_index)`` with a 2D ``valid_time``
coordinate -- so alignment, metrics and output need no changes. The spatial
grid is flattened in C order (y outer, x inner), which is the point order of
the flat anemoi ``values`` dimension the export was made from.

Unlike the other stores it finds its own files: the path template has a
``{lead}`` field as well as ``{yyyy}``/``{mm}``, and which files a batch needs
follows from the batch's init times, not from one file per init.
"""

import logging
from datetime import datetime
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .anemoi_inference import AnemoiInference
from ..grids.grid_mapping import add_xy
from ..utils.time import to_timedelta64

LOG = logging.getLogger(__name__)

# CF bookkeeping that is not data and would otherwise travel into the metrics.
DROP_VARS = ["time_bnds", "lat_bounds", "lon_bounds", "crs", "lat", "lon"]

# Per-file dask chunking. The export writes pr with on-disk chunks of
# (186, 106, 106); reading whole (y, x) planes in blocks of 186 time steps
# keeps each dask chunk aligned with whole disk chunks (~132 MB float32).
DEFAULT_CHUNKS = {"time": 186, "y": -1, "x": -1}


class CFLeadTime(AnemoiInference):
    """Forecast datastore for per-(month, lead time) CF files.

    Config (under ``datastores:``)::

        my-experiment:
          type: cf-lead-time
          path: /.../pr/<exp>-only-pr-month-{yyyy}-{mm}_lead-time-{lead:02d}hrs_pr_{yyyy}_{mm}.nc
          mapping: alaro4km
          variables: [pr]
          lead_times: [1, 72]     # first, last lead time in hours (inclusive)

    The ``{yyyy}``/``{mm}`` fields are the INIT month. An init whose file or
    valid time is absent (e.g. December 2022, which stops at 2022-12-29T01)
    raises unless ``allow_missing: true``, in which case it is NaN and drops out
    of the scores.

    Inherits ``unstack`` and the base-class properties from AnemoiInference;
    only how the data is opened differs.
    """

    # Tells load_datastores to hand over the path template and the batch dates
    # instead of a per-init file list.
    finds_own_files = True

    def __init__(
        self,
        path_fmt: str,
        start: np.datetime64,
        end: np.datetime64,
        frequency: np.timedelta64,
        variables: Union[List[str], Tuple[str], set] = None,
        mapping: Union[Dict[str, str], str] = None,
        lead_times: Sequence[int] = (1, 72),
        chunks: Optional[Dict[str, Any]] = None,
        allow_missing: bool = False,
        engine: str = "netcdf4",
    ):
        LOG.info("Initializing CFLeadTime datastore")
        self._mapping = mapping
        self._stacked = True
        self._chunks = dict(chunks) if chunks is not None else DEFAULT_CHUNKS.copy()
        self._engine = engine

        if len(lead_times) != 2:
            raise ValueError(f"lead_times must be [first, last] in hours, got {lead_times}")
        first, last = (int(x) for x in lead_times)
        self._lead_hours = list(range(first, last + 1))

        start = np.datetime64(start, "ns")
        end = np.datetime64(end, "ns")
        frequency = np.timedelta64(to_timedelta64(frequency) if isinstance(frequency, str) else frequency, "ns")
        self._inits = np.arange(start, end + frequency, frequency).astype("datetime64[ns]")
        if self._inits.size == 0:
            raise ValueError(f"No init times between {start} and {end}")

        self._data = self._open(path_fmt, variables, allow_missing)

        if self._mapping:
            self._data = add_xy(self._data, self._mapping)
        LOG.info("Finished initializing CFLeadTime datastore")

    # ------------------------------------------------------------------ open
    def _months(self) -> List[Tuple[int, int]]:
        months = self._inits.astype("datetime64[M]")
        return [(int(str(m)[:4]), int(str(m)[5:7])) for m in np.unique(months)]

    def _open_lead(self, path_fmt, lead, variables, allow_missing) -> xr.Dataset:
        files = []
        for yyyy, mm in self._months():
            path = path_fmt.format(yyyy=f"{yyyy:04d}", yy=f"{yyyy % 100:02d}", mm=f"{mm:02d}", lead=lead)
            if os.path.exists(path):
                files.append(path)
            elif allow_missing:
                LOG.warning("Missing %s (lead %d h); its inits will be NaN", path, lead)
            else:
                raise FileNotFoundError(f"Forecast file missing for lead {lead} h: {path}")
        if not files:
            raise FileNotFoundError(f"No forecast files for lead {lead} h for inits {self._inits[0]}..{self._inits[-1]}")

        # drop_variables at open time, not drop_vars afterwards: decoding the
        # CF bounds variables made each open ~40x slower (0.8 s vs 0.02 s per
        # file), and a batch opens 72+ files.
        with warnings.catch_warnings():
            # The 186-step dask chunks start at each file's first time step, so
            # they do line up with the on-disk chunks; xarray warns regardless.
            warnings.filterwarnings("ignore", message="The specified chunks separate")
            parts = [xr.open_dataset(f, engine=self._engine, chunks=self._chunks,
                                     drop_variables=DROP_VARS) for f in files]

        valid = self._inits + np.timedelta64(lead, "h")
        # Cut each file to the batch's run of valid times BEFORE concatenating.
        # A reindex over the concatenated files that spans two dask chunks
        # (e.g. a Feb 24 - Mar 3 batch) becomes dask's shuffle-based take:
        # shuffle tasks over every chunk of both files, output chunks re-sized
        # across the file boundary, and many leads held in memory at once. That
        # OOM-killed all 31 CF campaign jobs (> 24 GB, 2026-11-06). A slice is a
        # plain getitem and keeps the file boundary as the chunk boundary.
        sliced = []
        for part in parts:
            t = part["time"].values
            pos = np.searchsorted(t, valid)
            hit = (pos < t.size) & (t[np.minimum(pos, t.size - 1)] == valid)
            if not hit.any():
                continue
            p = pos[hit]
            if not (np.diff(p) == 1).all():
                sliced = None
                break
            sliced.append(part.isel(time=slice(int(p[0]), int(p[-1]) + 1)))
        if sliced:
            parts = sliced
        elif sliced is None:
            LOG.warning("Lead %d h: valid times are not one contiguous run per file; "
                        "using the (slower, memory-hungry) reindex over whole files", lead)
        ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time", data_vars="minimal",
                                                        coords="minimal", compat="override")
        if sliced:
            # One chunk per lead for the whole batch (a plain concatenation of
            # the sliced pieces, ~132 MB for 186 inits), whatever the file and
            # chunk boundaries: every batch then has the layout of the January
            # batches the pilot measured. Split batches (114 + 72) peaked at
            # 18 GB RSS instead of ~10 GB (memory profile, job 23517285).
            ds = ds.chunk({"time": -1})
        # Encoding keys left in attrs break zarr writes downstream.
        for v in ds.data_vars:
            for k in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
                ds[v].attrs.pop(k, None)
        if variables:
            missing = [v for v in variables if v not in ds.data_vars]
            if missing:
                raise ValueError(f"Variables {missing} not in {files[0]}")
            ds = ds[list(variables)]

        present = np.isin(valid, ds["time"].values)
        if not present.all():
            msg = (f"{(~present).sum()} of {valid.size} inits have no valid time at lead {lead} h "
                   f"(first missing init {self._inits[~present][0]})")
            if not allow_missing:
                raise KeyError(msg + "; set allow_missing: true to score them as NaN")
            LOG.warning(msg)
        # After slicing, the time axis already equals `valid` unless times are
        # missing; only then reindex (on the short, sliced array): a missing
        # valid time becomes NaN instead of a KeyError.
        if ds.sizes["time"] != valid.size or not (ds["time"].values == valid).all():
            ds = ds.reindex(time=valid)
        ds = ds.assign_coords(time=self._inits).rename({"time": "reference_time"})
        return ds

    def _open(self, path_fmt, variables, allow_missing) -> xr.Dataset:
        per_lead = [self._open_lead(path_fmt, lead, variables, allow_missing) for lead in self._lead_hours]

        # Grid coordinates from the first file, flattened in C order.
        first_lead = self._lead_hours[0]
        months = self._months()
        probe = None
        for yyyy, mm in months:
            path = path_fmt.format(yyyy=f"{yyyy:04d}", yy=f"{yyyy % 100:02d}", mm=f"{mm:02d}", lead=first_lead)
            if os.path.exists(path):
                probe = path
                break
        with xr.open_dataset(probe, engine=self._engine,
                             drop_variables=[v for v in DROP_VARS if v not in ("lat", "lon")]) as g:
            lat = g["lat"].values.astype(np.float32).ravel()
            lon = g["lon"].values.astype(np.float32).ravel()
            ny, nx = g.sizes["y"], g.sizes["x"]

        lead_td = np.array([np.timedelta64(h, "h") for h in self._lead_hours]).astype("timedelta64[ns]")
        ds = xr.concat(per_lead, dim=xr.DataArray(lead_td, dims="lead_time", name="lead_time"),
                       data_vars="all", coords="minimal", compat="override", join="exact")
        ds = ds.drop_vars(["x", "y"], errors="ignore")
        ds = ds.stack(grid_index=("y", "x"), create_index=False)
        ds = ds.transpose("reference_time", "lead_time", "grid_index")
        assert ds.sizes["grid_index"] == ny * nx == lat.size

        ds = ds.assign_coords(
            {
                "grid_index": ("grid_index", np.arange(ny * nx)),
                "valid_time": (["reference_time", "lead_time"],
                               self._inits[:, np.newaxis] + lead_td[np.newaxis, :]),
                "longitude": ("grid_index", lon),
                "latitude": ("grid_index", lat),
            }
        )
        ds["reference_time"].attrs["standard_name"] = "forecast_reference_time"
        ds.attrs = {"is_observation": False}
        LOG.info("CFLeadTime: %d inits x %d lead times x %d points, vars %s",
                 ds.sizes["reference_time"], ds.sizes["lead_time"], ds.sizes["grid_index"], list(ds.data_vars))
        return ds
