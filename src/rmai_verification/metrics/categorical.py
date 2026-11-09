"""
Categorical (threshold-exceedance) verification.

Stores the 2x2 contingency table rather than a score. Scores such as the
equitable threat score (ETS) are non-linear in the table, so the ETS of a
pooled table is not the mean of per-forecast ETS values. Keeping the raw
counts per (reference_time, lead_time, threshold) lets any aggregation --
over the whole period, per season, per lead-time bin, inside a bootstrap --
be done afterwards by summing counts and only then computing the score.

An event is ``value >= threshold``. A grid point counts only where both the
forecast and the reference are finite, so NaNs (e.g. pr at lead time 0) drop
out of every cell of the table instead of being read as "no event".

Sanity-check thresholds: at 0 every valid point is an event (hits == number of
valid points, the other three counts 0, ETS undefined); thresholds above the
data maximum (e.g. 200 mm/h) give hits == misses == false_alarms == 0.

Scores from the summed table (see ``scores_from_table``):

    ETS  = (hits - hits_random) / (hits + misses + false_alarms - hits_random)
    hits_random = (hits + misses) * (hits + false_alarms) / total
"""

from typing import List, Sequence
import logging

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

CATEGORIES = ["hits", "misses", "false_alarms", "correct_negatives"]

# Target input block size (bytes per input) for contingency_map, see there.
MAP_BLOCK_BYTES = 1024 * 1024**2


def contingency(
    obs: xr.Dataset | xr.DataArray,
    fcst: xr.Dataset | xr.DataArray,
    avg_dim: List[str],
    thresholds: Sequence[float],
) -> xr.Dataset | xr.DataArray:
    """Contingency-table counts, summed over ``avg_dim``.

    Called as ``metric(reference, model, avg_dims, **metric_kwargs)`` by
    ``calculate_metrics``, so the reference comes first.

    Returns the input type with two new dimensions, ``threshold`` and
    ``category`` (hits, misses, false_alarms, correct_negatives), as int64.

    All thresholds and counts come from ONE numpy kernel per block
    (``xr.apply_ufunc``, dask="parallelized"): each input block is read once
    and reduced in a single task. Composing the counts from xarray ops instead
    made ~10 graph layers per threshold, and with misaligned input chunks that
    became ~600k tasks per batch (ETS pilot OOM, job 23477403). ``avg_dim``
    must each be a single chunk (they are core dims). Thresholds are looped
    inside the kernel, so no (threshold x data) array is held in memory.
    """
    thresholds = _check_thresholds(thresholds)
    dims = list(avg_dim)
    out = xr.apply_ufunc(
        _contingency_kernel, obs, fcst,
        input_core_dims=[dims, dims],
        output_core_dims=[["threshold", "category"]],
        kwargs={"thresholds": np.asarray(thresholds), "n_core": len(dims)},
        dask="parallelized",
        output_dtypes=[np.int64],
        dask_gufunc_kwargs={"output_sizes": {"threshold": len(thresholds),
                                             "category": len(CATEGORIES)}},
        keep_attrs=False,
    )
    out = out.assign_coords(
        threshold=xr.DataArray(thresholds, dims="threshold",
                               attrs={"units": "same as the verified variable",
                                      "comment": "event is value >= threshold"}),
        category=xr.DataArray(CATEGORIES, dims="category"),
    )
    out = out.transpose("threshold", "category", ...)
    # Counts must not inherit the input variable's metadata: "units: mm h-1"
    # would be wrong, and encoding keys such as _FillValue in attrs make the
    # zarr append of every batch after the first fail.
    count_attrs = {"long_name": "contingency table counts", "units": "1"}
    if isinstance(out, xr.Dataset):
        for v in out.data_vars:
            out[v].attrs = dict(count_attrs)
            out[v].encoding = {}
    else:
        out.attrs = dict(count_attrs)
        out.encoding = {}
    LOG.info("Contingency counts for %d thresholds over dims %s", len(thresholds), dims)
    return out


def _contingency_kernel(o: np.ndarray, f: np.ndarray, thresholds: np.ndarray,
                        n_core: int) -> np.ndarray:
    """Counts over the last ``n_core`` axes; returns (..., threshold, category).

    A point counts only where both inputs are finite; an event is >= threshold.
    """
    axes = tuple(range(-n_core, 0))
    valid = np.isfinite(o) & np.isfinite(f)
    n_valid = valid.sum(axis=axes, dtype=np.int64)
    out = np.empty(o.shape[:o.ndim - n_core] + (thresholds.size, len(CATEGORIES)), np.int64)
    for k, t in enumerate(thresholds):
        oe = (o >= t) & valid
        fe = (f >= t) & valid
        hits = (oe & fe).sum(axis=axes, dtype=np.int64)
        n_obs = oe.sum(axis=axes, dtype=np.int64)
        n_fcst = fe.sum(axis=axes, dtype=np.int64)
        out[..., k, 0] = hits
        out[..., k, 1] = n_obs - hits
        out[..., k, 2] = n_fcst - hits
        out[..., k, 3] = n_valid - n_obs - n_fcst + hits
    return out


def contingency_map(
    obs: xr.Dataset | xr.DataArray,
    fcst: xr.Dataset | xr.DataArray,
    avg_dim: List[str],
    thresholds: Sequence[float],
) -> xr.Dataset | xr.DataArray:
    """Per-grid-point contingency counts, summed over the inits of a batch.

    The map counterpart of ``contingency``: instead of summing over space per
    (reference_time, lead_time), it sums over ``reference_time`` per
    (lead_time, grid point). ``avg_dim`` must be ``["reference_time"]``.
    Returns dims (threshold, category, lead_time, <space dims>), unsigned ints
    of the smallest width that holds the batch's init count (uint8 for <= 255
    inits). Verification sums the batches into an int32 running total (cluster
    option ``accumulate: true``).

    One kernel per (reference_time, lead_time) block produces that block's
    partial sums; dask then sums the partials over reference_time blocks.
    Same event and validity rules as ``contingency``.
    """
    if list(avg_dim) != ["reference_time"]:
        raise ValueError(f"contingency_map sums over reference_time only, got avg_dim={avg_dim}")
    thresholds = _check_thresholds(thresholds)
    if isinstance(obs, xr.Dataset):
        out = xr.Dataset({v: contingency_map(obs[v], fcst[v], avg_dim, thresholds)
                          for v in obs.data_vars if v in fcst.data_vars})
        return out

    import dask.array as da

    obs, fcst = xr.align(obs, fcst, join="exact")
    rest = [d for d in obs.dims if d != "reference_time"]
    o = obs.transpose("reference_time", *rest)
    f = fcst.transpose("reference_time", *rest)
    n_rt = o.sizes["reference_time"]
    dtype = np.uint8 if n_rt <= np.iinfo(np.uint8).max else (
        np.uint16 if n_rt <= np.iinfo(np.uint16).max else np.int64)
    od, fd = o.data, f.data
    if not hasattr(od, "dask"):
        od = da.from_array(od, chunks=-1)
    if not hasattr(fd, "dask"):
        fd = da.from_array(fd, chunks=od.chunks)
    idx = "abcdefghij"[:od.ndim]
    _, (od, fd) = da.core.unify_chunks(od, idx, fd, idx)
    # Each block yields a partial map as large as the block minus its
    # reference_time extent, and dask then sums the partials. With one init per
    # block (raw per-init files: 1 x 73 leads) that is 186 partials of ~620 MB
    # per batch -- ~115 GB of memory traffic, 438 s per batch (map test
    # 23498183). Merging consecutive inits into ~MAP_BLOCK_BYTES blocks is a
    # plain concatenation along reference_time (no crossing rechunk) and cuts
    # the partials ~10x. Blocks already long in reference_time (CF files:
    # 186 x 1 lead) are left as they are.
    per_init = od.dtype.itemsize * int(np.prod([c[0] for c in od.chunks[1:]]))
    n_merge = int(max(1, min(n_rt, MAP_BLOCK_BYTES // max(per_init, 1))))
    if max(od.chunks[0]) < n_merge:
        od = od.rechunk({0: n_merge})
        fd = fd.rechunk({0: n_merge})
    T = len(thresholds)
    partial = da.map_blocks(
        _map_kernel, od, fd,
        thresholds=np.asarray(thresholds), dtype_out=dtype,
        new_axis=[od.ndim, od.ndim + 1],
        chunks=((1,) * od.numblocks[0],) + od.chunks[1:] + ((T,), (len(CATEGORIES),)),
        dtype=dtype,
    )
    summed = partial.sum(axis=0, dtype=dtype, split_every=2)
    coords = {k: c for k, c in o.coords.items() if "reference_time" not in c.dims}
    res = xr.DataArray(summed, dims=rest + ["threshold", "category"], coords=coords)
    res = res.assign_coords(
        threshold=xr.DataArray(thresholds, dims="threshold",
                               attrs={"units": "same as the verified variable",
                                      "comment": "event is value >= threshold"}),
        category=xr.DataArray(CATEGORIES, dims="category"),
    ).transpose("threshold", "category", *rest)
    res.attrs = {"long_name": "contingency counts per grid point, summed over reference_time",
                 "units": "1"}
    return res


def _map_kernel(o: np.ndarray, f: np.ndarray, thresholds: np.ndarray, dtype_out) -> np.ndarray:
    """Partial map of one block: counts summed over axis 0 (kept, size 1).

    Unsigned arithmetic wraps, but every result is a true count in
    [0, block length] and so is exact modulo the dtype range.
    """
    valid = np.isfinite(o) & np.isfinite(f)
    n_valid = valid.sum(axis=0, dtype=dtype_out)
    out = np.empty((1,) + o.shape[1:] + (thresholds.size, len(CATEGORIES)), dtype_out)
    for k, t in enumerate(thresholds):
        oe = (o >= t) & valid
        fe = (f >= t) & valid
        hits = (oe & fe).sum(axis=0, dtype=dtype_out)
        n_obs = oe.sum(axis=0, dtype=dtype_out)
        n_fcst = fe.sum(axis=0, dtype=dtype_out)
        out[0, ..., k, 0] = hits
        out[0, ..., k, 1] = n_obs - hits
        out[0, ..., k, 2] = n_fcst - hits
        out[0, ..., k, 3] = n_valid - n_obs - n_fcst + hits
    return out


def _check_thresholds(thresholds: Sequence[float]) -> List[float]:
    if not thresholds:
        raise ValueError("contingency needs at least one threshold (metric_kwargs.thresholds)")
    thresholds = [float(t) for t in thresholds]
    if sorted(thresholds) != thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError(f"thresholds must be strictly increasing, got {thresholds}")
    return thresholds


def scores_from_table(table: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Derive categorical scores from (already summed) contingency counts.

    ``table`` must have a ``category`` dimension; sum it over whatever you are
    aggregating (reference_time, ...) *before* calling this. Returns the input
    type with ``category`` replaced by ``score``: ets, ts (threat score / CSI),
    pod, far, fbias. Undefined ratios (zero denominators) are NaN.
    """
    a = table.sel(category="hits").astype("float64")
    c = table.sel(category="misses").astype("float64")
    b = table.sel(category="false_alarms").astype("float64")
    d = table.sel(category="correct_negatives").astype("float64")
    n = a + b + c + d

    def ratio(num, den):
        return num / den.where(den != 0)

    a_random = ratio((a + c) * (a + b), n)
    scores = {
        "ets": ratio(a - a_random, a + b + c - a_random),
        "ts": ratio(a, a + b + c),
        "pod": ratio(a, a + c),
        "far": ratio(b, a + b),
        "fbias": ratio(a + b, a + c),
    }
    out = xr.concat(
        [s.drop_vars("category", errors="ignore") for s in scores.values()],
        dim=xr.DataArray(list(scores), dims="score", name="score"),
    )
    return out
