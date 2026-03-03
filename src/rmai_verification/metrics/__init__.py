import xarray as xr
import logging
import gc

from . import xskill, scrs
from typing import List, Dict, Optional
from ..utils.sanitation import concat_dict_along_keys

LOG = logging.getLogger(__name__)

def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(n)
    for u in units:
        if n < 1024 or u == units[-1]:
            return f"{n:,.2f} {u}"
        n /= 1024
    return f"{n:,.2f} B"


def log_xarray_chunks(obj, name: str, max_vars: int = 10) -> None:
    """
    Log chunk structure + estimated bytes per chunk for an xarray Dataset/DataArray.
    Works for NumPy-backed, dask-backed, and CuPy-backed dask chunks.

    - For Dataset: logs up to max_vars variables.
    - For DataArray: logs that array.
    """
    try:
        import xarray as xr
    except Exception:
        LOG.info("[%s] xarray not available?", name)
        return

    def _log_da(da: xr.DataArray, label: str) -> None:
        """Log chunk structure and estimated bytes per chunk for a single DataArray."""
        data = da.data
        chunks = getattr(data, "chunks", None)
        if chunks is None:
            # eager array
            try:
                nbytes = getattr(data, "nbytes", None)
            except Exception:
                nbytes = None
            LOG.info("[%s] %s: eager array, shape=%s dtype=%s nbytes=%s",
                     name, label, da.shape, da.dtype, _fmt_bytes(nbytes))
            return

        # dask array
        dims = da.dims
        # chunk_sizes = {d: list(chunks[i]) for i, d in enumerate(dims)}
        chunk_sizes = {d: len(chunks[i]) for i, d in enumerate(dims)}

        # estimate bytes per *one* chunk (use the first chunk size in each dim)
        try:
            import numpy as np
            dtype_nbytes = np.dtype(da.dtype).itemsize
        except Exception:
            dtype_nbytes = None

        if dtype_nbytes is not None and len(dims) > 0:
            first_chunk_elems = 1
            for i in range(len(dims)):
                first_chunk_elems *= int(chunks[i][0])
            bytes_per_chunk = first_chunk_elems * dtype_nbytes
        else:
            bytes_per_chunk = None

        # also estimate how many chunks total
        try:
            n_chunks_total = 1
            for i in range(len(dims)):
                n_chunks_total *= len(chunks[i])
        except Exception:
            n_chunks_total = None

        LOG.info("[%s] %s: dask array, shape=%s dtype=%s",
                 name, label, da.shape, da.dtype)
        LOG.info("[%s] %s: chunks_by_dim=%s", name, label, chunk_sizes)
        LOG.info("[%s] %s: approx bytes/chunk=%s, approx #chunks=%s",
                 name, label, _fmt_bytes(bytes_per_chunk), n_chunks_total)

        # also log if it's a CuPy-backed dask array
        try:
            import cupy as cp
            if isinstance(data, cp.ndarray):
                LOG.info("[%s] %s: CuPy-backed dask array", name, label)
        except Exception:
            pass

        # also log total size of the dask array if possible
        try:
            nbytes = data.nbytes
            LOG.info("[%s] %s: total dask array size=%s", name, label, _fmt_bytes(nbytes))
        except Exception:
            pass

    if isinstance(obj, xr.DataArray):
        _log_da(obj, "DataArray")
        return

    if isinstance(obj, xr.Dataset):
        LOG.info("[%s] Dataset vars=%d dims=%s", name, len(obj.data_vars), dict(obj.sizes))
        for i, v in enumerate(obj.data_vars):
            if i >= max_vars:
                LOG.info("[%s] ... (skipping remaining vars; max_vars=%d)", name, max_vars)
                break
            _log_da(obj[v], f"var={v}")
        return

    LOG.info("[%s] Not an xarray object: %s", name, type(obj))


def clear_gpu_memory():
    """Clear GPU memory if CuPy is being used.
    
    This function attempts to free all GPU memory blocks to prevent
    out-of-memory errors during long computations. It safely handles
    cases where CuPy is not installed or GPU is not available.
    """
    try:
        import cupy as cp
        # Free all GPU memory blocks
        cp.get_default_memory_pool().free_all_blocks()
        # Also run garbage collection
        gc.collect()
        LOG.debug("GPU memory cleared")
    except (ImportError, RuntimeError):
        # CuPy not available or GPU not accessible, just run CPU GC
        gc.collect()

METRICS = {
    "xskillscore" : {
        "rmse": xskill.rmse,
        "mse": xskill.mse,
        "bias": xskill.bias,
    },
    "scores" : {
        "rmse": scrs.rmse,
        "mse": scrs.mse,
        "bias": scrs.bias,
    },
}

def calculate_metrics(reference : xr.Dataset,
                    
                      dict_of_datasets: Dict[str, xr.Dataset], 
                      package : str, 
                      metrics : List[str], 
                      avg_dims : str | List[str],
                      large_memory: bool = False
                      ) -> xr.Dataset:
    """Calculate specified metrics between a reference dataset and multiple model datasets.

        This function computes selected verification metrics comparing a reference dataset
        against multiple model datasets. The metrics are calculated along specified dimensions
        and concatenated into a single dataset.
        
        Parameters
        ----------
        reference : xr.Dataset
            The reference/observation dataset to compare against
        dict_of_datasets : Dict[str, xr.Dataset]
            Dictionary containing the model datasets to evaluate, with model names as keys
        package : str
            Name of the metrics package to use (must be defined in METRICS)
        metrics : List[str] 
            List of metric names to calculate
        avg_dims : str | List[str]
            Dimension(s) over which to average the metrics
        
        Returns
        -------
        xr.Dataset
            Dataset containing all calculated metrics, with dimensions 'model' and 'metric'
            concatenating results across models and metric types
        
        Notes
        -----
        - Input datasets are aligned before metric calculation
        - Metrics are computed variable-by-variable to reduce memory usage
        - Results are concatenated first across models, then across metrics
    """
    
    LOG.debug(dict_of_datasets.keys())
    metrics_dict = dict()
    #chunks = {dim: -1 for dim in avg_dims}
    #print(reference.chunk(chunks))
    LOG.info("Chunks for reference dataset: %s", reference.chunks)
    LOG.info(
        "Chunks for model datasets: %s",
        {name: model.chunks for name, model in dict_of_datasets.items()},
    )
    LOG.info("Type of reference dataset: %s", type(reference))
    LOG.info(
        "Type of model datasets: %s",
        {name: str(type(model)) for name, model in dict_of_datasets.items()},
    )
    LOG.info(f"Calculating metrics: {metrics} using package: {package} with avg_dims: {avg_dims}")
    LOG.info(f"Large memory mode: {large_memory}")

    LOG.info("=== Chunk diagnostics BEFORE metrics ===")
    log_xarray_chunks(reference, "reference", max_vars=20)
    for model_name, model in dict_of_datasets.items():
        log_xarray_chunks(model, f"model={model_name}", max_vars=20)

    for metric_name in metrics:
        _metric = dict()
        metric = METRICS[package][metric_name]
        for name, model in dict_of_datasets.items():
            LOG.info(f"Calculating {metric_name} for model {name}")
            reference_aligned, model_aligned = xr.align(reference, model)
            # copy=False
            LOG.info("=== Chunk diagnostics AFTER xr.align for model=%s ===", name)
            log_xarray_chunks(reference_aligned, f"reference_aligned({name})", max_vars=20)
            log_xarray_chunks(model_aligned, f"model_aligned({name})", max_vars=20)
            
            if large_memory:
                # Compute metrics variable-by-variable to reduce memory usage
                # This prevents OOM errors when working with large datasets
                result_lazy = metric(reference_aligned, model_aligned, avg_dims)
                
                # Free aligned datasets early to reduce memory pressure
                del reference_aligned, model_aligned
                
                # Handle both Dataset and DataArray return types
                if isinstance(result_lazy, xr.DataArray):
                    # For DataArray, just compute directly
                    LOG.info(f"Computing {metric_name} for DataArray in model {name}")
                    _metric[name] = result_lazy.compute()
                    clear_gpu_memory()
                else:
                    # For Dataset, process each data variable separately
                    computed_vars = {}
                    # Get all variable references first to avoid repeated graph traversals
                    var_names = list(result_lazy.data_vars)
                    for var_name in var_names:
                        LOG.info(f"Computing {metric_name} for variable {var_name} in model {name}")
                        var_data = result_lazy[var_name]
                        computed_vars[var_name] = var_data.compute()
                        # Clear GPU memory after each variable computation
                        clear_gpu_memory()
                    
                    # Reconstruct the dataset with computed variables
                    _metric[name] = xr.Dataset(computed_vars, attrs=result_lazy.attrs)
                
                # Free the lazy result to avoid holding references
                del result_lazy
            else:
                _metric[name]=metric(reference_aligned, model_aligned, avg_dims).compute()
            
            # Clear GPU memory after each metric computation to prevent OOM
            clear_gpu_memory()
        metrics_dict[metric_name] = concat_dict_along_keys(_metric, "model")
        # Clear GPU memory after concatenating metrics for a metric type
        clear_gpu_memory()
    return concat_dict_along_keys(metrics_dict, "metric")