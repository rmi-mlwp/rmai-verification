import xarray as xr
import logging
import gc

from typing import Dict, Optional

from . import comparative
from . import diagnostic
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
    "comparative" : {
        "rmse": comparative.rmse,
        "mse": comparative.mse,
        "bias": comparative.bias,
        "crps": comparative.crps,
        "skill": comparative.skill,
        "ssr": comparative.SSR
    },
    "diagnostic" : {       
        "frequency_spectrum": diagnostic.spatial_frequency_spectrum,
        "checkerboard": diagnostic.checkerboard,
        "spread": diagnostic.spread
    }
}
def calculate_metrics(reference: xr.Dataset, dict_of_datasets: Dict[str, xr.Dataset], comparative: Dict[str, Dict], diagnostic: Dict[str, Dict], large_memory: bool = False) -> xr.Dataset:
    """
    Calculate specified metrics between a reference dataset and multiple model datasets.
 
    This function computes selected verification metrics comparing a reference dataset
    against multiple model datasets. Metrics are organized by type (comparative vs diagnostic).
    All metrics are combined into a single Dataset for easy saving to zarr.
    
    Parameters
    ----------
    reference : xr.Dataset
        The reference/observation dataset to compare against
    dict_of_datasets : Dict[str, xr.Dataset]
        Dictionary containing the model datasets to evaluate, with model names as keys
    comparative : Dict[str, Dict]
        Configuration for the comparative metrics
    diagnostic : Dict[str, Dict]
        Configuratio for the diagnostic metrics
    large_memory : bool, optional
        If True, compute metrics variable-by-variable to reduce memory usage. Default is False.
        
    Returns
    -------
    xr.Dataset
        Combined Dataset containing all computed metrics as data variables.
        All metrics are merged into a single Dataset with variable names following the pattern:
        {metric_name}_{variable_name}
        
    Notes
    -----
    - For comparative metrics, both reference and model are aligned
    - For diagnostic metrics, only the model is used
    """
    
    # Debug and log information
    LOG.debug("Available model names: %s", list(dict_of_datasets.keys()))
    LOG.info("Chunks for reference dataset: %s", reference.chunks)
    LOG.info("Chunks for model datasets: %s", {name: model.chunks for name, model in dict_of_datasets.items()})
    
    # Initialize all metrics dictionary
    all_metrics = {}
 
    # Process comparative metrics if available
    if comparative:
        LOG.info("=== Processing comparative metrics ===")
        
        # Extract package and metrics
        metrics_list = comparative.get("metrics", [])

        # Loop over each metric in the metric list
        for metric_config in metrics_list:
            # Extract metric parameters
            metric_name = metric_config["name"]
            metric_params = {k: v for k, v in metric_config.items() if k != "name"}

            LOG.info("Calculating metric: %s", metric_name)

            # Get the metric function from the registry
            metric_func = METRICS["comparative"][metric_name]

            # Store results for each model
            metric_results = {}

            # Calculate metric for each model
            for model_name, model in dict_of_datasets.items():
                LOG.info("Calculating %s for model %s", metric_name, model_name)
                
                # For comparative metrics, use both model and reference data
                LOG.info("Using model data and reference data for comparative metric")

                # Align both datasets for comparative metrics
                reference_aligned, model_aligned = xr.align(reference, model)

                # Calculate metric with both reference and model + any extra parameters
                result_lazy = metric_func(reference_aligned, model_aligned, **metric_params)

                # Clean up aligned datasets
                del reference_aligned, model_aligned

                # Result is only loaded lazily, compute the active result based on the large_memory flag
                if large_memory:
                    # Compute per variable (if variables available)
                    if isinstance(result_lazy, xr.DataArray):
                        LOG.info("Computing %s for DataArray in model %s", metric_name, model_name)
                        metric_results[model_name] = result_lazy.compute()
                    else:
                        LOG.info("Computing %s for Dataset in model %s (variable-by-variable)", metric_name, model_name)
                        computed_vars = {}
                        var_names = list(result_lazy.data_vars)

                        for var_name in var_names:
                            LOG.info("  - Computing variable %s", var_name)
                            computed_vars[var_name] = result_lazy[var_name].compute()
                            clear_gpu_memory()

                        # Reconstruct dataset with computed variables
                        metric_results[model_name] = xr.Dataset(computed_vars, attrs=result_lazy.attrs)
                else:
                    # No seperate computation needed
                    metric_results[model_name] = result_lazy.compute()
                clear_gpu_memory()

            # Concatenate results across models for this metric
            all_metrics[metric_name] = concat_dict_along_keys(metric_results, "model")
            clear_gpu_memory()


    # Process diagnostic metrics if available
    if diagnostic:
        LOG.info("=== Processing diagnostic metrics ===")
        
        # Extract metrics
        metrics_list = diagnostic.get("metrics", [])

        # Loop over each metric in the metric list
        for metric_config in metrics_list:
            # Extract metric parameters
            metric_name = metric_config["name"]
            metric_params = {k: v for k, v in metric_config.items() if k != "name"}

            LOG.info("Calculating metric: %s (type: diagnostic)", metric_name)

            # Get the metric function from the registry
            metric_func = METRICS["diagnostic"][metric_name]

            # Store results for each model
            metric_results = {}

            # Calculate metric for each model
            for model_name, model in dict_of_datasets.items():
                LOG.info("Calculating %s for model %s", metric_name, model_name)

                # For diagnostic metrics, only use the model data
                LOG.info("Using model data directly for diagnostic metric")

                # Calculate metric with only model + any extra parameters
                result_lazy = metric_func(model, **metric_params)
                
                # Result is only loaded lazily, compute the active result based on the large_memory flag
                if large_memory:
                    # Compute per variable (if variables available)
                    if isinstance(result_lazy, xr.DataArray):
                        LOG.info("Computing %s for DataArray in model %s", metric_name, model_name)
                        metric_results[model_name] = result_lazy.compute()
                    else:
                        LOG.info("Computing %s for Dataset in model %s (variable-by-variable)", metric_name, model_name)
                        computed_vars = {}
                        var_names = list(result_lazy.data_vars)

                        for var_name in var_names:
                            LOG.info("  - Computing variable %s", var_name)
                            computed_vars[var_name] = result_lazy[var_name].compute()
                            clear_gpu_memory()

                        # Reconstruct dataset with computed variables
                        metric_results[model_name] = xr.Dataset(computed_vars, attrs=result_lazy.attrs)
                else:
                    # No seperate computation needed
                    metric_results[model_name] = result_lazy.compute()
                clear_gpu_memory()

            # Concatenate results across models for this metric
            all_metrics[metric_name] = concat_dict_along_keys(metric_results, "model")
            clear_gpu_memory()
    
    LOG.info("All metrics computed. Keys: %s", list(all_metrics.keys()))
    
    # Combine all metrics into a single Dataset
    LOG.info("Combining all metrics into single Dataset")

    combined_dataset = concat_dict_along_keys(all_metrics, "metric")

    LOG.info("Final combined dataset shape: %s", combined_dataset.dims)
    LOG.info("Final combined dataset variables: %s", list(combined_dataset.data_vars))
    
    return combined_dataset