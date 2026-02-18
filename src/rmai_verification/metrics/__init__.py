import xarray as xr
import logging
import gc

from . import xskill, scrs
from typing import List, Dict
from ..utils.sanitation import concat_dict_along_keys

LOG = logging.getLogger(__name__)

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
                      avg_dims : str | List[str]
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
    for metric_name in metrics:
        _metric = dict()
        metric = METRICS[package][metric_name]
        for name, model in dict_of_datasets.items():
            LOG.info(f"Calulating {metric_name} for model {name}")
            reference_aligned, model_aligned = xr.align(reference, model)
            
            # Compute metrics variable-by-variable to reduce memory usage
            # This prevents OOM errors when working with large datasets
            result_lazy = metric(reference_aligned, model_aligned, avg_dims)
            
            # Handle both Dataset and DataArray return types
            if isinstance(result_lazy, xr.DataArray):
                # For DataArray, just compute directly
                LOG.debug(f"Computing {metric_name} for DataArray in model {name}")
                _metric[name] = result_lazy.compute()
                clear_gpu_memory()
            else:
                # For Dataset, process each data variable separately
                computed_vars = {}
                for var_name in result_lazy.data_vars:
                    LOG.debug(f"Computing {metric_name} for variable {var_name} in model {name}")
                    computed_vars[var_name] = result_lazy[var_name].compute()
                    # Clear GPU memory after each variable computation
                    clear_gpu_memory()
                
                # Reconstruct the dataset with computed variables
                _metric[name] = xr.Dataset(computed_vars, attrs=result_lazy.attrs)
            
            # Clear GPU memory after each model
            clear_gpu_memory()
        metrics_dict[metric_name] = concat_dict_along_keys(_metric, "model")
        # Clear GPU memory after concatenating metrics for a metric type
        clear_gpu_memory()
    return concat_dict_along_keys(metrics_dict, "metric")