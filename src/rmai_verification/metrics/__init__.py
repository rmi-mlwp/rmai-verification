import xarray as xr
import logging

from . import xskill, scrs, rmai
from typing import List, Dict
from ..utils.sanitation import concat_dict_along_keys

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
    "rmai" : {
        "act": rmai.act, # activity
        "nfa": rmai.nfa, #normalized forecast activity
        "acc": rmai.acc,
    },
}

LOG = logging.getLogger(__name__)

def calculate_metrics(reference : xr.Dataset,
                      dict_of_datasets: Dict[str, xr.Dataset], 
                      package : str, 
                      metrics : List[str], 
                      avg_dims : str | List[str],
                      climatology : str = "",
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
        climatology : str = None
            Name of the dataset to use as climatology with respect to which to compute anomalies. Optional.
        Returns
        -------
        xr.Dataset
            Dataset containing all calculated metrics, with dimensions 'model' and 'metric'
            concatenating results across models and metric types
        
        Notes
        -----
        - Input datasets are aligned before metric calculation
        - Metrics are computed using dask with .compute() call
        - Results are concatenated first across models, then across metrics
    """
    
    LOG.debug(dict_of_datasets.keys())
    metrics_dict = dict()
    #chunks = {dim: -1 for dim in avg_dims}
    #print(reference.chunk(chunks))
    if climatology:
        climatology_data = dict_of_datasets.pop(climatology)
        # probably also better to xr.align this? 
    for metric_name in metrics:
        _metric = dict()
        metric = METRICS[package][metric_name]
        for name, model in dict_of_datasets.items():
            LOG.info(f"Calulating {metric_name} for model {name}")
            reference, model = xr.align(reference,model)
            args = [model, reference, avg_dims]
            if climatology:
                args.insert(2, climatology_data)
            _metric[name]=metric(*args).compute()
        metrics_dict[metric_name] = concat_dict_along_keys(_metric, "model")
    return concat_dict_along_keys(metrics_dict, "metric")