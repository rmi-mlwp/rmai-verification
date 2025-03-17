import xarray as xr
import numpy as np
import logging

from . import xskill
from typing import List, Dict
from utils.sanitation import concat_dict_along_keys

METRICS = {
    "xskillscore" : {
        "rmse": xskill.rmse,
        "mse": xskill.mse,
        "bias": xskill.bias,
    },
}

LOG = logging.getLogger(__name__)

def calculate_metrics(reference : xr.Dataset, dict_of_datasets: Dict[str, xr.Dataset], package : str, metrics : List[str], avg_dims : str | List[str] ):
    LOG.debug(dict_of_datasets.keys())
    metrics_dict = dict()
    for metric_name in metrics:
        _metric = dict()
        metric = METRICS[package][metric_name]
        for name, model in dict_of_datasets.items():
            LOG.info(f"Calulating {metric_name} for model {name}")
            _reference = reference.reindex(valid_time=np.unique(model["valid_time"].values.ravel()))
            _reference = _reference.sel(valid_time=model["valid_time"])
            print(f"Reference data dimension after aligning: {_reference.sizes}")
            _reference, model = xr.align(_reference,model)
            _metric[name]=metric(_reference,model,avg_dims)
        metrics_dict[metric_name] = concat_dict_along_keys(_metric, "model")
    return concat_dict_along_keys(metrics_dict, "metric")