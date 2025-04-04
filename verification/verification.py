
import logging
import xarray as xr
import os
from typing import Dict, List

import homogenization
from homogenization import load_datastores, select_variables, apply_transformations

from utils.sanitation import broadcast_nans, prep_config

from metrics import calculate_metrics

from output import save_dataset

from visualization import plot_overview

from utils.files import load_yaml

LOG = logging.getLogger(__name__)

VERIF_VARS = ["2t", "10s"]

class Verification():
    def __init__(self,config):
        if isinstance(config, str):
            config = load_yaml(config)
        elif not isinstance(config, dict):
            LOG.ERROR("Unsupported config type")
            raise TypeError
        
        self._config = config
        self._start = config["dates"]["start"]
        self._end = config["dates"]["end"]
        self._frequency = config["dates"]["frequency"]
        self._output_type = config["output"].get("type",None)
        self._datastores = load_datastores(
            datastores=self._config["datastores"],
            start_date=self._start,
            end_date=self._end,
            frequency=self._frequency
        )
        self._reference_datastore = config["verification"]["reference_datastore"]
        apply_transformations(
            datastores=self._datastores, 
            transformations=self._config["transformations"]
        )

    def homogenize_datastores(self):
        select_variables(
            datastores=self._datastores,
            variables=self._config["verification"].get("variables", VERIF_VARS)
        )
        homogenization.reference_times(
            datastores=self._datastores
        )
        homogenization.valid_times(
            datastores=self._datastores,
            reference_datastore=self._reference_datastore
        )
        self._homogenized_data = homogenization.common_grid_or_points(
            datastores=self._datastores,
            reference_datastore=self._reference_datastore
        )
        
    def calculate_clusters(self):
        reference = self._homogenized_data.pop(self._reference_datastore)
        broadcast_nans(list(self._homogenized_data.values()))
        
        clusters = dict()
        for cluster, config in self._config["verification"]["clusters"].items():
            clusters[cluster] = calculate_metrics(
                reference=reference,
                dict_of_datasets=self._homogenized_data,
                **config
            )
            output_config = prep_config(self._config["output"], cluster)
            output_type = output_config.pop("type",None)
            if output_type:
                output_path = output_config.pop("path", f"{cluster}.{output_type}")
                save_dataset(
                    dataset=clusters[cluster],
                    type=output_type,
                    path=output_path,
                    **output_config,
                )
        self._clusters = clusters

    def visualize_clusters(self):
        if not (self._config.get("visualization",None) == None):
            for cluster, metrics in self._clusters.items():
                config = prep_config(self._config["visualization"],cluster)
                prefix = os.path.join(
                    config.pop("directory","./"),
                    config.pop("prefix",cluster)
                )
                plot_overview(
                    dataset = metrics,
                    prefix=prefix,
                    **config
                )
    
    def verify(self):
        self.homogenize_datastores()
        self.calculate_clusters()
        self.visualize_clusters()
        





    
        
        