
import logging
import xarray as xr
import os
from typing import Dict, List, Union


from ..alignment import align_reference_times, align_spatial, align_valid_times
from ..transformations import apply_transformations
from ..utils.sanitation import broadcast_nans, prep_config
from ..metrics import calculate_metrics
from ..output import save_dataset
from ..visualization import plot_overview
from ..utils.files import load_yaml
from ..datastores import select_variables, load_datastores, BaseDataStore

LOG = logging.getLogger(__name__)

VERIF_VARS = ["2t", "10s"]

class Verification():
    """A class for handling data verification and visualization workflows.
    This class manages the entire verification process including data alignment,
    metric calculations, and visualization. It handles multiple datastores,
    applies transformations, and generates comparison metrics between a reference
    dataset and other datasets.
    Args:
        config (Union[str, dict]): Configuration for the verification process.
            Can be either a path to a YAML file or a dictionary containing the configuration.
            Must include the following keys:
            - dates: Dict with 'start', 'end', and 'frequency' keys
            - output: Dict with optional 'type' key
            - datastores: Dict of datastore configurations
            - verification: Dict with 'reference_datastore' and optional 'variables'
            - transformations: Dict of transformations to apply
            - clusters: Dict of cluster configurations for metrics calculation
            - visualization: Dict of visualization parameters

    Methods:
        align_datastores(): Aligns all datastores in time and space
        calculate_clusters(): Calculates verification metrics for defined clusters
        visualize_clusters(): Creates visualization for calculated metrics
        verify(): Executes the complete verification workflow
    """

    def __init__(self,config: Union[str, dict]) -> None:
        """Initialize the Verification class.
        This constructor sets up a verification instance based on a configuration that specifies
        data sources, date ranges, and transformations to be applied.
        Args:
            config (Union[str, dict]): Configuration for the verification process. Can be either:
                - A string path to a YAML configuration file
                - A dictionary containing the configuration
        
        The configuration must include:
            - dates: Dictionary with 'start', 'end', and 'frequency' keys
            - output: Dictionary with optional 'type' key
            - datastores: Dictionary defining data sources
            - verification: Dictionary containing 'reference_datastore'
            - transformations: List of transformations to apply to datastores

        Attributes:
            _config (dict): Stored configuration
            _start: Start date from config
            _end: End date from config
            _frequency: Time frequency from config
            _output_type: Output type from config
            _datastores: Loaded data stores
            _reference_datastore: Reference data store name
        """

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
        self._climatology_datastore = config["verification"].get("climatology_datastore", "")

        apply_transformations(
            datastores=self._datastores, 
            transformations=self._config["transformations"]
        )

    def align_datastores(self) -> None:
        """Align all datastores in time and space.
        This method selects the specified variables from the datastores,
        aligns the reference times, and aligns the spatial dimensions.
        It also handles the selection of variables based on the configuration.
        The reference datastore is used as the baseline for spatial alignment.
        """

        select_variables(
            datastores=self._datastores,
            variables=self._config["verification"].get("variables", VERIF_VARS)
        )
        align_reference_times(
            datastores=self._datastores
        )
        align_valid_times(
            datastores=self._datastores
        )

        align_spatial_kwargs = dict(
            interpolation=self._config.get("interpolation",dict()),
            regrid=self._config.get("regrid",dict())
        )
        
        self._aligned_data = align_spatial(
            datastores=self._datastores,
            reference_datastore=self._reference_datastore,
            kwargs=align_spatial_kwargs
        )
        
    def calculate_clusters(self) -> None:
        """Calculate verification metrics for defined clusters.
        
        This method iterates through the clusters defined in the configuration,
        calculates the metrics using the reference datastore, and saves the results.
        It also handles the output configuration for each cluster.
        """

        reference = self._aligned_data.pop(self._reference_datastore)
        if self._config.get('broadcast_nans', True):
            broadcast_nans(list(self._aligned_data.values()))
        
        clusters = dict()
        for cluster, config in self._config["verification"]["clusters"].items():
            clusters[cluster] = calculate_metrics(
                reference=reference,
                climatology=self._climatology_datastore,
                dict_of_datasets=self._aligned_data,
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

    def visualize_clusters(self) -> None:
        """Visualizes the metrics for each cluster using the specified configuration.

        This method iterates through all clusters and their associated metrics, preparing
        visualization configurations and generating plots using the plot_overview function.
        Each cluster's visualizations are saved in the specified directory with the given prefix.
        The visualization configuration is extracted from self._config["visualization"] and can be
        customized per cluster. The method handles the directory structure and file naming
        conventions automatically.
        
        Returns:
            None
        Note:
            - The visualization settings should be defined in self._config["visualization"]
            - Output files will be saved in the directory specified in the config
            - Default directory is "./" if not specified
            - Default prefix is the cluster name if not specified
        """

        if not (self._config.get("visualization",None) == None):
            for cluster, metrics in self._clusters.items():
                LOG.info(f"Visualizing cluster {cluster}")
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
        else:
            LOG.info("No visualization configuration found. Skipping visualization.")
    
    def verify(self):
        self.align_datastores()
        self.calculate_clusters()
        self.visualize_clusters()
        




