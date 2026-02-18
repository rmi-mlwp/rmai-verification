
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

def maybe_to_cupy(obj, use_gpu: bool):
    if not use_gpu:
        return obj

    try:
        import cupy as cp
    except ImportError:
        LOG.warning("use_gpu=True but CuPy is not installed; falling back to CPU/NumPy.")
        return obj

    import numpy as np
    import xarray as xr

    def _convert_da(da: xr.DataArray) -> xr.DataArray:
        # Only move numeric arrays (avoid datetime/timedelta/object which CuPy may not support)
        if da.dtype.kind not in ("b", "i", "u", "f", "c"):
            return da

        data = da.data  # can be numpy or dask array
        if hasattr(data, "map_blocks"):
            # dask array -> dask array with cupy chunks
            new_data = data.map_blocks(cp.asarray, dtype=da.dtype)
        else:
            # eager numpy -> eager cupy
            new_data = cp.asarray(np.asarray(data))
        return da.copy(data=new_data)

    if isinstance(obj, xr.DataArray):
        return _convert_da(obj)

    if isinstance(obj, xr.Dataset):
        # Apply to each data_var (coords are left alone)
        return obj.map(_convert_da)

    # If it's something else (e.g. plain dask array), do a best-effort conversion
    if hasattr(obj, "map_blocks"):
        return obj.map_blocks(cp.asarray)
    return cp.asarray(obj)


def maybe_to_numpy(obj, use_gpu: bool):
    """
    If use_gpu=True, convert CuPy-backed data (eager or dask chunks) back to NumPy.
    If use_gpu=False, return obj unchanged.

    Supports: xarray.DataArray, xarray.Dataset, and falls back for dask arrays / cupy arrays.
    """
    if not use_gpu:
        return obj

    try:
        import cupy as cp
    except ImportError:
        # If CuPy isn't available, we can't have CuPy-backed arrays anyway.
        return obj

    import xarray as xr

    def _to_numpy_da(xda: xr.DataArray) -> xr.DataArray:
        data = xda.data

        # Eager CuPy -> eager NumPy
        if isinstance(data, cp.ndarray):
            return xda.copy(data=cp.asnumpy(data))  # or data.get()

        # Dask array (possibly with CuPy chunks)
        if hasattr(data, "map_blocks"):
            # Robust check: look at meta if available, otherwise try a tiny conversion
            meta = getattr(data, "_meta", None)
            if isinstance(meta, cp.ndarray):
                new = data.map_blocks(cp.asnumpy, dtype=xda.dtype)
                return xda.copy(data=new)

        return xda

    # xarray wrappers
    if isinstance(obj, xr.DataArray):
        return _to_numpy_da(obj)

    if isinstance(obj, xr.Dataset):
        # Only data variables are mapped; coords preserved
        return obj.map(_to_numpy_da)

    # Non-xarray fallbacks
    if isinstance(obj, cp.ndarray):
        return cp.asnumpy(obj)

    if hasattr(obj, "map_blocks"):
        # dask array: if meta is cupy, map back; else leave as-is
        meta = getattr(obj, "_meta", None)
        if isinstance(meta, cp.ndarray):
            return obj.map_blocks(cp.asnumpy, dtype=obj.dtype)
        return obj

    return obj


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

        use_gpu = bool(self._config.get("verification", {}).get("use_gpu", False))
        print(use_gpu)

        reference = self._aligned_data.pop(self._reference_datastore)
        broadcast_nans(list(self._aligned_data.values()))

        reference_x = maybe_to_cupy(reference, use_gpu)
        data_x = {k: maybe_to_cupy(v, use_gpu) for k, v in self._aligned_data.items()}
        
        clusters = dict()
        for cluster, config in self._config["verification"]["clusters"].items():
            metrics = calculate_metrics(
                reference=reference_x,
                dict_of_datasets=data_x,
                **config
            )
            metrics = maybe_to_numpy(metrics, use_gpu)
            clusters[cluster] = metrics
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
        





    
        
        