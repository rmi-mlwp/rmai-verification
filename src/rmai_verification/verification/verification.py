
import logging
import xarray as xr
import os
from typing import Dict, List, Union
import numpy as np


from ..alignment import align_reference_times, align_spatial, align_valid_times
from ..transformations import apply_transformations
from ..utils.sanitation import broadcast_nans, prep_config
from ..utils.time import to_timedelta64, datetime_batches
from ..metrics import calculate_metrics, clear_gpu_memory
from ..output import save_dataset
from ..visualization import plot_overview
from ..utils.files import load_yaml
from ..datastores import select_variables, load_datastores, BaseDataStore
from ..utils.profiling import get_profiler

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

    def align_datastores(self) -> None:
        """Align all datastores in time and space.
        This method selects the specified variables from the datastores,
        aligns the reference times, and aligns the spatial dimensions.
        It also handles the selection of variables based on the configuration.
        The reference datastore is used as the baseline for spatial alignment.
        """
        profiler = get_profiler()
        if profiler:
            profiler.start_stage("align_datastores")

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
        
        if profiler:
            profiler.end_stage("align_datastores")
        
    def calculate_clusters(self) -> None:
        """Calculate verification metrics for defined clusters.
        
        This method iterates through the clusters defined in the configuration,
        calculates the metrics using the reference datastore, and saves the results.
        It also handles the output configuration for each cluster.
        """
        profiler = get_profiler()
        if profiler:
            profiler.start_stage("calculate_clusters")

        use_gpu = bool(self._config.get("verification", {}).get("use_gpu", False))
        LOG.info(f"Using GPU for metric calculations: {use_gpu}")

        reference = self._aligned_data.pop(self._reference_datastore)
        broadcast_nans(list(self._aligned_data.values()))

        reference_x = maybe_to_cupy(reference, use_gpu)
        data_x = {k: maybe_to_cupy(v, use_gpu) for k, v in self._aligned_data.items()}
        
        clusters = dict()
        large_memory = bool(self._config.get("verification", {}).get("large_memory", False))

        # Batch info (set by your outer batch loop)
        batch_idx = getattr(self, "_batch_idx", None)

        for cluster, config in self._config["verification"]["clusters"].items():
            metrics = calculate_metrics(
                reference=reference_x,
                dict_of_datasets=data_x,
                large_memory=large_memory,
                **config
            )
            metrics = maybe_to_numpy(metrics, use_gpu)
            # Clear GPU memory after processing each cluster
            if use_gpu:
                clear_gpu_memory()

            # Set attributes for the metrics dataset
            metrics.attrs = {}

            clusters[cluster] = metrics
            output_config = prep_config(self._config["output"], cluster)
            output_type = output_config.pop("type",None)

            LOG.info("Attributes of metrics for cluster %s: %s", cluster, clusters[cluster].attrs)

            if output_type:
                output_path = output_config.pop("path", f"{cluster}.{output_type}")

                # --- minimal batching hook for zarr ---
                if output_type == "zarr" and batch_idx is not None:
                    ds = clusters[cluster]
                    if "reference_time" not in ds.dims:
                        raise ValueError("Batching with zarr output requires a 'reference_time' dimension in the metrics dataset.")

                    # "w" only when genuinely starting from scratch. When
                    # resuming an interrupted run the store already holds the
                    # earlier batches, so the first batch we process must append.
                    resume_from = getattr(self, "_resume_from", 0)
                    mode = "w" if (batch_idx == 0 and resume_from == 0) else "a"

                    save_dataset(
                        dataset=ds,
                        type=output_type,
                        path=output_path,
                        append_dim="reference_time",
                        mode=mode,
                        **output_config,
                    )
                else:
                    # original behavior
                    save_dataset(
                        dataset=clusters[cluster],
                        type=output_type,
                        path=output_path,
                        **output_config,
                    )
        self._clusters = clusters
        
        if profiler:
            profiler.end_stage("calculate_clusters")

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
        profiler = get_profiler()
        if profiler:
            profiler.start_stage("visualize_clusters")

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
            
        if profiler:
            profiler.end_stage("visualize_clusters")
    
    def _batch_reference_times(self, batch):
        """The reference times a batch covers, as datetime64[ns]."""
        freq = to_timedelta64(self._config["dates"]["frequency"])
        start = np.datetime64(batch[0])
        end = np.datetime64(batch[1])
        return np.arange(start, end + freq, freq).astype("datetime64[ns]")

    def _zarr_output_paths(self):
        """Output zarr path per cluster, for clusters that write zarr."""
        paths = []
        for cluster in self._config["verification"]["clusters"]:
            output_config = prep_config(self._config["output"], cluster)
            if output_config.get("type") != "zarr":
                continue
            paths.append(output_config.get("path", f"{cluster}.zarr"))
        return paths

    def _resume_start_batch(self, batches):
        """
        How many leading batches are already complete in the output store(s).

        A node failure or walltime kill leaves a store holding whole batches
        (each batch is appended in a single to_zarr call). Counting them lets a
        restart continue instead of recomputing from batch 0.

        Conservative on purpose: the stored reference_time values must match the
        expected ones exactly, and must end on a batch boundary. Anything else --
        a store from a different config, or a write torn mid-append -- returns 0,
        so the run starts over rather than appending onto something inconsistent.
        """
        paths = self._zarr_output_paths()
        if not paths:
            return 0

        expected = [self._batch_reference_times(b) for b in batches]
        done = None
        for path in paths:
            if not os.path.exists(path):
                return 0
            try:
                stored = xr.open_zarr(path).reference_time.values.astype("datetime64[ns]")
            except Exception as err:
                LOG.warning("Could not read %s to resume (%s); starting from batch 0", path, err)
                return 0

            n = 0
            pos = 0
            for times in expected:
                end = pos + len(times)
                if end > len(stored):
                    break
                if not np.array_equal(stored[pos:end], times):
                    LOG.warning(
                        "%s diverges from the configured dates at batch %d; starting from batch 0",
                        path, n,
                    )
                    return 0
                pos, n = end, n + 1
            if pos != len(stored):
                LOG.warning(
                    "%s holds %d reference times, which is not a whole number of "
                    "batches (%d complete); starting from batch 0 rather than "
                    "appending onto a torn write.",
                    path, len(stored), n,
                )
                return 0
            done = n if done is None else min(done, n)

        return done or 0

    def verify(self):
        config = self._config

        # Introduce batch size.
        # First convert start, end and frequency to numpy datetime64 and timedelta64 for easier manipulation
        start = config["dates"]["start"]
        end = config["dates"]["end"]
        frequency = config["dates"]["frequency"]
        batch_size = config["dates"].get("batch_size", -1)
        batches = list(datetime_batches(start, end, frequency, batch_size))
        LOG.info("Batches: %d", len(batches))

        self._frequency = config["dates"]["frequency"]
        self._output_type = config["output"].get("type",None)

        self._resume_from = self._resume_start_batch(batches)
        if self._resume_from:
            LOG.info(
                "Resuming: %d of %d batches already in the output store, "
                "continuing at batch %d",
                self._resume_from, len(batches), self._resume_from,
            )

        profiler = get_profiler()

        for i, batch in enumerate(batches):
            if i < self._resume_from:
                continue
            batch_label = f"batch_{i:03d}"
            profiler.start_batch(label=batch_label)
            LOG.info("Processing batch: %s", batch)

            self._batch_idx = i
            self._start, self._end = batch[0], batch[1]

            if profiler:
                profiler.start_stage("load_datastores")
            
            self._datastores = load_datastores(
                datastores=self._config["datastores"],
                start_date=self._start,
                end_date=self._end,
                frequency=self._frequency
            )
            self._reference_datastore = config["verification"]["reference_datastore"]
            
            if profiler:
                profiler.end_stage("load_datastores")
            
            # Profile transformations
            if profiler:
                profiler.start_stage("apply_transformations")

            apply_transformations(
                datastores=self._datastores, 
                transformations=self._config["transformations"]
            )
            
            if profiler:
                profiler.end_stage("apply_transformations")

            self.align_datastores()
            self.calculate_clusters()
            # self.visualize_clusters()

            profiler.end_batch()
