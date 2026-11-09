
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

# Cluster keys read by Verification itself, not passed on to calculate_metrics.
CLUSTER_OPTIONS = ("accumulate", "checkpoint_every")


def _close_cached_files() -> int:
    """Close every file in xarray's file cache (shrink it to 0 and back), then
    collect garbage and return freed heap to the OS. Returns the number of
    files that were open."""
    import ctypes
    import gc
    from xarray.backends.file_manager import FILE_CACHE
    n, size = len(FILE_CACHE), FILE_CACHE.maxsize
    FILE_CACHE.maxsize = 0
    FILE_CACHE.maxsize = size
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    return n


def _accumulates(cluster_config: Dict) -> bool:
    """Whether a cluster is summed across batches instead of written per batch."""
    return bool(cluster_config.get("accumulate", False))

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

        All clusters active in this batch are built lazily and computed in ONE
        ``dask.compute``, so input chunks shared between clusters (e.g. the
        per-init contingency tables and the per-grid-point maps) are read once.

        Ordinary clusters are written per batch (appended along
        reference_time). Clusters with ``accumulate: true`` are summed into an
        int32 running total across batches and checkpointed to their zarr
        path (see ``_accumulate``).
        """
        import dask

        profiler = get_profiler()
        if profiler:
            profiler.start_stage("calculate_clusters")

        use_gpu = bool(self._config.get("verification", {}).get("use_gpu", False))
        LOG.info(f"Using GPU for metric calculations: {use_gpu}")

        reference = self._aligned_data.pop(self._reference_datastore)
        broadcast_nans(list(self._aligned_data.values()))

        reference_x = maybe_to_cupy(reference, use_gpu)
        data_x = {k: maybe_to_cupy(v, use_gpu) for k, v in self._aligned_data.items()}

        large_memory = bool(self._config.get("verification", {}).get("large_memory", False))

        # Batch info (set by your outer batch loop)
        batch_idx = getattr(self, "_batch_idx", None)
        active = getattr(self, "_active_clusters", None)

        lazy = dict()
        for cluster, config in self._config["verification"]["clusters"].items():
            if active is not None and cluster not in active:
                LOG.info("Cluster %s already complete for batch %s, skipping", cluster, batch_idx)
                continue
            metric_config = {k: v for k, v in config.items() if k not in CLUSTER_OPTIONS}
            lazy[cluster] = calculate_metrics(
                reference=reference_x,
                dict_of_datasets=data_x,
                large_memory=large_memory,
                compute=False,
                **metric_config
            )
        computed = dict(zip(lazy, dask.compute(*lazy.values()))) if lazy else {}

        clusters = dict()
        for cluster, metrics in computed.items():
            metrics = maybe_to_numpy(metrics, use_gpu)
            # Clear GPU memory after processing each cluster
            if use_gpu:
                clear_gpu_memory()

            # Set attributes for the metrics dataset
            metrics.attrs = {}
            clusters[cluster] = metrics

            if _accumulates(self._config["verification"]["clusters"][cluster]):
                continue  # after the ordinary outputs, see below

            output_config = self._output_config(cluster)
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

        # Accumulating clusters strictly AFTER the per-batch outputs: a kill in
        # between leaves the running total one batch behind the tables, never
        # ahead, and the resume logic recomputes that batch for the maps only.
        for cluster in clusters:
            if _accumulates(self._config["verification"]["clusters"][cluster]):
                self._accumulate(cluster, clusters[cluster])
        self._clusters = clusters

        if profiler:
            profiler.end_stage("calculate_clusters")

    def _output_config(self, cluster: str) -> Dict:
        """The output settings of one cluster: the shared keys (e.g. type)
        overridden by the cluster's own block. Other clusters' blocks are
        dropped; with several clusters they would otherwise be passed on to
        the saver as keyword arguments (e.g. to_zarr(categorical_map=...))."""
        others = set(self._config["verification"]["clusters"]) - {cluster}
        return {k: v for k, v in prep_config(self._config["output"], cluster).items()
                if k not in others}

    # --- accumulating clusters (e.g. per-grid-point contingency maps) --------

    def _accumulate(self, cluster: str, ds: xr.Dataset) -> None:
        """Add one batch to the cluster's int32 running total; checkpoint it.

        The total is written to the cluster's zarr path every
        ``checkpoint_every`` batches (cluster option, default 6) and after the
        last batch, with attrs n_batches / n_inits / first_reference_time /
        last_reference_time saying exactly which inits it contains.
        """
        cfg = self._config["verification"]["clusters"][cluster]
        ds = ds.astype(np.int32)
        if cluster in self._accum:
            total = self._accum[cluster]
            for v in ds.data_vars:
                total[v].values += ds[v].values  # in place; layouts are identical
        else:
            self._accum[cluster] = total = ds
        batch = self._batches[self._batch_idx]
        info = self._accum_info.setdefault(cluster, {"n_batches": 0, "n_inits": 0})
        times = self._batch_reference_times(batch)
        if info["n_batches"] == 0:
            info["first_reference_time"] = str(times[0])
        info["n_batches"] += 1
        info["n_inits"] += int(times.size)
        info["last_reference_time"] = str(times[-1])

        every = int(cfg.get("checkpoint_every", 6))
        last = self._batch_idx == len(self._batches) - 1
        if last or info["n_batches"] % every == 0:
            self._write_checkpoint(cluster)

    def _release_worker_memory(self) -> None:
        """Close the netCDF/HDF5 files cached by xarray, in every dask worker
        and in the client, after each batch.

        xarray keeps up to 128 files open per process (FILE_CACHE), each with
        its HDF5 chunk cache full of decompressed chunks. A CF batch opens 72+
        files, so this memory -- invisible to dask ("unmanaged") -- grew from
        batch to batch: ~5 GiB per worker after ~5 batches, and the 32 GB jobs
        were OOM-killed after 16-25 batches (ETS campaign, 2026-11-06).
        Restarting the workers instead took > 120 s per restart (imports from
        Lustre). Files are reopened on demand, so this changes no result.
        """
        try:
            from distributed import get_client
            client = get_client()
        except (ImportError, ValueError):
            client = None  # no distributed client (e.g. GPU mode)
        n = {"client": _close_cached_files()}
        if client is not None:
            n.update(client.run(_close_cached_files))
        LOG.info("Closed cached files after batch %s: %s", self._batch_idx, sorted(n.values()))

    def _accum_path(self, cluster: str) -> str:
        output_config = self._output_config(cluster)
        if output_config.get("type") != "zarr":
            raise ValueError(f"Accumulating cluster {cluster} needs zarr output")
        return output_config.get("path", f"{cluster}.zarr")

    def _write_checkpoint(self, cluster: str) -> None:
        """Write the running total to <path>.tmp, then swap it into <path>.

        A kill during the write leaves the previous checkpoint intact; a kill
        between removing <path> and the rename leaves a complete <path>.tmp,
        which _load_checkpoint picks up.
        """
        import shutil
        path = self._accum_path(cluster)
        tmp = path.rstrip("/") + ".tmp"
        ds = self._accum[cluster].copy()
        ds.attrs = dict(self._accum_info[cluster], complete="yes")
        for v in ds.data_vars:
            ds[v].encoding = {}
            chunks = tuple(1 if d == "lead_time" else n for d, n in ds[v].sizes.items())
            ds[v].encoding["chunks"] = chunks
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        save_dataset(dataset=ds, type="zarr", path=tmp, mode="w")
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(tmp, path)
        LOG.info("Checkpointed %s: %d batches, %d inits up to %s -> %s", cluster,
                 ds.attrs["n_batches"], ds.attrs["n_inits"], ds.attrs["last_reference_time"], path)

    def _load_checkpoint(self, cluster: str, batches, max_batches: int) -> int:
        """Restore a cluster's running total; returns how many batches it holds.

        Accepted only if its attrs match the first n configured batches exactly
        and n <= max_batches (the batches already in the per-batch outputs).
        Anything else returns 0: the total is then rebuilt from batch 0, and
        batches whose per-batch outputs exist are recomputed for it only.
        """
        import shutil
        path = self._accum_path(cluster)
        tmp = path.rstrip("/") + ".tmp"
        if os.path.exists(path) and os.path.exists(tmp):
            shutil.rmtree(tmp)          # torn write; the old checkpoint stands
        elif not os.path.exists(path) and os.path.exists(tmp):
            path = tmp                  # killed between rmtree and rename
        if not os.path.exists(path):
            return 0
        try:
            ds = xr.open_zarr(path).load()
        except Exception as err:
            LOG.warning("Could not read checkpoint %s (%s); rebuilding it", path, err)
            return 0
        a = ds.attrs
        n = int(a.get("n_batches", 0))
        ok = a.get("complete") == "yes" and 0 < n <= min(max_batches, len(batches))
        if ok:
            times = [self._batch_reference_times(b) for b in batches[:n]]
            ok = (a.get("first_reference_time") == str(times[0][0])
                  and a.get("last_reference_time") == str(times[-1][-1])
                  and int(a.get("n_inits", -1)) == sum(t.size for t in times))
        if not ok:
            LOG.warning("Checkpoint %s (n_batches=%s) does not match this run; rebuilding it", path, n)
            return 0
        for v in ds.data_vars:
            ds[v] = ds[v].astype(np.int32)
        ds.attrs = {}
        self._accum[cluster] = ds
        self._accum_info[cluster] = {k: a[k] for k in
                                     ("n_batches", "n_inits", "first_reference_time", "last_reference_time")}
        self._accum_info[cluster]["n_batches"] = n
        self._accum_info[cluster]["n_inits"] = int(a["n_inits"])
        LOG.info("Resuming %s from checkpoint %s: %d batches", cluster, path, n)
        return n

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
        for cluster, cfg in self._config["verification"]["clusters"].items():
            if _accumulates(cfg):
                continue  # no reference_time axis; resumed via its checkpoint
            output_config = self._output_config(cluster)
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

        self._batches = batches
        clusters_cfg = self._config["verification"]["clusters"]
        ordinary = [c for c, cfg in clusters_cfg.items() if not _accumulates(cfg)]
        self._resume_from = self._resume_start_batch(batches) if ordinary else len(batches)
        if self._resume_from:
            LOG.info(
                "Resuming: %d of %d batches already in the output store, "
                "continuing at batch %d",
                self._resume_from, len(batches), self._resume_from,
            )
        # Accumulating clusters resume from their own checkpoints, which may be
        # behind the per-batch outputs; those batches are redone for them only.
        self._accum, self._accum_info = {}, {}
        accum_from = {c: self._load_checkpoint(c, batches, self._resume_from)
                      for c, cfg in clusters_cfg.items() if _accumulates(cfg)}
        first = min([self._resume_from if ordinary else len(batches), *accum_from.values()])

        profiler = get_profiler()

        for i, batch in enumerate(batches):
            if i < first:
                continue
            self._active_clusters = [c for c in ordinary if i >= self._resume_from] + \
                                    [c for c, n in accum_from.items() if i >= n]
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
            self._release_worker_memory()
            # self.visualize_clusters()

            profiler.end_batch()
