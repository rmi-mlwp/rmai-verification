"""Profiling utilities for tracking verification performance across different execution modes."""

import time
import json
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Any
from pathlib import Path

LOG = logging.getLogger(__name__)


class Profiler:
    """Simple profiler for tracking execution time of verification stages.
    
    Tracks timing information for different stages of the verification process,
    with special handling for SLURM queuing times.
    
    Attributes:
        mode: Execution mode (local, local_gpu, slurm)
        timings: Dictionary storing timing data for different stages
        _start_times: Temporary storage for stage start times
    """
    
    def __init__(self, mode: str):
        """Initialize the profiler.

        Args:
            mode: Execution mode (local, local_gpu, slurm)
        """
        self.mode = mode
        self.timings: Dict[str, Any] = {
            "mode": mode,
            "stages": {},        # stage_name -> float (no batching) OR list[float] (batching)
            "batches": [],       # list of {"batch": int, "label": Optional[str], "stages": {stage: float}}
            "total_time": 0.0,
        }
        self._start_times: Dict[str, float] = {}
        self._overall_start: Optional[float] = None

        # Batch bookkeeping (None means "no batching")
        self._current_batch_idx: Optional[int] = None
        self._batch_counter: int = 0

    def start(self):
        """Start overall timing."""
        self._overall_start = time.time()
        LOG.info(f"Profiling started for mode: {self.mode}")

    # ---------
    # BATCHING
    # ---------
    def start_batch(self, label: Optional[str] = None) -> int:
        """Start a new batch.

        Call this at the start of each processing batch. Stage timings will be
        stored under this batch, and aggregated per stage across batches.

        Args:
            label: Optional label for the batch (e.g., a datetime window)

        Returns:
            The batch index (0-based)
        """
        batch_idx = self._batch_counter
        self._batch_counter += 1
        self._current_batch_idx = batch_idx

        self.timings["batches"].append(
            {"batch": batch_idx, "label": label, "stages": {}}
        )
        LOG.debug(f"Batch {batch_idx} started (label={label!r})")
        return batch_idx

    def end_batch(self):
        """End the current batch."""
        if self._current_batch_idx is None:
            LOG.warning("end_batch() called but no batch was started")
            return
        LOG.debug(f"Batch {self._current_batch_idx} ended")
        self._current_batch_idx = None

    # -------------
    # STAGE TIMINGS
    # -------------  
    def start_stage(self, stage_name: str):
        """Start timing a verification stage.

        Args:
            stage_name: Name of the stage (e.g., 'align_datastores', 'calculate_clusters')
        """
        self._start_times[stage_name] = time.time()
        LOG.debug(f"Stage '{stage_name}' started")

    def _record_stage_elapsed(self, stage_name: str, elapsed: float):
        """Record stage elapsed time, with or without batching."""
        if self._current_batch_idx is None:
            # Backwards-compatible behavior: single value per stage
            self.timings["stages"][stage_name] = elapsed
            return

        # Batch mode: store per batch + accumulate per stage as list of runs
        batch_rec = self.timings["batches"][self._current_batch_idx]
        batch_rec["stages"][stage_name] = elapsed

        existing = self.timings["stages"].get(stage_name)
        if existing is None or not isinstance(existing, list):
            self.timings["stages"][stage_name] = [elapsed]
        else:
            existing.append(elapsed)
        
    def end_stage(self, stage_name: str):
        """End timing a verification stage.

        Args:
            stage_name: Name of the stage to end
        """
        if stage_name not in self._start_times:
            LOG.warning(f"Stage '{stage_name}' was never started")
            return

        elapsed = time.time() - self._start_times[stage_name]
        self._record_stage_elapsed(stage_name, elapsed)

        LOG.info(f"Stage '{stage_name}' completed in {elapsed:.2f} seconds")
        del self._start_times[stage_name]
        
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for timing a stage.

        Example:
            with profiler.stage("align_datastores"):
                align_datastores()
        """
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage(stage_name)

    def record_slurm_queuing(self, queuing_time: float):
        """Record SLURM queuing time.

        Args:
            queuing_time: Time spent waiting for workers (seconds)
        """
        self.timings["slurm_queuing_time"] = queuing_time
        LOG.info(f"SLURM queuing time: {queuing_time:.2f} seconds")

    def record_cluster_startup(self, startup_time: float):
        """Record cluster startup time.

        Args:
            startup_time: Time to initialize cluster (seconds)
        """
        self.timings["cluster_startup_time"] = startup_time
        LOG.info(f"Cluster startup time: {startup_time:.2f} seconds")

    def end(self):
        """End overall timing and calculate total time."""
        if self._overall_start is None:
            LOG.warning("Profiler was never started")
            return

        self.timings["total_time"] = time.time() - self._overall_start
        LOG.info(f"Total execution time: {self.timings['total_time']:.2f} seconds")

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        return self.timings

    def _stage_stats(self, values: list[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def print_summary(self):
        """Print a formatted summary of profiling results."""
        print("\n" + "=" * 60)
        print(f"PROFILING SUMMARY - Mode: {self.mode}")
        print("=" * 60)

        # Print cluster startup info
        if "cluster_startup_time" in self.timings:
            print(f"\nCluster Startup: {self.timings['cluster_startup_time']:.2f}s")

        # Print SLURM queuing time if applicable
        if "slurm_queuing_time" in self.timings:
            print(f"SLURM Queuing:   {self.timings['slurm_queuing_time']:.2f}s")

        # Print stage timings
        stages = self.timings.get("stages", {})
        batches = self.timings.get("batches", [])

        if stages:
            print("\nVerification Stages:")

            # Batch mode: mean/min/max across runs
            if batches:
                # Sort for stable output
                for stage in sorted(stages.keys()):
                    vals = stages[stage]
                    if not isinstance(vals, list):
                        vals = [vals]
                    stats = self._stage_stats(vals)
                    print(
                        f"  {stage:24s} "
                        f"mean={stats['mean']:8.2f}s "
                        f"min={stats['min']:8.2f}s "
                        f"max={stats['max']:8.2f}s "
                        f"(n={len(vals)})"
                    )
            else:
                # Backwards-compatible: single timing per stage
                for stage, duration in stages.items():
                    if isinstance(duration, list):
                        # Shouldn't happen without batches, but keep it robust
                        stats = self._stage_stats(duration)
                        print(
                            f"  {stage:24s} "
                            f"mean={stats['mean']:8.2f}s "
                            f"min={stats['min']:8.2f}s "
                            f"max={stats['max']:8.2f}s "
                            f"(n={len(duration)})"
                        )
                    else:
                        print(f"  {stage:20s} {duration:8.2f}s")

        # Optional: show per-batch stage durations (compact)
        if batches:
            print("\nPer-batch timings:")
            for b in batches:
                label = b.get("label")
                header = f"  Batch {b['batch']}"
                if label:
                    header += f" ({label})"
                print(header)

                batch_total = 0.0
                for stage, dur in sorted(b["stages"].items()):
                    print(f"    {stage:24s} {dur:8.2f}s")
                    batch_total += dur

                # Also print total per-batch
                print(f"    {'TOTAL':24s} {batch_total:8.2f}s")


        # Print total
        print(f"\nTotal Execution: {self.timings['total_time']:.2f}s")
        print("=" * 60 + "\n")

    def save_to_file(self, filepath: Optional[str] = None):
        """Save profiling results to a JSON file.

        Args:
            filepath: Path to output file. If None, uses default name based on mode and timestamp.
        """
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"profiling_{self.mode}_{timestamp}.json"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.timings, f, indent=2)

        LOG.info(f"Profiling results saved to {filepath}")


# Global profiler instance (optional, can be passed explicitly)
_current_profiler: Optional[Profiler] = None


def get_profiler() -> Optional[Profiler]:
    """Get the current global profiler instance."""
    return _current_profiler


def set_profiler(profiler: Optional[Profiler]):
    """Set the global profiler instance."""
    global _current_profiler
    _current_profiler = profiler


def wait_for_workers(client, n_workers: int, timeout: int = 300) -> float:
    """Wait for SLURM workers to become available and measure queuing time."""
    start_time = time.time()
    LOG.info(f"Waiting for {n_workers} workers to become available...")

    while True:
        elapsed = time.time() - start_time

        if elapsed > timeout:
            raise TimeoutError(
                f"Timed out waiting for workers after {timeout}s. "
                f"Only {len(client.scheduler_info()['workers'])} workers available."
            )

        n_available = len(client.scheduler_info()["workers"])

        if n_available >= n_workers:
            queuing_time = time.time() - start_time
            LOG.info(f"All {n_workers} workers ready after {queuing_time:.2f}s")
            return queuing_time

        if int(elapsed) % 10 == 0 and elapsed > 0:  # Log every 10 seconds
            LOG.info(
                f"Waiting for workers... ({n_available}/{n_workers} available, {elapsed:.0f}s elapsed)"
            )

        time.sleep(1)
