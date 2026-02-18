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
            "stages": {},
            "total_time": 0.0,
        }
        self._start_times: Dict[str, float] = {}
        self._overall_start: Optional[float] = None
        
    def start(self):
        """Start overall timing."""
        self._overall_start = time.time()
        LOG.info(f"Profiling started for mode: {self.mode}")
        
    def start_stage(self, stage_name: str):
        """Start timing a verification stage.
        
        Args:
            stage_name: Name of the stage (e.g., 'align_datastores', 'calculate_clusters')
        """
        self._start_times[stage_name] = time.time()
        LOG.debug(f"Stage '{stage_name}' started")
        
    def end_stage(self, stage_name: str):
        """End timing a verification stage.
        
        Args:
            stage_name: Name of the stage to end
        """
        if stage_name not in self._start_times:
            LOG.warning(f"Stage '{stage_name}' was never started")
            return
            
        elapsed = time.time() - self._start_times[stage_name]
        self.timings["stages"][stage_name] = elapsed
        LOG.info(f"Stage '{stage_name}' completed in {elapsed:.2f} seconds")
        del self._start_times[stage_name]
        
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for timing a stage.
        
        Args:
            stage_name: Name of the stage
            
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
        """Get profiling summary.
        
        Returns:
            Dictionary containing all timing information
        """
        return self.timings
        
    def print_summary(self):
        """Print a formatted summary of profiling results."""
        print("\n" + "="*60)
        print(f"PROFILING SUMMARY - Mode: {self.mode}")
        print("="*60)
        
        # Print cluster startup info
        if "cluster_startup_time" in self.timings:
            print(f"\nCluster Startup: {self.timings['cluster_startup_time']:.2f}s")
            
        # Print SLURM queuing time if applicable
        if "slurm_queuing_time" in self.timings:
            print(f"SLURM Queuing:   {self.timings['slurm_queuing_time']:.2f}s")
            
        # Print stage timings
        if self.timings["stages"]:
            print("\nVerification Stages:")
            for stage, duration in self.timings["stages"].items():
                print(f"  {stage:20s} {duration:8.2f}s")
                
        # Print total
        print(f"\nTotal Execution: {self.timings['total_time']:.2f}s")
        print("="*60 + "\n")
        
    def save_to_file(self, filepath: Optional[str] = None):
        """Save profiling results to a JSON file.
        
        Args:
            filepath: Path to output file. If None, uses default name based on mode and timestamp.
        """
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"profiling_{self.mode}_{timestamp}.json"
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
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
    """Wait for SLURM workers to become available and measure queuing time.
    
    Args:
        client: Dask client
        n_workers: Expected number of workers
        timeout: Maximum time to wait in seconds
        
    Returns:
        Time spent waiting for workers (seconds)
        
    Raises:
        TimeoutError: If workers don't appear within timeout
    """
    start_time = time.time()
    LOG.info(f"Waiting for {n_workers} workers to become available...")
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            raise TimeoutError(
                f"Timed out waiting for workers after {timeout}s. "
                f"Only {len(client.scheduler_info()['workers'])} workers available."
            )
            
        n_available = len(client.scheduler_info()['workers'])
        
        if n_available >= n_workers:
            queuing_time = time.time() - start_time
            LOG.info(f"All {n_workers} workers ready after {queuing_time:.2f}s")
            return queuing_time
            
        if int(elapsed) % 10 == 0 and elapsed > 0:  # Log every 10 seconds
            LOG.info(f"Waiting for workers... ({n_available}/{n_workers} available, {elapsed:.0f}s elapsed)")
            
        time.sleep(1)
