# Profiling for RMAI Verification

This directory contains tools and examples for profiling the verification system across different execution modes.

## Overview

The profiling system tracks execution time across all stages of the verification pipeline:

```
┌─────────────────────────────────────────────────────────┐
│ INITIALIZATION PHASE                                    │
├─────────────────────────────────────────────────────────┤
│ 1. load_datastores      ← NEW! Track data loading     │
│ 2. apply_transformations ← NEW! Track transformations  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ VERIFICATION PHASE                                      │
├─────────────────────────────────────────────────────────┤
│ 3. align_datastores     ← Temporal/spatial alignment   │
│ 4. calculate_clusters   ← Compute metrics              │
│ 5. visualize_clusters   ← Generate plots               │
└─────────────────────────────────────────────────────────┘
```

### Tracked Metrics

The profiling system tracks:
- **Cluster startup time**: Time to initialize the Dask cluster
- **SLURM queuing time**: Time waiting for SLURM workers to become available (SLURM mode only)
- **Verification stages**: Time spent in each stage:
  - `load_datastores`: Loading data from various sources
  - `apply_transformations`: Applying data transformations (e.g., unit conversions)
  - `align_datastores`: Aligning data in time and space
  - `calculate_clusters`: Computing verification metrics
  - `visualize_clusters`: Generating plots and visualizations
- **Total execution time**: Overall runtime from start to finish

## Execution Modes

The profiling system now tracks all major stages of the verification pipeline:

1. **Initialization Phase**:
   - `load_datastores`: Loading data from file systems, remote sources, or zarr stores
   - `apply_transformations`: Unit conversions, variable renaming, derived variables

2. **Verification Phase** (tracked in `verify()` method):
   - `align_datastores`: Temporal and spatial alignment of datasets
   - `calculate_clusters`: Metric computation (RMSE, bias, etc.)
   - `visualize_clusters`: Plot generation

3. **Overhead**:
   - Cluster startup time (local/SLURM modes)
   - SLURM queuing time (SLURM mode only)

### 1. Local Mode
Uses a local Dask cluster with multiple workers on a single machine.

```bash
rmai-verification local config.yaml \
    --n_workers 4 \
    --threads_per_worker 1 \
    --profile_output profiling_local.json
```

### 2. Local GPU Mode
Single-process execution using GPU acceleration (CuPy).

```bash
rmai-verification local config.yaml \
    --use_gpu \
    --profile_output profiling_local_gpu.json
```

### 3. SLURM Mode
Distributed execution on a SLURM cluster with multiple nodes.

```bash
rmai-verification slurm config.yaml \
    --queue standard \
    --account your_account \
    --jobs 3 \
    --cores 8 \
    --processes 8 \
    --memory 64GB \
    --walltime 01:00:00 \
    --worker_timeout 600 \
    --profile_output profiling_slurm.json
```

## Running Profiling Experiments

### Quick Start

1. Edit `run_profiling_experiments.sh` to set your configuration file and SLURM parameters:
   ```bash
   CONFIG_FILE="path/to/your/config.yaml"
   ```

2. Make the script executable:
   ```bash
   chmod +x run_profiling_experiments.sh
   ```

3. Run the experiments:
   ```bash
   ./run_profiling_experiments.sh
   ```

### Comparing Results

After running experiments, compare the profiling results:

```bash
python compare_profiles.py profiling_results/profile_*.json
```

This will generate a comparison table showing:
- Time breakdown by stage
- Overhead (cluster startup, queuing)
- Speedup relative to the baseline mode

## Output Format

Profiling results are saved as JSON files with the following structure:

```json
{
  "mode": "slurm",
  "cluster_startup_time": 5.23,
  "slurm_queuing_time": 45.67,
  "stages": {
    "load_datastores": 85.34,
    "apply_transformations": 12.56,
    "align_datastores": 120.45,
    "calculate_clusters": 450.12,
    "visualize_clusters": 30.78
  },
  "total_time": 749.15
}
```

## SLURM-Specific Considerations

### Queuing Time
The profiler measures the time between submitting jobs and all workers becoming available. This includes:
- Time in the SLURM queue
- Worker startup time
- Network initialization

Configure the timeout with `--worker_timeout` (default: 300 seconds).

### Optimal Worker Configuration
The number of expected workers is calculated as:
```
expected_workers = jobs × processes
```

Make sure your SLURM configuration provides enough resources for all workers.

## Tips for Profiling

1. **Run multiple iterations**: Performance can vary between runs. Consider averaging results from multiple executions.

2. **Control for system load**: Try to run experiments when cluster load is similar to get comparable results.

3. **Adjust worker counts**: Test different configurations (e.g., 2, 4, 8 workers) to find optimal settings.

4. **Monitor resource usage**: Use tools like `dask.distributed.performance_report()` for more detailed profiling.

5. **SLURM queue depth**: Queuing times vary significantly with cluster load. Note the queue depth when running experiments.

6. **Identify bottlenecks**: The stage breakdown helps identify where to focus optimization efforts:
   - High `load_datastores` time → consider data caching or faster storage
   - High `align_datastores` time → may benefit from spatial chunking optimization
   - High `calculate_clusters` time → main computation target for parallelization

## Example Results

A typical comparison might show:

```
PROFILING COMPARISON
================================================================================
Metric                         local  local_gpu     slurm
--------------------------------------------------------------------------------
Cluster Startup                 2.3s      0.1s      5.2s
SLURM Queuing                    N/A       N/A     45.7s

Verification Stages:
  Load Datastores               89.2s     87.5s     85.3s
  Apply Transformations         13.1s     12.8s     12.6s
  Align Datastores             125.4s    118.2s    120.5s
  Calculate Clusters           512.3s    245.1s    450.1s
  Visualize Clusters            34.5s     32.1s     30.8s

TOTAL TIME                     776.8s    498.6s    750.2s

Speedup vs local                1.00x     1.56x     1.04x
================================================================================
```

This shows:
- GPU mode is 1.56x faster due to accelerated computations
- SLURM mode has significant queuing overhead but can be faster for larger datasets
- Data loading (`load_datastores`) is consistent across modes (~85-89s)
- Most time is spent in calculation stage (optimization target)
- Transformations are relatively cheap (~13s across all modes)

## Advanced Usage

### Custom Profiling in Code

You can access the profiler directly in your code:

```python
from rmai_verification.utils.profiling import Profiler, set_profiler

# Create and configure profiler
profiler = Profiler(mode="custom")
set_profiler(profiler)
profiler.start()

# Profile a custom stage
with profiler.stage("custom_operation"):
    # Your code here
    pass

profiler.end()
profiler.print_summary()
```

### Programmatic Access

Load and analyze profiling data programmatically:

```python
import json

with open('profiling_local.json', 'r') as f:
    data = json.load(f)

# Access specific metrics
total_time = data['total_time']
compute_time = sum(data['stages'].values())
overhead_pct = (total_time - compute_time) / total_time * 100

print(f"Overhead: {overhead_pct:.1f}%")

# Identify the slowest stage
stages = data['stages']
slowest_stage = max(stages.items(), key=lambda x: x[1])
print(f"Slowest stage: {slowest_stage[0]} ({slowest_stage[1]:.2f}s)")

# Calculate I/O vs computation ratio
io_time = stages.get('load_datastores', 0) + stages.get('apply_transformations', 0)
compute_time = stages.get('calculate_clusters', 0)
io_ratio = io_time / compute_time if compute_time > 0 else 0
print(f"I/O to computation ratio: {io_ratio:.2f}")

# If I/O ratio is high (>0.3), consider:
# - Using faster storage (SSD, parallel file systems)
# - Pre-loading or caching datasets
# - Optimizing chunk sizes in zarr stores
```
