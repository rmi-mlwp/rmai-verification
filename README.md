> :warning: The package is still very much a work in progress.  
> All basic functionality should work technically, but has not undergone rigorous testing.  
> Edge cases or exotic usage might and probably will result in either crashes or incorrect results!

# RMAI-verification (working title)

The RMAI-verification package is an `xarray`-based toolbox setting up a verification pipeline.  
The toolbox is centred around datastores, every user can implement its own datastore to deal with his or her own specific dataformat.
The data in a datastore is accessed by the toolbox through the `.data` method which should return an `xarray` with specified dimenions and coordinates depending on the datastore (Observations vs Forecasts and Gridded data vs Point data).

Guidelines for implementing your own datastore can be found in [GUIDELINES](GUIDELINES.md) document.

For calculating the verification scores, the package relies on the existing packages (e.g. `xskillscore`, `scores`, `verif`).
What score from which package to use is configurable.  

The package provides basic plotting functionality. It can plot an overview of the calculated scores per variable. 
The idea of this package is that (apart from looking at a general overview) in-depth verification is best done by plotting things interactively.

By setting up the pipeline accordingly this can be done easily using the pipeline endproduct `xarray.Dataset` and `xarray`'s native plotting functionality.
## Installation
1. `git clone` this repo
2. `cd` to the cloned folder
3. `pip install .` or `pip install .[cluster]` if you want to submit the verification task to an hpc-cluster.  
   `pip install .[cluster,scores]` if you also want to use the `scores` package (in addition to `xskillscore`) to calculate verification scores
## Usage

### Using the config file 
The most straigthforward way to run a verification pipeline is through the use of a config file

Run using dask on a local cluster
```
rmai-verification local --n_workers 2 --threads_per_worker 1 configs/config.yaml
```
Run using dask on a slurm cluster
```
rmai-verification slurm --queue standard --account my_project --cores 64 --memory 240GB --interface hsn0 ./configs/config.yaml
```

The config is structured as follows:
```yaml
dates:
  ...
datastores:
  ...
transformations:
  ...
verification:
  ...
output:
  ...
visualization:
  ...
```
📅 **dates**  
The `dates` section sets the main forecast run-dates that will verified.  
example:
```yaml
dates:
  start: 2023-01-01 00:00:00
  end: 2023-01-31 12:00:00
  freq: 12h
```
The dates defined here are only used to fill in the placeholders used in the datastore paths (see section below).
Different datastores can have different forecast-range and or timestep. 

💼 **datastores**  
The `datastores` section defines the different datastores that need to be loaded and names them.  
Each datastore needs at least two entries in the config:
- `type`: the type of datastore
- `path`: the path of the datastore, which can contain date placeholders
  
Additionally datastore specific keywords can be provided in the config.  

For `anemoi-inference` datastores, you can optimize data reading performance:
- `chunks`: Custom chunking strategy (dict) for dask arrays. Specify chunks using the **original dimension names** from the NetCDF files (`time`, `values`) before they are renamed to `lead_time` and `grid_index`. Default is `{"reference_time": 1, "time": -1, "values": -1}`.
- `mf_kwargs`: Additional keyword arguments passed to `xarray.open_mfdataset`

example:
```yaml
datastores:
  low_resolution_model:
    type: anemoi-inference
    path: /path/to/my/data/{yyyy}/{mm}/{dd}/forecast_{hh}.nc
    chunks:  # Optional: customize chunking for better performance
      reference_time: 1
      time: -1  # Use -1 for automatic chunking along this dimension
      values: -1  # Adjust based on your grid size and memory constraints
  high_resolution_model:
    type: anemoi-datasets
    path: /path/to/datasets/high_resolution.zarr
  observations:
    type: obsdatastore:
    path: /path/to/my/observations/obs.nc
    observation_type: SYNOP
```

🔄 **transformations**  
The `transformation` section defines the transformations that should be applied to all or some datastores.
Current implemented transformations are:
- `rename`: renaming of the variables
- `uv_to_speed`: calculate the wind speed from the two wind components
- `kelvin_to_celcius`: transform kelvin to degrees celsius (or the inverse)

For each transformation a `datastores` entry can be provided where you can specify to what datastores the transformation should be applied. If the `datastores` entry is abscent, the transformation is applied to all datastores`. 
Other entries are transformation specific.
Transformations are applied in the order they are specified.  
example
```yaml
transformations:
  rename:
    rename_dict:
      2t: [2t_2, T2M ]
      10u: [10u_10, U10M]
      10v: [10v_10, V10M]
      10s: [S10M, 10s_10]
      msl: [msl_0, MLSP]
  uv_to_speed:
    datastores:
      - low_resolution_model
      - high_resolution_model
    u: 10u
    v: 10v
    speed: 10s
  kelvin_to_celcius:
    datastores:
      - low_resolution_model
      - high_resolution_model
    fields: [2t]
```

✔️ **verification**  
The `verification` section takes two main entries:
- `reference_datastore`: The datastore against which the other datastores will by verified
- `variables`: which variables will be verified

For the verfication different `clusters` of verification can be defined.
A cluster is defined by a name and a dictionary specifying:
  - `package`: Which scoring package to use (for now only `xskillscore` is implemented)
  - `metric`: What metrics to calculate (for now only `mse`, `rmse` and `bias` are implemented)
  - `avg_dims`: What dimensions to average over (chosen from the predescribed datastore dimensions)

This allows for different (e.g. spatial vs temporal) verification-types to be run using one pipeline.
example:
```yaml
verification:
  reference_datastore: observations    
  variables : [2t, 10s]
  clusters:
    general:
      package : xskillscore
      metrics:
        - mse
        - bias
        - rmse
      avg_dims: [code, reference_time]
    per_reference_time:
      package: xskillscore
      metrics: [bias]
      avg_dims: [code]
```

💾 **output**  
The `outout` section defines how the output will be saved. If the section is omitted or empty, the output will not be saved. In this section you can select the output-file type either generally for all clusters or (if specified) for clusters seperately. For each cluster the output path sould be specified.  
For each type also the specific kwargs can be provided
example:
```yaml
output:
  type: netcdf
  general:
    path: /path/to/verification/output/general_verification.nc
    engine: h5netcdf
  per_reference_time:
    type: zarr
    path: /path/to/verification/output/per_reference_time
```

📊 **visualization**  
In the `visualization` section, some basic overview plots can be configured. Per `cluster` a directory, filename-prefix and dimension to plot on the x and y axis can be set.
If x and/or y is not set, the defaults taken by `xarray.plot` will be used.  
Per variable a pdf file is created containing plot of all the metrics in the cluster.
```yaml
visualization:
  general:
    directory: /path/to/my/plots
    prefix: general_plot
    x: lead_time
# No plotting for the per_reference_time cluster, since it doesn't make sense at it still has the reference_time dimensions
# Currently no additional averaging is done if there is more then 1 dimension remain after selecting x and y
```
### Using the API
see the example [here](examples/dataset_example.ipynb)
