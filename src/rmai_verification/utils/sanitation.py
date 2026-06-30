import numpy as np
import xarray as xr
import itertools
from typing import List, Dict


def broadcast_nans(list_of_datasets : List[xr.Dataset]) -> None:
    """
    Broadcasts NaN values across a list of xarray Datasets by ensuring that if a value is NaN
    in one dataset at a specific coordinate, it becomes NaN in all datasets at that coordinate.
    This function modifies the input datasets in-place for shared variables at common coordinates.
    
    Parameters
    ----------
    list_of_datasets : List[xr.Dataset]
        A list of xarray Datasets to process. The datasets should share some common
        coordinates and variables.
   
    Returns
    -------
    None
        The function modifies the input datasets in-place.
    
    Notes
    -----
    - The function operates on pairs of datasets, comparing each dataset with every other dataset
      in the list.
    - Only coordinate values that exist in both datasets of a pair are considered.
    - Only variables that exist in both datasets of a pair are processed.
    - The NaN broadcasting is performed at the intersection of coordinates between each pair
      of datasets.
    
    Examples
    --------
    >>> ds1 = xr.Dataset(...)
    >>> ds2 = xr.Dataset(...)
    >>> ds3 = xr.Dataset(...)
    >>> broadcast_nans([ds1, ds2, ds3])
    """

    # Iterate over all pairs of datasets
    for dsA, dsB in itertools.combinations(list_of_datasets, 2):
        # Find the shared coordinates for all dimensions
        common_coords = {
            dim: sorted(set(dsA[dim].values) & set(dsB[dim].values))
            for dim in dsA.dims
            if dim in dsB.dims
        }
        
        # Iterate over all variables
        for var in dsA.data_vars:
            if var in dsB:  # Ensure both datasets have the variable
                # Select the data at common coordinates
                selA = dsA[var].sel(**common_coords)
                selB = dsB[var].sel(**common_coords)

                # Compute NaN mask for shared coordinates
                nan_mask = selA.isnull() | selB.isnull()

                # Apply NaN mask back to both datasets
                dsA[var].loc[common_coords] = dsA[var].sel(**common_coords).where(~nan_mask)
                dsB[var].loc[common_coords] = dsB[var].sel(**common_coords).where(~nan_mask)


def concat_dict_along_keys(dict_of_datasets : Dict[str,xr.Dataset], dim : str) -> xr.Dataset:
    """Concatenate a dictionary of xarray datasets along a given dimension.
    This function takes a dictionary of xarray Datasets and concatenates them along a specified
    dimension, using the dictionary keys as coordinate values for the new dimension.
    
    Parameters
    ----------
    dict_of_datasets : Dict[str, xr.Dataset]
        Dictionary of xarray Datasets to concatenate. The keys will be used as coordinate values
        for the new dimension.
    dim : str
        Name of the dimension along which to concatenate the datasets.
    
    Returns
    -------
    xr.Dataset
        A single concatenated xarray Dataset with a new dimension whose coordinates are the
        keys from the input dictionary.
    
    Examples
    --------
    >>> ds1 = xr.Dataset({'var': ('x', [1, 2])})
    >>> ds2 = xr.Dataset({'var': ('x', [3, 4])})
    >>> data_dict = {'a': ds1, 'b': ds2}
    >>> result = concat_dict_along_keys(data_dict, 'new_dim')
    """
    coords = list(dict_of_datasets.keys())
    combined_ds = xr.concat(
        dict_of_datasets.values(), 
        dim=xr.Variable(dim, coords)
    )
    return combined_ds


def prep_config(full_config : Dict, sub_config_key : str) -> Dict[str,str]:
    """
    Prepare a configuration dictionary by merging a sub-configuration with the main configuration.
    This function takes a full configuration dictionary and a key for a sub-configuration. 
    It creates a new dictionary that contains all key-value pairs from the full configuration 
    except for the sub-configuration key itself. If the sub-configuration exists and is a dictionary, 
    its key-value pairs are merged into the result, potentially overwriting existing keys.
    Args:
        full_config (Dict): The complete configuration dictionary containing all settings
        sub_config_key (str): The key identifying the sub-configuration to be merged
    Returns:
        Dict[str,str]: A new dictionary containing the merged configuration, where the 
            sub-configuration values override any duplicate keys from the main configuration
    Example:
        >>> full_config = {
        ...     'a': 1, 
        ...     'b': 2, 
        ...     'sub': {'b': 3, 'c': 4}
        ... }
        >>> prep_config(full_config, 'sub')
        {'a': 1, 'b': 3, 'c': 4}
    """
    # Create a copy without the config_key
    result = {k: v for k, v in full_config.items() if k != sub_config_key}
    
    # If the config_key exists and contains a dictionary
    if sub_config_key in full_config and isinstance(full_config[sub_config_key], dict):
        sub_config = full_config[sub_config_key]
        
        # Update the main level keys with values from user_config
        for key, value in sub_config.items():
            result[key] = value
            
    return result