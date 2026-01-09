from datetime  import datetime
import numpy as np
import yaml
import argparse
import os 
import logging

from typing import List, Dict

LOG = logging.getLogger(__name__)

def get_filenames(path_fmt: str ,start : np.datetime64, end : np.datetime64, frequency: np.timedelta64, ens_size: int = 1) -> List[str]:
    """Return a list of filenames matching the path format between start and end dates.
    
    This function generates filenames based on a path format string and a date range,
    skipping dates where no matching file exists.
    
    Args:
        path_fmt (str): Path format string with datetime placeholders:
            {yyyy}: 4-digit year
            {yy}: 2-digit year
            {mm}: 2-digit month
            {dd}: 2-digit day
            {HH}: 2-digit hour
            {MM}: 2-digit minute
            {SS}: 2-digit second
        start (np.datetime64): Start date
        end (np.datetime64): End date
        frequency (np.timedelta64): Time step between each filename check
    
    Returns:
        Union[List[str], str]: List of existing filenames matching the pattern,
            or single filename as string if only one match is found.
            Returns empty list if no files are found.
    
    Example:
        >>> path_fmt = "/data/{yyyy}/{mm}/data_{yyyy}{mm}{dd}.nc"
        >>> start = np.datetime64("2023-01-01")
        >>> end = np.datetime64("2023-01-03")
        >>> freq = np.timedelta64(1, "D")
        >>> get_filenames(path_fmt, start, end, freq)
        ['/data/2023/01/data_20230101.nc', '/data/2023/01/data_20230102.nc']
    """
    filenames = []
    date = start
    ens_idx = 1 
    print("ens_size: ", ens_size)
    while date <= end:
        for ens_idx in range(1, ens_size + 1):
            date_dt = date.astype(datetime)
            print("date_dt: ", date_dt)
            print("ens_idx: ", ens_idx)
            print("Generating path for date: ", date_dt.strftime("%Y%m%d %H:%M"))
            path = path_fmt.format(
                yyyy=date_dt.strftime("%Y"),
            yy=date_dt.strftime("%y"),
            mm=date_dt.strftime("%m"),
            dd=date_dt.strftime("%d"),
            HH=date_dt.strftime("%H"),
            MM=date_dt.strftime("%M"),
            SS=date_dt.strftime("%S"),
            x=str(ens_idx)
        )
            print(f"Checking for file: {path}")
            if not os.path.exists(path):
                LOG.warning(f"No file found for date {date_dt.strftime('%Y%m%d %H:%M')}, skipping file")
                print(f"File not found: {path}")
                pass
            else:
                filenames.append(path)
                LOG.info(f"Found file: {path}")
                print(f"File found: {path}")
        date += frequency
    filenames = list(set(filenames))
    LOG.info(f"Total files found: {len(filenames)}")
    print(f"Total files found: {len(filenames)}")
    if len(filenames) == 1: 
        filenames = filenames[0]
        LOG.info("Only one file found, returning as string instead of list")
        print("Only one file found, returning as string instead of list")
    return filenames

def write_yaml(x: Dict, fn: str, sort_keys: bool = False) -> None:
    """Write a dictionary to a YAML file.

    Args:
        x (Dict): Dictionary to write to the YAML file.
        fn (str): Filename for the YAML file.
        sort_keys (bool, optional): Whether to sort the keys in the output. Defaults to False.
    """
    with open (fn,'w') as f:
        yaml.dump(x,f,sort_keys=False)

def load_yaml(fn: str) -> Dict:
    """Load a YAML file into a dictionary.

    Args:
        fn (str): Filename of the YAML file.    
    
    Returns:
        Dict: Dictionary containing the contents of the YAML file.
    """
    with open(fn,'r') as f:
        return yaml.safe_load(f)
    

def yaml_file_type(filename):
    """Custom argparse type for YAML file validation."""
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(f"File '{filename}' does not exist.")
    if not (filename.endswith(".yaml") or filename.endswith(".yml")):
        raise argparse.ArgumentTypeError("Config file must have a .yaml or .yml extension.")
    
    # Try to load YAML file to check for syntax errors
    try:
        with open(filename, "r") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise argparse.ArgumentTypeError(f"Invalid YAML file: {e}")

    return filename  # Return the valid filename