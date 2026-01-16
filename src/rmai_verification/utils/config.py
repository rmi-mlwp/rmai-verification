import yaml
import os
import logging

from typing import List, Dict

class Config:
    def __init__(self, config: Dict | str):
        self.config = load_yaml(config) if isinstance(config, str) else config
        assert isinstance(self.config, dict), "config should be a dictionary"

    def __getitem__(self, main_key):
        config = self.config.get(main_key,None)
        if config:
            return config.copy()
        else:
            return config
        
    def save(self, path):
        if not (path.endswith(".yaml") or path.endswith(".yml")):
            raise NameError("config file must have a .yaml or .yml extension.")
        write_yaml(self.config, path)


def load_yaml(fn: str) -> Dict:
    """Load a YAML file into a dictionary.

    Args:
        fn (str): Filename of the YAML file.    
    
    Returns:
        Dict: Dictionary containing the contents of the YAML file.
    """
    with open(fn,'r') as f:
        return yaml.safe_load(f)
    
def write_yaml(x: Dict, fn: str, sort_keys: bool = False) -> None:
    """Write a dictionary to a YAML file.

    Args:
        x (Dict): Dictionary to write to the YAML file.
        fn (str): Filename for the YAML file.
        sort_keys (bool, optional): Whether to sort the keys in the output. Defaults to False.
    """
    with open (fn,'w') as f:
        yaml.dump(x, f, sort_keys=sort_keys)