from typing import Dict

from ..datastores.base import BaseDataStore
from .rename import Renamer
from .uv_to_speed import UVToSpeed
from .kelvin_to_celcius import KelvinToCelcius
from .specific_humidity_to_dewpoint import SpecificHumidityToDewpoint



TRANSFORMATIONS = {
    "rename": Renamer,
    "uv_to_speed": UVToSpeed,
    "kelvin_to_celcius": KelvinToCelcius,
    "specific_humidity_to_dewpoint": SpecificHumidityToDewpoint

}

def apply_transformations(datastores : Dict[str, BaseDataStore], transformations: Dict[str, Dict]) -> None:
    """Apply transformations to datastores (in place).

    Args:
        datastores (Dict[str, BaseDataStore]): Dictionary of datastores.
        transformations (Dict[str, Dict]): Dictionary of transformations to apply.
            Each key is the name of the transformation, and the value is a dictionary
            with the transformation configuration.
            The transformation configuration must contain the key "datastores" to specify
            which datastores to apply the transformation to.
            If no "datastores" key is present, the transformation is applied to all datastores.

    Returns:
        None
    """
    for transformation, config in transformations.items():
        active_stores = config.pop("datastores", datastores.keys())
        
        transformer = TRANSFORMATIONS[transformation]
        transformation = transformer(**config)
        for store in active_stores:
            datastores[store].transform(transformation)