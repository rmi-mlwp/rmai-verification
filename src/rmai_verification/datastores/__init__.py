import numpy as np
from typing import Dict, List
import logging

from .base import BaseDataStore

from ..utils.time import to_timedelta64
from ..utils.files import get_filenames

from .anemoi_datasets import AnemoiDatasets
from .anemoi_inference import AnemoiInference
from .bris_inference import BrisInference
from .rmi_re_pytools import RmiRePytoolsForecast, RmiRePytoolsObservation
from .base import PointObservations

LOG = logging.getLogger(__name__)


DATASTORES = {
    "anemoi-inference": AnemoiInference,
    "bris-inference": BrisInference,
    "anemoi-datasets" : AnemoiDatasets,
    "rmi-re-pytools-fc": RmiRePytoolsForecast,
    "rmi-re-pytools-obs": RmiRePytoolsObservation,
    "point-observations" : PointObservations
}
def load_datastores(datastores : Dict[str, Dict], start_date : str, end_date : str, frequency : str) -> Dict[str, BaseDataStore]:
    """Load datastores
    
    Args:
        datastores (Dict[str, Dict]): Dictionary of datastore configurations.
            Each key is the name of the datastore, and the value the datastore configuration dictionary
        start_date (str): Start date for file globbing.
        end_date (str): End date for file globbing.
        frequency (str): Frequency for file globbing.

    Returns:
        Dict[str, BaseDataStore]: Dictionary of loaded datastores.
    """
    stores = dict()

    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date)
    frequency = to_timedelta64(frequency)

    for name, config in datastores.items():
        files = get_filenames(
            path_fmt=config.pop("path"),
            start=start_date,
            end=end_date,
            frequency=frequency,
            ens_size=config.get("ens_size", 1)
        )
        LOG.debug("Loading files %s", ", ".join(files))
        store = DATASTORES[config.pop("type")]

        stores[name] = store(files=files,**config)

    return stores


def select_variables(datastores : Dict[str, BaseDataStore], variables : List[str]):
    for store in datastores.values():
        store.select_variables(variables)