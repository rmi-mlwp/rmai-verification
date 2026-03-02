import numpy as np
from typing import Dict, List
import logging

from .base import BaseDataStore

from ..utils.time import to_timedelta64
from ..utils.files import get_filenames

from .anemoi_datasets import AnemoiDatasets
from .anemoi_inference import AnemoiInference
from .rmi_re_pytools import RmiRePytoolsForecast, RmiRePytoolsObservation
from .ifs_fcst import IfsForecast
from .base import PointObservations

LOG = logging.getLogger(__name__)


DATASTORES = {
    "anemoi-inference": AnemoiInference,
    "anemoi-datasets" : AnemoiDatasets,
    "rmi-re-pytools-fc": RmiRePytoolsForecast,
    "rmi-re-pytools-obs": RmiRePytoolsObservation,
    "point-observations" : PointObservations,
    "ifs-forecast": IfsForecast
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
        path = config["path"]
        store_type = config["type"]

        files = get_filenames(
            path_fmt=path,
            start=start_date,
            end=end_date,
            frequency=frequency,
        )
        LOG.debug("Loading files %s", ", ".join(files))

        store_cls = DATASTORES[store_type]

        stores[name] = store_cls(
            files=files,
            **{k: v for k, v in config.items() if k not in ("path", "type")}
        )

    return stores


def select_variables(datastores : Dict[str, BaseDataStore], variables : List[str]):
    for store in datastores.values():
        store.select_variables(variables)