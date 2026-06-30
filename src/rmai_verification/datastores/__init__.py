import numpy as np
from typing import Dict, List
import logging

from .base import BaseDataStore

from ..utils.time import to_timedelta64
from ..utils.files import get_filenames

from .anemoi_datasets import AnemoiDatasets
from .anemoi_inference import AnemoiInference
from .anemoi_ensemble_inference import AnemoiEnsembleInference, AnemoiEnsembleMeanInference
from .rmi_re_pytools import RmiRePytoolsForecast, RmiRePytoolsObservation
from .ifs_fcst import IfsForecast
from .base import PointObservations

LOG = logging.getLogger(__name__)


DATASTORES = {
    "anemoi-inference": AnemoiInference,
    "anemoi-datasets": AnemoiDatasets,
    "anemoi-ensemble-inference": AnemoiEnsembleInference,
    "anemoi-ensemble-mean-inference": AnemoiEnsembleMeanInference,
    "rmi-re-pytools-fc": RmiRePytoolsForecast,
    "rmi-re-pytools-obs": RmiRePytoolsObservation,
    "point-observations": PointObservations,
    "ifs-forecast": IfsForecast,
}

ENSEMBLE_TYPES = {"anemoi-ensemble-inference", "anemoi-ensemble-mean-inference"}
ALLOWED_MEMBER_TYPE = "anemoi-inference"


def _load_single_store(
    config: Dict,
    start_date: np.datetime64,
    end_date: np.datetime64,
    frequency: np.timedelta64,
) -> BaseDataStore:
    """Instantiate a single non-ensemble datastore from a config dict.

    Args:
        config: Datastore configuration dictionary (must contain 'type' and 'path').
        start_date: Start date for file globbing.
        end_date: End date for file globbing.
        frequency: Frequency for file globbing.

    Returns:
        Instantiated datastore.

    Raises:
        ValueError: If the store type is an ensemble type (not allowed here).
    """
    store_type = config["type"]

    if store_type in ENSEMBLE_TYPES:
        raise ValueError(
            f"Ensemble type '{store_type}' cannot be used as an ensemble member. "
            f"Only '{ALLOWED_MEMBER_TYPE}' is allowed."
        )

    files = get_filenames(
        path_fmt=config["path"],
        start=start_date,
        end=end_date,
        frequency=frequency,
    )
    LOG.debug("Loading files %s", ", ".join(files))

    store_cls = DATASTORES[store_type]
    return store_cls(
        files=files,
        **{k: v for k, v in config.items() if k not in ("path", "type")},
    )


def _load_ensemble_store(
    store_type: str,
    config: Dict,
    start_date: np.datetime64,
    end_date: np.datetime64,
    frequency: np.timedelta64,
) -> BaseDataStore:
    """Instantiate an ensemble datastore from a config dict.

    Each entry in config['members'] is a full datastore config. Only
    'anemoi-inference' is allowed as a member type.

    Args:
        store_type: Either 'anemoi-ensemble-inference' or
            'anemoi-ensemble-mean-inference'.
        config: Datastore configuration dictionary. Must contain a 'members'
            key with a list of member configs.
        start_date: Start date for file globbing.
        end_date: End date for file globbing.
        frequency: Frequency for file globbing.

    Returns:
        Instantiated ensemble datastore.

    Raises:
        ValueError: If 'members' is missing/empty, or if any member has a
            type other than 'anemoi-inference'.
    """
    member_configs = config.get("members")

    if not member_configs:
        raise ValueError(
            f"Ensemble datastore '{store_type}' requires a non-empty 'members' list."
        )

    members = []
    for i, member_config in enumerate(member_configs):
        member_type = member_config.get("type")
        if member_type != ALLOWED_MEMBER_TYPE:
            raise ValueError(
                f"Member {i} has type '{member_type}', but only "
                f"'{ALLOWED_MEMBER_TYPE}' is allowed as an ensemble member."
            )
        LOG.debug("Loading ensemble member %d", i)
        members.append(
            _load_single_store(member_config, start_date, end_date, frequency)
        )

    store_cls = DATASTORES[store_type]
    return store_cls(
        members=members,
        **{k: v for k, v in config.items() if k not in ("members", "type")},
    )


def load_datastores(
    datastores: Dict[str, Dict],
    start_date: str,
    end_date: str,
    frequency: str,
) -> Dict[str, BaseDataStore]:
    """Load datastores from a configuration dictionary.

    Args:
        datastores: Dictionary of datastore configurations. Each key is the
            name of the datastore, and the value is the configuration dict.
        start_date: Start date for file globbing.
        end_date: End date for file globbing.
        frequency: Frequency for file globbing.

    Returns:
        Dictionary of loaded datastores.
    """
    stores = dict()

    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date)
    frequency = to_timedelta64(frequency)

    for name, config in datastores.items():
        store_type = config["type"]
        LOG.debug("Loading datastore '%s' of type '%s'", name, store_type)

        if store_type in ENSEMBLE_TYPES:
            stores[name] = _load_ensemble_store(
                store_type, config, start_date, end_date, frequency
            )
        else:
            stores[name] = _load_single_store(
                config, start_date, end_date, frequency
            )

    return stores


def select_variables(
    datastores: Dict[str, BaseDataStore], variables: List[str]
) -> None:
    for store in datastores.values():
        store.select_variables(variables)