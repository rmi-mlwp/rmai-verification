import logging
import numpy as np
import xarray as xr

from typing import List, Union, Dict

from .base import GridDataStore, FcstDataStore
from .anemoi_inference import AnemoiInference
from ..grids.grid_mapping import add_xy

LOG = logging.getLogger(__name__)


def _validate_members(members: List[AnemoiInference]) -> None:
    """Validate that all ensemble members have compatible dimensions.

    Args:
        members: List of AnemoiInference objects to validate.

    Raises:
        ValueError: If any member has incompatible dimensions, with a
            detailed breakdown of all members' dimensions.
    """
    if len(members) == 0:
        raise ValueError("No ensemble members provided.")

    reference_dims = dict(members[0].dims)

    mismatches = []
    for i, member in enumerate(members[1:], start=1):
        member_dims = dict(member.dims)
        if member_dims != reference_dims:
            mismatches.append(i)

    if mismatches:
        lines = ["Ensemble members have incompatible dimensions:"]
        lines.append(f"  Member 0 (reference): {reference_dims}")
        for i in mismatches:
            lines.append(f"  Member {i} (mismatch): {dict(members[i].dims)}")
        raise ValueError("\n".join(lines))


def _stack_members(members: List[AnemoiInference]) -> xr.Dataset:
    """Stack a list of AnemoiInference datastores along a new 'ensemble' dimension.

    Args:
        members: List of validated AnemoiInference objects.

    Returns:
        xr.Dataset with an additional 'ensemble' dimension.
    """
    datasets = [m.data for m in members]
    stacked = xr.concat(datasets, dim="ensemble").assign_coords(
        ensemble=("ensemble", np.arange(len(datasets)))
    )
    # After concat, shared coords like reference_time and valid_time gain
    # a redundant ensemble dimension. Drop it by reassigning from member 0.
    for coord in ["reference_time", "lead_time", "valid_time"]:
        if coord in stacked.coords and "ensemble" in stacked[coord].dims:
            stacked[coord] = members[0].data[coord]
    return stacked


class AnemoiEnsembleInference(GridDataStore, FcstDataStore):
    """Datastore for a collection of AnemoiInference ensemble members.

    Stacks all members along a new 'ensemble' dimension (integer indices
    starting from 0). The resulting dataset has dimensions:
    (ensemble, reference_time, lead_time, grid_index).
    """

    def __init__(
        self,
        members: List[AnemoiInference],
        variables: Union[List[str], tuple, set] = None,
        mapping: Union[Dict[str, str], str] = None,
    ) -> None:
        """Initialize the AnemoiEnsembleInference datastore.

        Args:
            members: List of already-constructed AnemoiInference objects,
                one per ensemble member. All must have identical dimensions.
            variables: Optional list of variables to select from the dataset.
                If None, all variables are kept.
            mapping: Optional grid mapping to add x/y coordinates.

        Raises:
            ValueError: If members list is empty or members have incompatible
                dimensions.
        """
        LOG.info("Initializing AnemoiEnsembleInference datastore")

        _validate_members(members)

        self._members = members
        self._mapping = mapping
        self._stacked = True

        self._data = _stack_members(members)

        if variables:
            self.select_variables(variables)

        if self._mapping:
            self._data = add_xy(self._data, self._mapping)

        LOG.info(
            "Finished initializing AnemoiEnsembleInference datastore with "
            "%d members and dimensions: %s",
            len(members),
            dict(self._data.sizes),
        )

    def unstack(self, mapping: Union[str, Dict[str, str]] = None) -> None:
        """Unstack the dataset from stacked (1-D grid_index) to 2-D (y, x).

        Resulting dimension order:
        (ensemble, reference_time, lead_time, y, x).

        Args:
            mapping: Optional grid mapping to add x/y coordinates. Required
                if no mapping was provided at construction time.

        Raises:
            ValueError: If no mapping is available or one was already set.
        """
        LOG.debug("Start unstacking, stacked state is currently: %s", self._stacked)
        if not self._stacked:
            return

        if mapping is not None:
            if self._mapping is not None:
                LOG.error("Dataset already contains a mapping")
                raise ValueError("Dataset already contains a mapping.")
            self._mapping = mapping
            self._data = add_xy(self._data, mapping)
        elif self._mapping is None:
            LOG.error("No grid mapping found!")
            raise ValueError(
                "No grid mapping found. Provide a mapping either at "
                "construction time or when calling unstack()."
            )

        ds_unstacked = self._data.unstack()
        self._data = ds_unstacked.transpose(
            "ensemble",
            "reference_time",
            "lead_time",
            "y",
            "x",
        )
        self._stacked = False


class AnemoiEnsembleMeanInference(GridDataStore, FcstDataStore):
    """Datastore for the ensemble mean of a collection of AnemoiInference members.

    Averages all members over the 'ensemble' dimension so the resulting
    dataset has the same dimensions as a single AnemoiInference:
    (reference_time, lead_time, grid_index).
    """

    def __init__(
        self,
        members: List[AnemoiInference],
        variables: Union[List[str], tuple, set] = None,
        mapping: Union[Dict[str, str], str] = None,
    ) -> None:
        """Initialize the AnemoiEnsembleMeanInference datastore.

        Args:
            members: List of already-constructed AnemoiInference objects,
                one per ensemble member. All must have identical dimensions.
            variables: Optional list of variables to select from the dataset.
                If None, all variables are kept.
            mapping: Optional grid mapping to add x/y coordinates.

        Raises:
            ValueError: If members list is empty or members have incompatible
                dimensions.
        """
        LOG.info("Initializing AnemoiEnsembleMeanInference datastore")

        _validate_members(members)

        self._members = members
        self._mapping = mapping
        self._stacked = True

        stacked = _stack_members(members)
        self._data = stacked.mean(dim="ensemble")

        if variables:
            self.select_variables(variables)

        if self._mapping:
            self._data = add_xy(self._data, self._mapping)

        LOG.info(
            "Finished initializing AnemoiEnsembleMeanInference datastore with "
            "%d members and dimensions: %s",
            len(members),
            dict(self._data.sizes),
        )

    def unstack(self, mapping: Union[str, Dict[str, str]] = None) -> None:
        """Unstack the dataset from stacked (1-D grid_index) to 2-D (y, x).

        Resulting dimension order: (reference_time, lead_time, y, x).

        Args:
            mapping: Optional grid mapping to add x/y coordinates. Required
                if no mapping was provided at construction time.

        Raises:
            ValueError: If no mapping is available or one was already set.
        """
        LOG.debug("Start unstacking, stacked state is currently: %s", self._stacked)
        if not self._stacked:
            return

        if mapping is not None:
            if self._mapping is not None:
                LOG.error("Dataset already contains a mapping")
                raise ValueError("Dataset already contains a mapping.")
            self._mapping = mapping
            self._data = add_xy(self._data, mapping)
        elif self._mapping is None:
            LOG.error("No grid mapping found!")
            raise ValueError(
                "No grid mapping found. Provide a mapping either at "
                "construction time or when calling unstack()."
            )

        ds_unstacked = self._data.unstack()
        self._data = ds_unstacked.transpose(
            "reference_time",
            "lead_time",
            "y",
            "x",
        )
        self._stacked = False