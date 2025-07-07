import numpy as np
import xarray as xr
import logging
from typing import List

LOG = logging.getLogger(__name__)

FACTOR = 100

class HectoPascalToPascal():
    """Convert pressure data between hecto Pascal and Pascal scales.
    This class implements a transformation to convert pressure units. It can operate on both xarray DataArrays and Datasets.
    Parameters
    ----------
    fields : str or List[str], default="msl"
        The field(s) to transform. Can be a single field name as string or a list of field names.
        These should correspond to temperature variables in the dataset.
    inverse : bool, default=False
        If False, converts from hPa to Pa.
        If True, converts from Pa to hPa.
    Methods
    -------
    execute(input_ds : xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset
        Performs the pressure conversion on the input data.
    Examples
    --------
    >>> transformer = HectoPascalToPascal(fields=['msl', 'sp'])
    >>> transformed_data = transformer.execute(input_dataset)
    Notes
    -----
    The transformation uses the standard conversion between hPa and Pa:
    - hecto Pascal to Pascal: T(Pa) = T(hPa)*100
    - Pascal to hecto Pascal: T(hPa) = T(Pa)/100
    """

    def __init__(self, fields: str | List[str] = "2t" , inverse: bool = False):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = fields
        self.factor =  FACTOR if not inverse else 1 / FACTOR

    def execute(self,input_ds : xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        """Execute the hPa to Pa transformation on input data.
        This method converts pressure values from hPa to Pa by multiplying by 100 (or dividing by 100 if inverse is True).
        (stored in self.factor) from the specified pressure fields.
        Parameters
        ----------
        input_ds : xr.DataArray or xr.Dataset
            Input data containing temperature values in hPa. Can be either a DataArray
            with multiple variables or a Dataset with temperature fields.
        Returns
        -------
        xr.DataArray or xr.Dataset
            Input data with temperature values converted to Pa for the specified fields.
            The input type (DataArray or Dataset) is preserved in the output.
        Notes
        -----
        - For DataArrays, the transformation is applied to all values where the "variable"
          coordinate matches the specified fields
        - For Datasets, the transformation is applied directly to the specified field variables
        - All attributes of the input data are preserved
        """
        
        if isinstance(input_ds, xr.DataArray):
            LOG.debug("Transforming an xr.DataArray")
            input_ds = xr.where(input_ds["variable"].isin(self.fields), input_ds*self.factor, input_ds, keep_attrs=True)          
        else:
            LOG.debug("Transforming an xr.Dataset")
            for field in self.fields:
                input_ds[field] = input_ds[field]*self.factor
        
        return input_ds
