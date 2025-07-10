import numpy as np
import xarray as xr

class UVToSpeed():
    """Convert U and V wind components to wind speed.
    This class transforms U and V wind components into wind speed using the formula:
    speed = sqrt(U^2 + V^2)
    Parameters
    ----------
    u : str, default="10u"
        Name of the U wind component variable in the input dataset
    v : str, default="10v"
        Name of the V wind component variable in the input dataset 
    speed : str, default="10s"
        Name of the output wind speed variable
    Methods
    -------
    execute(input_ds)
        Performs the U,V to speed transformation on the input dataset
    Examples
    --------
    >>> transformer = UVToSpeed(u="u10", v="v10", speed="speed10")
    >>> output_ds = transformer.execute(input_ds)
    Notes
    -----
    Currently only supports xarray Dataset inputs. DataArray inputs are not implemented.
    """
   
    def __init__(self, u : str | list = "10u", v : str | list = "10v", speed : str | list= "10s") -> None:
        if not isinstance(u, list):
            u = [u]
        if not isinstance(v, list):
            v = [v]
        if not isinstance(speed, list):
            speed = [speed]
        assert len(u) == len(v) == len(speed), "u, v, and speed must have the same length."
        self.n =len(u)
        self.u_wind = u
        self.v_wind = v
        self.wind_speed = speed

    def execute(self, input_ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """Calculates wind speed from U and V wind components.

        This method computes the wind speed by taking the square root of the sum of squared
        U and V wind components using the Pythagorean theorem.
        
        Args:
            input_ds (xr.Dataset | xr.DataArray): Input dataset containing U and V wind components.
                Currently only supports xarray Dataset, not DataArray.
        
        Returns:
            xr.Dataset | xr.DataArray: The input dataset with an additional variable for wind speed.
            For Dataset inputs, adds new wind speed variable while preserving original data.
        
        Raises:
            NotImplementedError: If input is an xarray DataArray, as this is not yet supported.
        """

        if isinstance(input_ds, xr.Dataset):
            for i in range(self.n):
                speed = np.sqrt(input_ds[self.u_wind[i]]**2 + input_ds[self.v_wind[i]]**2)
                input_ds[self.wind_speed[i]] = speed
        elif isinstance(input_ds, xr.DataArray):
            raise NotImplementedError("UVToSpeed does not support DataArray input yet.")
        return input_ds
