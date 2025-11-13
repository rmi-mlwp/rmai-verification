import numpy as np
import xarray as xr
from earthkit.meteo.thermo.array import dewpoint_from_specific_humidity
#REQUIRE pip install earthkit-meteo
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="verification.log"
)

class SpecificHumidityToDewpoint():
    """Convert specific humidity and pressure to dewpoint temperature.

    This class computes dewpoint temperature from specific humidity and pressure
    using `earthkit.meteo.thermo.array.dewpoint_from_specific_humidity`.

    Parameters
    ----------
    specific_humidity : str, default="q"
        Name of the specific humidity variable in the input dataset.
    pressure : str, default="p"
        Name of the pressure variable in the input dataset.
    dewpoint : str, default="2d"
        Name of the output dewpoint temperature variable.

    Methods
    -------
    execute(input_ds)
        Computes dewpoint temperature from specific humidity and pressure.

    Examples
    --------
    >>> transformer = SpecificHumidityToDewpoint(
            specific_humidity="q",
            pressure="p",
            dewpoint="2d"
        )
    >>> output_ds = transformer.execute(input_ds)
    """

    def __init__(
        self,
        specific_humidity: str = "q",
        pressure: str = "p",
        dewpoint: str = "2d"
    ) -> None:
        self.specific_humidity = specific_humidity
        self.pressure = pressure
        self.dewpoint = dewpoint

    def execute(self, input_ds: xr.Dataset) -> xr.Dataset:
        """Computes dewpoint temperature from specific humidity and pressure.

        Args:
            input_ds (xr.Dataset): Input dataset containing specific humidity and pressure.

        Returns:
            xr.Dataset: The input dataset with an additional dewpoint temperature variable.

        Raises:
            ValueError: If required variables are missing in the input dataset.
        """
        if not all(var in input_ds for var in [self.specific_humidity, self.pressure]):
            missing_vars = [var for var in [self.specific_humidity, self.pressure] if var not in input_ds]
            raise ValueError(f"Missing required variables: {missing_vars}")

        # # Compute dewpoint using earthkit.meteo
        # dewpoint_kelvin = dewpoint_from_specific_humidity(
        #     specific_humidity=input_ds[self.specific_humidity].values,
        #     pressure=input_ds[self.pressure].values
        # )
        
        # # Convert Kelvin to Celsius
        # dewpoint_celsius = dewpoint_kelvin - 273.15

        # # Add dewpoint to the dataset
        # input_ds[self.dewpoint] = (input_ds[self.specific_humidity].dims, dewpoint_celsius)
        # Ensure data is chunked
        # input_ds = input_ds.chunk({"reference_time": 1, "lead_time": 1, "grid_index": 1000000})  # Adjust chunk sizes

        logging.info("Input dataset structure:\n%s", input_ds)
        # Use xr.apply_ufunc to avoid loading full arrays
        dewpoint_kelvin = xr.apply_ufunc(
            dewpoint_from_specific_humidity,
            input_ds[self.specific_humidity],
            input_ds[self.pressure],
            input_core_dims=[[], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Convert Kelvin to Celsius
        dewpoint_celsius = dewpoint_kelvin - 273.15

        # Add dewpoint to the dataset
        input_ds[self.dewpoint] = dewpoint_celsius
        input_ds[self.dewpoint].attrs = {
            "standard_name": "dew_point_temperature",
            "long_name": "Dewpoint temperature",
            "units": "Celsius",
        }

        return input_ds