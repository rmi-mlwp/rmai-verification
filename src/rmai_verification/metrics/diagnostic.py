import logging

import numpy as np
import xarray as xr

from typing import List, Literal

LOG = logging.getLogger(__name__)


def spatial_frequency_spectrum(
    fcst: xr.Dataset | xr.DataArray,
    avg_dim: List[str],
    dx: float = 5.5,
    method: Literal["fft", "dct"] = "dct",
    k_bins: int = 50,
    skipna: bool = True,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate the spatial frequency spectrum of the forecast field using 2D FFT or DCT.

    This is a diagnostic metric that computes the power spectrum of the forecast field by:
    1. Applying 2D FFT or DCT to the (x, y) spatial grid
    2. Computing power spectrum |F|²
    3. Performing isotropic radial averaging (binning by radial frequency k)
    4. Returning spectrum with (x, y) dimensions replaced by 'k' (wavenumber)

    Parameters
    ----------
    fcst : xr.Dataset | xr.DataArray
        Forecast data with dimensions including 'x' and 'y' spatial dimensions.
    avg_dim : List[str]
        Dimensions over which to average the final spectrum.
    dx : float, optional
        Grid spacing in km. Default is 1.0. Used to convert normalized frequencies to physical wavenumbers.
    method : {'fft', 'dct'}, optional
        Transform method to use. Default is 'dct'.
        - 'fft': 2D Fast Fourier Transform (assumes periodic boundaries)
        - 'dct': 2D Discrete Cosine Transform Type II (assumes mirror boundaries)
    k_bins : int, optional
        Number of radial frequency bins for isotropic averaging. Default is 50.
    skipna : bool, optional
        If True, skip NaN values when averaging. Default is True.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Power spectrum with (x, y) dimensions replaced by 'k' (cycles per unit dx = wave number).
        Other dimensions are preserved (except those in avg_dim).

    Notes
    -----
    This is a DIAGNOSTIC metric (forecast-only, no observations needed).
    Assumes data is already on a regular (x, y) grid.

    Physical interpretation:
    - Low k → large-scale structures (weather systems, synoptic scale)
    - High k → small-scale features (gravity waves, turbulence, noise)

    Boundary Conditions:
    - FFT: Periodic boundaries (suitable for global models, spectral cycling)
    - DCT: Mirror boundaries (suitable for LAM, bounded domains, more physical)
    """
    # Validate inputs
    if method.lower() not in ["fft", "dct"]:
        raise ValueError(f"method must be 'fft' or 'dct', got {method}")

    if method.lower() == "dct":
        try:
            from scipy.fftpack import dctn
        except ImportError:
            raise ImportError("DCT method requires scipy. Install with: pip install scipy")

    LOG.info(
        f"Computing spatial frequency spectrum of the forecast field using {method.upper()}, "
        f"with {k_bins} frequency bins, dx={dx}, and average dimensions: {avg_dim}"
    )

    # Recursively apply the function for each variable if the input is a Dataset
    if isinstance(fcst, xr.Dataset):
        LOG.info("Processing Dataset with variables: %s", list(fcst.data_vars))
        results = {}
        for var_name, da in fcst.data_vars.items():
            LOG.info("Processing variable: %s", var_name)
            results[var_name] = spatial_frequency_spectrum(
                da,
                avg_dim=avg_dim,
                dx=dx,
                method=method,
                k_bins=k_bins,
                skipna=skipna,
            )
        return xr.Dataset(results)

    # Ensure x and y dimensions exist
    if "x" not in fcst.dims or "y" not in fcst.dims:
        raise ValueError("Input must have 'x' and 'y' dimensions")

    # Get grid dimensions
    ny, nx = fcst.sizes["y"], fcst.sizes["x"]

    # Compute wavenumber grids based on method
    if method.lower() == "dct":
        # DCT uses only non-negative wavenumbers [0, 0.5/dx]
        ky = np.arange(ny) / (2 * ny * dx)
        kx = np.arange(nx) / (2 * nx * dx)
        k_max = 0.5 / dx
        LOG.info(f"Using DCT with k_max = {k_max:.6f} cycles per unit dx")
    else:  # FFT
        # FFT uses full symmetric spectrum, but we only care about positive frequencies
        # Range is [0, 0.5/dx] for consistency with DCT
        ky = np.fft.fftshift(np.fft.fftfreq(ny)) / dx
        kx = np.fft.fftshift(np.fft.fftfreq(nx)) / dx
        # Keep only non-negative frequencies
        k_max = 0.5 / dx
        LOG.info(f"Using FFT with k_max = {k_max:.6f} cycles per km")

    # Create 2D meshgrid of wavenumbers
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    k_radial = np.sqrt(kx_mesh**2 + ky_mesh**2)

    # Compute wavenumber bins - use scalar Nyquist limit [0, 0.5/dx]
    k_bin_edges = np.linspace(0, k_max, k_bins + 1)
    k_values = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2

    LOG.info(f"Wavenumber range: [{k_bin_edges[0]:.6f}, {k_bin_edges[-1]:.6f}] cycles per unit dx")
    LOG.info(f"Bin centers: {k_values[0]:.6f} to {k_values[-1]:.6f}")

    # Precompute bin masks for vectorized binning
    bin_masks = []
    for bin_idx in range(k_bins):
        mask = (k_radial >= k_bin_edges[bin_idx]) & (k_radial < k_bin_edges[bin_idx + 1])
        bin_masks.append(mask)

    # Stack all non-spatial dimensions for iteration
    other_dims = [d for d in fcst.dims if d not in ["x", "y"]]
    if other_dims:
        fcst_stacked = fcst.stack(stacked_dim=other_dims)
        n_stacked = fcst_stacked.sizes["stacked_dim"]
    else:
        fcst_stacked = fcst.expand_dims("stacked_dim")
        n_stacked = 1

    # Initialize output array
    power_spectra = np.zeros((n_stacked, k_bins))

    # Process each combination of other dimensions
    for idx in range(n_stacked):
        if idx % max(1, n_stacked // 10) == 0:
            LOG.info("Progress: %d/%d combinations processed", idx, n_stacked)

        # Extract field for this combination
        if other_dims:
            field_2d = fcst_stacked.isel(stacked_dim=idx).values
        else:
            field_2d = fcst_stacked.isel(stacked_dim=0).values

        # Handle skipna by replacing NaNs with 0 or mean
        if skipna:
            if np.all(np.isnan(field_2d)):
                LOG.debug("All values are NaN at index %d. Using NaN spectrum.", idx)
                power_spectra[idx, :] = np.nan
                continue
            # Replace NaNs with the mean of non-NaN values
            valid_mask = ~np.isnan(field_2d)
            if np.any(valid_mask):
                field_2d = np.where(valid_mask, field_2d, np.nanmean(field_2d))
            else:
                field_2d = np.zeros_like(field_2d)

        # Compute transform based on method
        if method.lower() == "dct":     # DCT Type II with orthonormal normalization
            transform_coeffs = dctn(field_2d, type=2, norm="ortho")
            power_2d = transform_coeffs**2
        else:                           # 2D FFT
            fft_2d = np.fft.fft2(field_2d)
            power_2d = np.abs(fft_2d) ** 2
            power_2d = np.fft.fftshift(power_2d)

        # Vectorized isotropic radial averaging using precomputed masks
        for bin_idx in range(k_bins):
            mask = bin_masks[bin_idx]
            if np.any(mask):
                power_spectra[idx, bin_idx] = np.mean(power_2d[mask])
            else:
                power_spectra[idx, bin_idx] = np.nan

    # Reshape back to original dimensions (except x, y now k)
    if other_dims:
        # Create coordinate dict for other dimensions
        coords_dict = {d: fcst.coords[d] for d in other_dims if d in fcst.coords}
        coords_dict["k"] = k_values

        # Reshape power_spectra back to (other_dims..., k_bins)
        output_shape = [fcst.sizes[d] for d in other_dims] + [k_bins]
        power_reshaped = power_spectra.reshape(output_shape)

        result = xr.DataArray(
            power_reshaped,
            dims=other_dims + ["k"],
            coords=coords_dict,
            attrs={
                "long_name": f"Isotropic radial power spectrum of forecast (computed with {method.upper()})",
                "units": "power",
                "transform": method.upper(),
                "k_bins": k_bins,
                "dx": dx,
                "metric_type": "diagnostic",
            },
        )
    else:
        result = xr.DataArray(
            power_spectra[0],
            dims=["k"],
            coords={"k": k_values},
            attrs={
                "long_name": f"Isotropic radial power spectrum of forecast (computed with {method.upper()})",
                "units": "power",
                "transform": method.upper(),
                "k_bins": k_bins,
                "dx": dx,
                "metric_type": "diagnostic",
            },
        )

    # Average over specified dimensions
    if avg_dim:
        LOG.info("Averaging spectrum over dimensions: %s", avg_dim)
        result = result.mean(dim=avg_dim, skipna=skipna)

    LOG.info(
        "Spatial frequency spectrum computation complete using %s. Shape: %s",
        method.upper(),
        result.shape,
    )
    return result


def checkerboard(fcst: xr.Dataset | xr.DataArray, avg_dim: List[str] = None, skipna: bool = True) -> xr.Dataset | xr.DataArray:
    """
    Calculate the checkerboard error of the forecast field.
 
    This is a diagnostic metric that computes the checkerboard (small-scale noise) error
    by comparing alternating quadrants of the 2D spatial field:
    1. Partition the (x, y) grid into four overlapping checkerboard quadrants
    2. Compute normalized difference between opposite quadrants
    3. Average over specified dimensions
 
    Parameters
    ----------
    fcst : xr.Dataset | xr.DataArray
        Forecast data with dimensions including 'x' and 'y' spatial dimensions.
    avg_dim : List[str], optional
        Dimensions over which to average the final metric. If None, no additional
        averaging is performed beyond the spatial averaging. Default is None.
    skipna : bool, optional
        If True, skip NaN values when averaging. Default is True.
 
    Returns
    -------
    xr.Dataset | xr.DataArray
        Checkerboard error metric with (x, y) dimensions roughly halved in size.
        Other dimensions are preserved (except those in avg_dim).
 
    Notes
    -----
    This is a DIAGNOSTIC metric (forecast-only, no observations needed).
    Assumes data is already on a regular (x, y) grid.
    The metric quantifies grid-scale alternating patterns typical of numerical artifacts.
    """
    # Deal with avg_dim None
    if avg_dim is None:
        avg_dim = []
 
    LOG.info(
        f"Computing checkerboard error of the forecast field, "
        f"with average dimensions: {avg_dim if avg_dim else 'None'}"
    )
 
    # Recursively apply the function for each variable if the input is a Dataset
    if isinstance(fcst, xr.Dataset):
        LOG.info("Processing Dataset with variables: %s", list(fcst.data_vars))
        results = {}
        for var_name, da in fcst.data_vars.items():
            LOG.info("Processing variable: %s", var_name)
            results[var_name] = checkerboard(
                da,
                avg_dim=avg_dim,
                skipna=skipna,
            )
        return xr.Dataset(results)
 
    # Ensure x and y dimensions exist
    if "x" not in fcst.dims or "y" not in fcst.dims:
        raise ValueError("Input must have 'x' and 'y' dimensions")
 
    # Stack all non-spatial dimensions for iteration
    other_dims = [d for d in fcst.dims if d not in ["x", "y"]]
    
    if other_dims:
        fcst_stacked = fcst.stack(stacked_dim=other_dims)
        n_stacked = fcst_stacked.sizes["stacked_dim"]
    else:
        fcst_stacked = fcst.expand_dims("stacked_dim")
        n_stacked = 1
 
    # Initialize output array, output will be subsampled (approximately half size)
    checkerboard_errors = np.zeros((n_stacked, fcst.sizes["y"] // 2, fcst.sizes["x"] // 2))

    # Process each combination of other dimensions
    for idx in range(n_stacked):
        if idx % max(1, n_stacked // 10) == 0:
            LOG.info("Progress: %d/%d combinations processed", idx, n_stacked)
 
        # Extract field for this combination
        if other_dims:
            field_2d = fcst_stacked.isel(stacked_dim=idx).values
        else:
            field_2d = fcst_stacked.isel(stacked_dim=0).values
 
        # Handle skipna by replacing NaNs with mean
        if skipna:
            if np.all(np.isnan(field_2d)):
                LOG.debug("All values are NaN at index %d. Using NaN error.", idx)
                checkerboard_errors[idx] = np.nan
                continue
            valid_mask = ~np.isnan(field_2d)
            if np.any(valid_mask):
                field_2d = np.where(valid_mask, field_2d, np.nanmean(field_2d))
            else:
                field_2d = np.zeros_like(field_2d)
 
        # Partition into even-even, even-odd, odd-even, odd-odd quadrants
        EE = field_2d[::2,  ::2]
        EO = field_2d[::2,  1::2]
        OE = field_2d[1::2, ::2]
        OO = field_2d[1::2, 1::2]
    
        # Handle different sizes due to odd dimensions
        mn_y = min(EE.shape[0], OO.shape[0])
        mn_x = min(EE.shape[1], OO.shape[1])
        EE, EO, OE, OO = (a[:mn_y, :mn_x] for a in (EE, EO, OE, OO))
    
        # Compute and save checkerboard error: difference between opposite quadrants
        error = 0.5 * ((EE + OO) - (EO + OE))
        checkerboard_errors[idx] = error
 
    # Reshape back to original dimensions
    if other_dims: 
        # Reshape checkerboard_errors back to (other_dims..., y, x) shape
        output_shape = [fcst.sizes[d] for d in other_dims] + [fcst.sizes["y"] // 2, fcst.sizes["x"] // 2]
        errors_reshaped = checkerboard_errors.reshape(output_shape)

        # Create coordinate dict for other dimensions
        coords_dict = {d: fcst.coords[d] for d in other_dims if d in fcst.coords}
        coords_dict["y"] = fcst.coords["y"][::2][:output_shape[-2]]     # Get y coordinates, subsample by 2, truncate to y output size
        coords_dict["x"] = fcst.coords["x"][::2][:output_shape[-1]]     # Get x coordinates, subsample by 2, truncate to x output size

        result = xr.DataArray(
            errors_reshaped,
            dims=other_dims + ["y", "x"],
            coords=coords_dict,
            attrs={
                "long_name": "Checkerboard error of forecast (grid-scale noise metric)",
                "units": "same as input",
                "metric_type": "diagnostic",
            },
        )
    else:
        result = xr.DataArray(
            checkerboard_errors[0],
            attrs={
                "long_name": "Checkerboard error of forecast (grid-scale noise metric)",
                "units": "same as input",
                "metric_type": "diagnostic",
            },
        )
 
    # Average over specified dimensions
    if avg_dim:
        LOG.info("Averaging checkerboard error over dimensions: %s", avg_dim)
        result = result.mean(dim=avg_dim, skipna=skipna)
 
    LOG.info(
        "Checkerboard error computation complete. Shape: %s",
        result.shape,
    )
    return result


def spread(fcst: xr.Dataset | xr.DataArray, avg_dim: List[str], member_dim: str = "ensemble", ddof: int = 0, skipna: bool = True) -> xr.Dataset | xr.DataArray:
    """
    Ensemble spread: sqrt of mean ensemble variance, pooled over avg_dim and member_dim.
    """
    var = fcst.var(dim=member_dim, ddof=ddof, skipna=skipna)
    mean_var = var.mean(dim=avg_dim, skipna=skipna)
    return np.sqrt(mean_var)