from typing import Dict 
import xarray as xr

class Renamer():
    """Rename variables in xarray Dataset or DataArray based on a provided mapping dictionary.
    This class allows for renaming of variables in either xarray Datasets or DataArrays using
    a dictionary that maps new names to lists of old names.
    Parameters
    ----------
    rename_dict : Dict[str, str]
        Dictionary mapping new names (keys) to old names (values). For each new name,
        multiple old names can be provided that will be renamed to the new name.
    Methods
    -------
    execute(input_ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray
        Applies the renaming transformation to the input Dataset or DataArray.
    Examples
    --------
    >>> rename_dict = {"temperature": ["temp", "t2m"], "precipitation": ["prec", "tp"]}
    >>> renamer = Renamer(rename_dict)
    >>> renamed_ds = renamer.execute(dataset)
    Notes
    -----
    - For DataArrays, it assumes the presence of a 'variable' coordinate
    - When processing DataArrays, if an old name isn't found in the mapping,
      it will keep the original name
    """
    
    def __init__(self, rename_dict: Dict[str, str]):
        self.rename_dict: Dict = rename_dict
    
    def execute(self, input_ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """
        Executes the renaming transformation on the input dataset or data array.

        This method renames variables in either an xarray Dataset or DataArray based on the
        rename dictionary provided during initialization. For Datasets, it directly renames
        the variables. For DataArrays, it renames values in the 'variable' coordinate.

        Parameters
        ----------
        input_ds : xr.Dataset | xr.DataArray
            The input dataset or data array to be transformed. If DataArray, must have a
            'variable' coordinate.

        Returns
        -------
        xr.Dataset | xr.DataArray
            A new dataset or data array with renamed variables according to rename_dict.
            Original data is preserved, only names are changed.

        Raises
        ------
        AssertionError
            If a DataArray is provided without a 'variable' coordinate.
        """
        if isinstance(input_ds, xr.Dataset):
            new_dict = dict()
            for new_name,old_names in self.rename_dict.items():
                if isinstance(old_names, str):
                    old_names = [old_names]
                for name in input_ds.keys():
                    if name in old_names:
                        new_dict[name]=new_name
                    else:
                        pass
            print(f"Renaming variables in Dataset using mapping: {new_dict}")
            print(f"Original variables: {list(input_ds.keys())}")
            new_ds = input_ds.rename(new_dict)
        else:
            assert "variable" in input_ds.coords(), "DataArray must have a variable coordinate"
            new_names = []
            for variable in input_ds["variable"].values:
                name = None
                for new_name, old_names in self.rename_dict.items():
                    if variable in old_names:
                        name = new_name
                if name == None:
                    name = variable.astype(str)
                new_names.append(name)
            new_ds = input_ds.assign_coords(
                {
                    "variable": ("variable", new_names)
                }
            )
        return new_ds

