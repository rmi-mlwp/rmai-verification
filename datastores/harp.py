import numpy as np
import xarray as xr
import sqlite3
import pandas as pd

import logging

from .base import PointDataStore, ObsDataStore
from transformations import Renamer

LOG = logging.getLogger(__name__)

COORDS = dict(
    longitude="lon",
    latitude="lat",
    valid_time="validdate",
    code="SID",
    altitude="elev"
)


class HarpObsTable(PointDataStore, ObsDataStore):
    def __init__(self, files, variables=None):
        LOG.info("Initializing HarpObsTable datastore")
        if isinstance(files,list) and len(files) > 1:
            LOG.error("Reading from mutliple SQLite-files not implemented")
            raise NotImplementedError
        self._files = files
        self._conn = sqlite3.connect(self._files)

        vars = [ var for var in 
                pd.read_sql("SELECT * FROM SYNOP LIMIT 0", self._conn).columns 
                if var not in COORDS.values() ]
    
        if variables is not None:
            vars = [var for var in vars if var in variables]

        vars_dict = dict()
        for var in vars:
            vars_dict[var] = var
        self._vars = vars_dict

        df = pd.read_sql_query(f"SELECT DISTINCT validdate FROM SYNOP",self._conn)
        self._valid_times = pd.to_datetime(df["validdate"],unit="s").values

        self._data = None
        self._transformations = []
        
        self._codes = pd.read_sql(
            f"SELECT SID as code, MIN(lat) AS latitude, MIN(lon) AS longitude, elev as altitude FROM SYNOP GROUP BY SID",
            self._conn,
            index_col="code"
        )
        self._dims = dict(
            code = len(self._codes),
            valid_time = len(self._valid_times)
        )

    @property
    def dims(self):
        if self._data is None:
            return self._dims
        else:
            return super().dims

    @property
    def vars(self):
        if self._data is None:
            return list(self._vars.keys())
        else:
            return super().vars
    
    @property
    def valid_times(self):
        if self._data is None:
            return self._valid_times
        else: 
            return super().valid_times
    
    @property
    def longitudes(self):
        if self._data is None:
            return self._codes["lon"].to_list()
        else:
            return super().longitudes()

    @property
    def latitudes(self):
        if self._data is None:
            return self._codes["lat"].to_list()
        else: 
            return super().latitudes()
    
    def select_variables(self, variables):
        if self._data is None:
            for var in variables:
                if var not in self._vars.keys():
                    LOG.error(f"variable {var} not present in the datastore")
                    raise KeyError
            self._vars = {key: self._vars[key] for key in variables}
        else:
            new_data = self._data[variables]
            self._data = new_data
    
    def select_valid_times(self, valid_times):
        # TODO: What in the rare case that we want to use the observations 
        # verified against other observations, in that case the 
        # valid_dates are a 2-D array as function of reference_time and lead_time
        # We should first find the unique valid_dates, load these from the database
        # and the use .sel on the xarray
        if self._data is None:
            _temp_valid = []
            for valid_time in valid_times:
                if valid_time not in self._valid_times:
                    LOG.warning(f"valid_time {valid_time} not present in the datastore")
                else:
                    _temp_valid.append(valid_time)
            self._valid_times = valid_times
        else:
            new_data = self._data.sel(valid_time=valid_times)
            self._data = new_data
    
    @property
    def data(self):
        if self._data is None:
            query = f"""
                SELECT SID as code, validdate as valid_time, {", ".join(self._vars.values())}
                FROM SYNOP
                WHERE validdate IN ({','.join(['?']*len(self._valid_times))})
            """
            LOG.info("Reading from SQLite file, this might take some time")
            df = pd.read_sql(
                query,
                self._conn,
                params=self._valid_times.astype("datetime64[s]").astype(np.int64).tolist(),
                index_col=["code","valid_time"],
                parse_dates={"valid_time": {"unit": "s"}}
            )

            data = df.to_xarray()

            #FIXME: do we need to go to xarray here?
            codes = self._codes.to_xarray()
            lon_values = codes["longitude"].sel(code=data["code"]).data
            lat_values = codes["latitude"].sel(code=data["code"]).data
            alt_values = codes["altitude"].sel(code=data["code"]).data
            
            
            data_coords = data.assign_coords(
                longitude=("code", lon_values),
                latitude=("code", lat_values),
                altitude=("code", alt_values)
            )
            self._data = data_coords

            if len(self._transformations) > 0:
                for transformation in self._transformations:
                    old_data = self._data
                    new_data = transformation.execute(old_data)
                    self._data = new_data

            return self._data

        else:
            return self._data

    def transform(self,transformation):
        if self._data is None:
            self._transformations.append(transformation)
            if isinstance(transformation, Renamer):
                old_keys = self._vars.copy().keys()
                for new_name, old_names in transformation.rename_dict.items():
                    for key in old_keys:
                        if key in old_names:
                            self._vars[new_name] = self._vars.pop(key)
        else:
            super().transform(transformation)
        


        










        