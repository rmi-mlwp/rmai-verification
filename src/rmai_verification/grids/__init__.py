import cartopy.crs as ccrs

# Predefined grids
GRIDS = dict(
  cerra=dict(
    projection="lcc",
    projection_kws=dict(
      globe=dict(
        semimajor_axis=6371229.0,
        semiminor_axis=6371229.0,
      ),
      central_longitude=8.0,
      central_latitude=50.0,
      standard_parallels=[50.0, 50.0],
    ),
    grid_kws=dict(
      lower_left=(-17.4859, 20.2923),
      upper_right=(74.1051, 63.7695),
      delta_x=5500.0,
      delta_y=5500.0,
      nx=1069,
      ny=1069
    ),
  ),
  alaro4km=dict(
    projection="lcc",
    projection_kws=dict(
      globe=dict(
        semimajor_axis=6371229.0,
        semiminor_axis=6371229.0,
      ),
      central_longitude=3.7,
      central_latitude=51.07,
      standard_parallels=[51.07, 51.07],
    ),
    grid_kws=dict(
      lower_left=(-6.5677443, 43.014187),
      upper_right=(17.888363, 57.87745),
      delta_x=4000.0,
      delta_y=4000.0,
      nx=421,
      ny=421
      
    ),
  ),
)

PROJECTIONS=dict(
    lcc=ccrs.LambertConformal,
    latlon=ccrs.PlateCarree,
    PlateCarree=ccrs.PlateCarree,
    Mercator=ccrs.Mercator,
    Orthographic=ccrs.Orthographic
)