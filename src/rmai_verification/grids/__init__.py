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
  uwcw=dict(
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
      lower_left=(-25.446747000000016, 39.639006000000016),
      upper_right=(40.151049804882405, 62.671198981050765),
      delta_x=2000.0,
      delta_y=2000.0,
      nx=1909,
      ny=1609
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