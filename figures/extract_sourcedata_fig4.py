import os
import pandas as pd
import numpy as np
import pyddem.tdem_tools as tt
from pyddem.vector_tools import latlon_to_SRTMGL1_naming, SRTMGL1_naming_to_latlon

in_csv_dh = '/home/atom/ongoing/work_worldwide/vol/final/dh_world_tiles_1deg.csv'
in_csv= '/home/atom/ongoing/work_worldwide/era5_analysis/final/mb_climate_1deg_change_20202010_20102000_all_700hpa_temp.csv'

df=pd.read_csv(in_csv)
df_dh = pd.read_csv(in_csv_dh)

tiles = [latlon_to_SRTMGL1_naming(df.lat.values[i]-0.5,(df.lon.values[i]+180)%360-180-0.5) for i in range(len(df))]
dhs = []
errs = []
areas = []
lats = []
lons = []
for tile in tiles:
    print('Working on tile '+tile)
    lat, lon = SRTMGL1_naming_to_latlon(tile)
    df_tile = df_dh[np.logical_and(df_dh.tile_latmin==lat,df_dh.tile_lonmin==lon)]
    lats.append(lat)
    lons.append(lon)
    dhs.append(df_tile[df_tile.period=='2010-01-01_2020-01-01'].dhdt.values[0]-df_tile[df_tile.period=='2000-01-01_2010-01-01'].dhdt.values[0])
    errs.append(np.sqrt(df_tile[df_tile.period=='2010-01-01_2020-01-01'].err_dhdt.values[0]**2+df_tile[df_tile.period=='2000-01-01_2010-01-01'].err_dhdt.values[0]**2))
    areas.append(df_tile.area.values[0]/1000000)

dps = df.d_P.tolist()
dts = df.d_T.tolist()

df_out = pd.DataFrame()
df_out = df_out.assign(tile_latmin=lats,tile_lonmin=lons,dh=dhs,err_dh=errs,area=areas,dp=dps,dt=dts)

df_out = df_out.round(4)

df_out.to_csv('/home/atom/ongoing/work_worldwide/tables/final/Source_Data_Fig4.csv',index=False)
