import pandas as pd
import numpy as np
from pyddem.vector_tools import latlon_to_SRTMGL1_naming, SRTMGL1_naming_to_latlon

in_csv = '/home/atom/ongoing/work_worldwide/vol/final/dh_world_tiles_1deg.csv'
period = '2000-01-01_2020-01-01'

group_by_spec = True
def latlon_to_2x2_tile_naming(lat, lon):

    lon_sw = np.floor(lon / 2) * 2
    lat_sw = np.floor(lat/2) * 2
    if lat_sw >= 0:
        str_lat = 'N'
    else:
        str_lat = 'S'
    if lon_sw >= 0:
        str_lon = 'E'
    else:
        str_lon = 'W'
    tile_name_tdx = str_lat + str(int(abs(lat_sw))).zfill(2) + str_lon + str(int(abs(lon_sw))).zfill(3)
    return tile_name_tdx

def latlon_to_spec_tile_naming(lat,lon):

    if np.abs(lat)>=74:
        lon_sw=np.floor((lon-0.5)/2)*2
        lat_sw = np.floor((lat-0.5)/2)*2
    elif np.abs(lat) >=60:
        lon_sw = np.floor((lon-0.5)/ 2) * 2
        lat_sw = lat
    else:
        lon_sw = lon
        lat_sw = lat

    if lat_sw>=0:
        str_lat='N'
    else:
        str_lat='S'
    if lon_sw>=0:
        str_lon='E'
    else:
        str_lon='W'
    tile_name_tdx= str_lat + str(int(abs(lat_sw))).zfill(2) + str_lon + str(int(abs(lon_sw))).zfill(3)

    return tile_name_tdx

def lat_to_latlontilesize(lat):

    if np.abs(lat)>=74:
        return 2, 2
    elif np.abs(lat) >=60:
        return 1, 2
    else:
        return 1,1

def latlon_to_spec_center(lat,lon):

    if np.abs(lat) >= 74:
        center_lon = lon + 1.5
        center_lat = lat + 1.5

    elif np.abs(lat) >= 60:
        center_lon = lon + 1.5
        center_lat = lat + 0.5

    else:
        center_lon = lon + 0.5
        center_lat = lat + 0.5


    return center_lat, center_lon

df_all = pd.read_csv(in_csv)

ind = np.logical_and.reduce((df_all.category=='all',df_all.period==period))
df_all = df_all[ind]
filt = np.logical_or.reduce((df_all.perc_area_meas<0.5,df_all.valid_obs<4))
df_all.loc[filt,'dhdt']=np.nan

# tiles = df_all.tile.tolist()
tiles = [latlon_to_SRTMGL1_naming(df_all.tile_latmin.values[i],df_all.tile_lonmin.values[i]) for i in range(len(df_all))]

areas = df_all.area.tolist()
dhs = df_all.dhdt.tolist()
errs = df_all.err_dhdt.tolist()

if group_by_spec:
    list_tile_grouped = []
    final_areas = []
    final_dhs = []
    final_errs = []
    for tile in tiles:
        lat, lon = SRTMGL1_naming_to_latlon(tile)
        list_tile_grouped.append(latlon_to_spec_tile_naming(lat,lon))
    final_tiles = list(set(list_tile_grouped))
    for final_t in final_tiles:
        orig_tiles = [orig_t for orig_t in tiles if list_tile_grouped[tiles.index(orig_t)] == final_t]

        group_areas = np.array([areas[tiles.index(orig_t)] for orig_t in orig_tiles])
        group_dhs = np.array([dhs[tiles.index(orig_t)] for orig_t in orig_tiles])
        group_errs = np.array([errs[tiles.index(orig_t)] for orig_t in orig_tiles])
        final_areas.append(np.nansum(group_areas))
        if np.count_nonzero(~np.isnan(group_dhs))!=0:
            final_dhs.append(np.nansum(group_dhs*group_areas)/np.nansum(group_areas))
        else:
            final_dhs.append(np.nan)
        final_errs.append(np.nansum(group_errs**2*group_areas**2)/np.nansum(group_areas)**2)
    tiles = final_tiles
    areas = final_areas
    dhs = final_dhs
    errs = final_errs

areas = [area/1000000 for _, area in sorted(zip(tiles,areas))]
dhs = [dh for _, dh in sorted(zip(tiles,dhs))]
errs = [err for _, err in sorted(zip(tiles,errs))]
tiles = sorted(tiles)
tile_latmin = [SRTMGL1_naming_to_latlon(tile)[0] for tile in tiles]
tile_lonmin = [SRTMGL1_naming_to_latlon(tile)[1] for tile in tiles]
tile_latsize = [lat_to_latlontilesize(lat)[0] for lat in tile_latmin]
tile_lonsize = [lat_to_latlontilesize(lat)[1] for lat in tile_latmin]

df = pd.DataFrame()
df = df.assign(tile_latmin=tile_latmin,tile_lonmin=tile_lonmin,tile_latsize=tile_latsize,tile_lonsize=tile_lonsize,dh=dhs,err_dh=errs,area=areas)

df = df.round(3)

df.to_csv('/home/atom/ongoing/work_worldwide/tables/final/Source_Data_Fig2.csv',index=False)