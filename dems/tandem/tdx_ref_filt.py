"""
@author: hugonnet
extract and filter all TanDEM-X tiles intersecting glaciers for HEM superior than 0.5 meter
reproject TanDEM-X tiles for each UTM zone, and enlarge to cover any 180km by 60km ASTER strips with centroid in the UTM zone
"""

import os, sys, shutil
dirf='/home/echos/hugonnet/code/devel/rh_pygeotools/'
sys.path.append(dirf)
import pandas as pd
import numpy as np
import operator
from demtileproducts import TDX_90m_tile
from shlib import make_pdirs
from misclib import SRTMGL1_naming_to_latlon, latlon_to_UTM, extended_factor_latitude_L1A, along_track_Terra

ww_dir = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/'
tdx_dir = '/data/echos/travail_en_cours/aster_tdem/data/DEM/TDX-90m/'

rgi_naming_txt = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/rgi_neighb_merged_naming_convention.txt'
text_file = open(rgi_naming_txt, 'r')
rgi_list = text_file.readlines()

for rgi in rgi_list:
    print('Working on rgi region: '+rgi)
    ref_cov_csv = os.path.join(ww_dir, rgi[:-1].split('rgi60')[0] + 'rgi60', 'cov','list_ref_DEM_tiles_'+rgi[:-1].split('rgi60')[0] + 'rgi60.csv')

    df=pd.read_csv(ref_cov_csv)
    tilelist=df['Tile_name'].tolist()

    ref_dir = os.path.join(ww_dir, rgi[:-1].split('rgi60')[0] + 'rgi60', 'ref')
    if os.path.exists(ref_dir):
        shutil.rmtree(ref_dir)
    make_pdirs(ref_dir)

    for tile in tilelist:
        try:
            print('Processing tile '+str(tilelist.index(tile)+1) + ' out of '+str(len(tilelist))+': '+tile)

            lat,lon = SRTMGL1_naming_to_latlon(tile)
            lat+=0.5
            lon+=0.5
            ext_lon, _ = extended_factor_latitude_L1A(lat,abs(along_track_Terra(lat)))

            list_utm = []
            list_epsg = []
            possible_lons = np.arange(lon-ext_lon,lon+ext_lon,0.1)
            for lons in possible_lons:
                if lons < -180:
                    lons += 360
                elif lons > 180:
                    lons -= 360
                epsg, utm = latlon_to_UTM(lat,lons)

                if utm not in list_utm:
                    list_utm.append(utm)
                    list_epsg.append(epsg)

            for utm in list_utm:
                utm_dir = os.path.join(ref_dir,utm)
                tgt_epsg = list_epsg[list_utm.index(utm)]
                print('Possible UTM zone: '+utm)

                make_pdirs(utm_dir)

                pref_dem_out = os.path.join(utm_dir,'TDX_90m_05hem')
                TDX_90m_tile(tdx_dir,tile,pref_dem_out,filter_params=[[operator.gt],[0.5],None],tgt_EPSG=int(tgt_epsg),tgt_res=[30,-30],nodata_out=-9999)
        except SystemExit:
            print('Tile '+str(tilelist.index(tile)+1) + ' out of '+str(len(tilelist))+': '+tile + ' does not exist')
