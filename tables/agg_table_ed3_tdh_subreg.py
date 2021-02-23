"""
@author: hugonnet
aggregate results by subregion for Fig. 1
"""

import os, sys
import numpy as np
import pandas as pd
import gdal, ogr
from glob import glob
import pyddem.tdem_tools as tt
# regions_shp = '/home/atom/data/inventory_products/RGI/00_rgi60/00_rgi60_regions/regions_split_adaptedHMA.shp'
regions_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_O1Regions.shp'

# outfile = '/home/atom/ongoing/work_worldwide/vol/aggreg_subregions.csv'
outfile = '/data/icesat/travail_en_cours/romain/results/vol_final/subreg_tidewater_regs.csv'
nproc=64

# main_dir = '/home/atom/proj/ww_tvol_study/worldwide/'
main_dir = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide'
rgi_naming_txt=os.path.join(main_dir,'rgi_neighb_merged_naming_convention.txt')

# int_dir = '/home/atom/ongoing/work_worldwide/vol/'
int_dir = '/data/icesat/travail_en_cours/romain/results/vol_final'
text_file = open(rgi_naming_txt, 'r')
rgi_list = text_file.readlines()
int_list = [os.path.join(int_dir,'dh_'+rgi[:-1].split('rgi60')[0] + 'rgi60'+'_int_base.csv') for rgi in rgi_list]
# int_list = [os.path.join(int_dir,'dh_10_rgi60_int_base.csv')]
# fn_base='/home/atom/data/inventory_products/RGI/base_rgi.csv'
fn_base = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/base_rgi.csv'
df_base = pd.read_csv(fn_base)
# fn_tarea='/home/atom/data/inventory_products/RGI/tarea_zemp.csv'
fn_tarea = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/tarea_zemp.csv'

ds_shp_in = ogr.GetDriverByName('ESRI Shapefile').Open(regions_shp, 0)
layer_in = ds_shp_in.GetLayer()

# int_list = [int_list[0]]

list_region_codes = [1,3,4,5,7,9,17,19]

df_all = pd.DataFrame()

list_df_all = []
for feature in layer_in:
    geom = feature.GetGeometryRef()
    region = feature.GetField('FULL_NAME')
    centroid = geom.Centroid()
    center_lon, center_lat, _ = centroid.GetPoint()
    rgi_code = feature.GetField('RGI_CODE')

    print('Working on region: '+region+' with code '+ str(rgi_code))

    if not rgi_code in list_region_codes:
        continue

    for fn_int in int_list:

        if str(rgi_code).zfill(2) in os.path.basename(fn_int):

            print('Working on region ' +region + ' with file '+os.path.basename(fn_int))
            df = pd.read_csv(fn_int)

            df_area = df.groupby('time')['area'].sum()
            regional_area = df_area.values[0]

            # list_rgiid = list(set(list(df.rgiid)))

            # keep only points in polygon
            df_rgiid = df.groupby('rgiid')['lat','lon'].mean()
            df_rgiid['rgiid'] = df_rgiid.index

            list_subreg_rgiid = []
            for i in np.arange(len(df_rgiid)):
                print('Working on glacier '+str(i+1)+' out of '+str(len(df_rgiid)))
                lat = df_rgiid.lat.values[i]
                lon = df_rgiid.lon.values[i]

                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(lon, lat)

                if point.Intersects(geom):
                    list_subreg_rgiid.append(df_rgiid.rgiid.values[i])

            ind_tw = np.logical_or(df_base.term == 1,df_base.term==5)
            keep_tw = list(df_base.rgiid[ind_tw])
            keep_ntw = list(df_base.rgiid[~ind_tw])
            list_keeps = [keep_tw, keep_ntw, list(df_base.rgiid)]
            name_keeps = ['tw', 'ntw', 'all']

            for keeps in list_keeps:

                subset = [rgiid for rgiid in list_subreg_rgiid if rgiid in keeps]

                df_subreg_int = df[df.rgiid.isin(subset)]

                if len(df_subreg_int)==0:
                    continue

                df_subreg_reg = tt.aggregate_int_to_all(df_subreg_int,get_corr_err=True,nproc=nproc)
                df_subreg_reg['time'] = df_subreg_reg.index.values

                if len(df_subreg_reg) == 0:
                    continue

                subreg_area = df_subreg_reg.area.values[0]
                frac_area = subreg_area/regional_area

                list_df_mult = []
                for mult_ann in [1, 2, 4, 5, 10, 20]:
                    df_mult = tt.aggregate_all_to_period(df_subreg_reg, mult_ann=mult_ann,fn_tarea=fn_tarea,frac_area=frac_area)
                    list_df_mult.append(df_mult)

                df_mult_all = pd.concat(list_df_mult)
                df_mult_all['subreg'] = region
                df_mult_all['lon_reg'] = center_lon
                df_mult_all['lat_reg'] = center_lat
                df_mult_all['category'] = name_keeps[list_keeps.index(keeps)]
                df_mult_all['frac_area'] = frac_area

                list_df_all.append(df_mult_all)

df_all = pd.concat(list_df_all)
df_all.to_csv(outfile)