"""
@author: hugonnet
sort by UTM zone and co-register to TanDEM-X all ArcticDEM enlarged strips
"""
from __future__ import print_function
import os, sys, shutil
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
import pandas as pd
import numpy as np
from pybob.coreg_tools import dem_coregistration
from pyddem.vector_tools import SRTMGL1_naming_to_latlon, latlon_to_UTM
from pymmaster.mmaster_tools import clean_coreg_dir
from glob import glob
import multiprocessing as mp
import gdal
from pybob.GeoImg import GeoImg

rgi_naming_txt = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/rgi_neighb_merged_naming_convention.txt'
main_dir = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide'
setsm_dir = '/data/icesat/travail_en_cours/romain/data/dems/arcticdem'
out_dir = '/data/icesat/travail_en_cours/romain/data/dems/arcticdem_corr'
fn_gla_excl_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'
fn_gla_incl_mask = None
proc_ref_dir = '/calcul/santo/hugonnet/tandem2/'
ref_dir = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/'
fn_gis_excl_mask = '/data/icesat/travail_en_cours/romain/data/outlines/gis/Greenland_IceMask.shp'
fn_gis_incl_mask = '/data/icesat/travail_en_cours/romain/data/outlines/gis/Greenland_Landmask.shp'

text_file = open(rgi_naming_txt, 'r')
rgi_list = text_file.readlines()

#removing already processed regions
# rgi_list = rgi_list[4:]
rgi_list = ['06_rgi60']

tile_list_csv = [os.path.join(main_dir, rgi[:-1].split('rgi60')[0] + 'rgi60', 'cov', 'list_glacierized_tiles_' + rgi[:-1].split('rgi60')[0] + 'rgi60' + '.csv') for rgi in rgi_list]
nproc = 16

def coreg_wrapper(argsin):

    ref_vrt, in_dem, fn_excl_mask, fn_incl_mask, strip_out_dir = argsin

    print('Coregistering strip: ' + in_dem)

    if not os.path.exists(strip_out_dir):

        try:
            # _, outslave, _, stats = dem_coregistration(ref_vrt, in_dem, glaciermask=fn_excl_mask, landmask=fn_incl_mask,
            #                                            outdir=strip_out_dir, inmem=True)
            # rmse = stats[3]
            # clean_coreg_dir(strip_out_dir, '.')
            # if rmse < 10:
            #     outslave.write(os.path.basename(strip_out_dir) + '_adj.tif', out_folder=strip_out_dir)
            _, _, shift_params, stats = dem_coregistration(ref_vrt, in_dem, glaciermask=fn_excl_mask, landmask=fn_incl_mask,
                                                       outdir=strip_out_dir, inmem=True)
            rmse = stats[3]
            clean_coreg_dir(strip_out_dir, '.')
            orig_slv = GeoImg(in_dem)
            if rmse < 10:
                orig_slv.shift(shift_params[0], shift_params[1])
                orig_slv.img = orig_slv.img + shift_params[2]
                orig_slv.write(os.path.basename(strip_out_dir)+ '_adj.tif', out_folder=strip_out_dir)
                # outslave.write(os.path.basename(strip_out_dir) + '_adj.tif', out_folder=strip_out_dir)
        except Exception:
            clean_coreg_dir(strip_out_dir, '.')

    else:
        print('Output dir already exists, skipping...')


for rgi_csv in tile_list_csv:

    rgi = rgi_list[tile_list_csv.index(rgi_csv)]
    rgi = rgi[:-1].split('rgi60')[0]+'rgi60'

    if rgi == '05_rgi60':
        fn_excl_mask = fn_gis_excl_mask
        fn_incl_mask = fn_gis_incl_mask
    else:
        fn_excl_mask = fn_gla_excl_mask
        fn_incl_mask = fn_gla_incl_mask

    print('Working on rgi: '+rgi)

    df = pd.read_csv(rgi_csv)
    tiles = df['Tile_name'].tolist()

    list_in_dem = []
    list_strip_out_dir = []
    list_ref_vrt = []
    for tile in tiles:

        setsm_tile = os.path.join(setsm_dir, 'processed_' + tile.lower())

        if os.path.exists(setsm_tile):

            print('Found corresponding tile:' +tile.lower())

            strips = [strip for strip in os.listdir(setsm_tile) if strip.endswith('.tif')]

            lat, lon = SRTMGL1_naming_to_latlon(tile)
            epsg, utm = latlon_to_UTM(lat, lon)

            # create reference DEM VRT if does not exist
            ref_utm_dir = os.path.join(ref_dir,rgi,'ref',utm)
            proc_ref_utm_dir = os.path.join(proc_ref_dir,rgi,'ref',utm)
            ref_vrt = os.path.join(proc_ref_utm_dir, 'tmp_' + utm + '.vrt')

            if not os.path.exists(proc_ref_utm_dir):
                shutil.copytree(ref_utm_dir, proc_ref_utm_dir)

            if not os.path.exists(ref_vrt):
                ref_list = glob(os.path.join(proc_ref_utm_dir, '**/*.tif'), recursive=True)
                gdal.BuildVRT(ref_vrt, ref_list, resampleAlg='bilinear')

            for strip in strips:
                in_dem = os.path.join(setsm_tile, strip)
                strip_out_dir = os.path.join(out_dir, rgi, utm, os.path.splitext(strip)[0])

                list_in_dem.append(in_dem)
                list_strip_out_dir.append(strip_out_dir)
                list_ref_vrt.append(ref_vrt)


    print('Co-registering '+str(len(list_ref_vrt))+' SETSM strips with '+str(nproc)+' processors...')
    argsin = [(list_ref_vrt[i], list_in_dem[i], fn_excl_mask, fn_incl_mask, list_strip_out_dir[i]) for i in range(len(list_ref_vrt))]
    pool = mp.Pool(nproc,maxtasksperchild=1)
    pool.map(coreg_wrapper,argsin,chunksize=1)
    pool.close()
    pool.join()













