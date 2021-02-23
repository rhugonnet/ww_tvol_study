"""
@author: hugonnet
stack ASTER, ArcticDEM and REMA DEMs by 1x1° tiles + fit GP time series at monthly resolution
"""

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
from pyddem.stack_tools import create_mmaster_stack
import multiprocessing as mp
import gdal
import shutil
from glob import glob
import pyddem.fit_tools as ft
from pyddem.vector_tools import SRTMGL1_naming_to_latlon, latlon_to_UTM, niceextent_utm_latlontile
import pandas as pd
import numpy as np
import xarray as xr

region = '11_rgi60'
world_data_dir = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/'
world_calc_dir = '/calcul/santo/hugonnet/worldwide_30m/'
res = 30
y0 = 2000
nproc= 64
nfit = 1
skip_stacking = False
skip_fitting = False

ref_dir = os.path.join(world_data_dir,region,'ref')
tmp_ref_dir = os.path.join(world_calc_dir,region,'ref')

aster_dir = os.path.join('/data/icesat/travail_en_cours/romain/data/dems/aster_corr/',region)
tmp_aster_dir = os.path.join(world_calc_dir,region,'aster_corr')

#change for REMA here for region 19
setsm_dir = os.path.join('/data/icesat/travail_en_cours/romain/data/dems/arcticdem_corr',region)
tmp_setsm_dir = os.path.join(world_calc_dir,region,'arcticdem_corr')

out_dir = os.path.join(world_calc_dir,region,'stacks')

ref_gla_csv = os.path.join(world_data_dir,region,'cov','list_glacierized_tiles_'+region+'.csv')
df = pd.read_csv(ref_gla_csv)
tilelist = df['Tile_name'].tolist()

if not skip_stacking:

    print('Copying DEM data to '+os.path.join(world_calc_dir,region)+'...')
    if not os.path.exists(tmp_aster_dir):
        shutil.copytree(aster_dir,tmp_aster_dir)
    if not os.path.exists(tmp_setsm_dir) and os.path.exists(setsm_dir):
        shutil.copytree(setsm_dir,tmp_setsm_dir)
    if not os.path.exists(tmp_ref_dir):
        shutil.copytree(ref_dir,tmp_ref_dir)

    def stack_tile_wrapper(arg_dict):

        return stack_tile(**arg_dict)

    def stack_tile(tile,tmp_ref_dir,tmp_aster_dir,tmp_setsm_dir,out_dir):

        lat, lon = SRTMGL1_naming_to_latlon(tile)
        epsg, utm = latlon_to_UTM(lat, lon)

        outfile = os.path.join(out_dir, utm, tile + '.nc')

        if not os.path.exists(outfile):

            print('Stacking tile: ' + tile + ' in UTM zone ' + utm)

            # reference DEM
            ref_utm_dir = os.path.join(tmp_ref_dir, utm)
            ref_vrt = os.path.join(ref_utm_dir, 'tmp_' + utm + '.vrt')
            ref_list = glob(os.path.join(ref_utm_dir, '**/*.tif'), recursive=True)
            if not os.path.exists(ref_vrt):
                gdal.BuildVRT(ref_vrt, ref_list, resampleAlg='bilinear')

            # DEMs to stack
            flist1 = glob(os.path.join(tmp_aster_dir, '**/*_final.zip'), recursive=True)
            if os.path.exists(tmp_setsm_dir):
                flist2 = glob(os.path.join(tmp_setsm_dir,'**/*.tif'), recursive=True)
            else:
                flist2=[]

            flist = flist1 + flist2


            extent = niceextent_utm_latlontile(tile, utm, res)
            bobformat_extent = [extent[0], extent[2], extent[1], extent[3]]

            print('Nice extent is:')
            print(extent)
            if len(flist)>0:
                nco = create_mmaster_stack(flist, extent=bobformat_extent, epsg=int(epsg), mst_tiles=ref_vrt, res=res, outfile=outfile, coreg=False, uncert=True, clobber=True, add_ref=True, add_corr=True,latlontile_nodata=tile, filt_mm_corr=False, l1a_zipped=True ,y0=y0,tmptag=tile)
                nco.close()
            else:
                print('No DEM intersecting tile found. Skipping...')

        else:
            print('Tile '+tile+' already exists.')

    if nproc == 1:
        for tile in tilelist:
            stack_tile(tile,tmp_ref_dir,tmp_aster_dir,tmp_setsm_dir,out_dir)
    else:
        pool = mp.Pool(nproc,maxtasksperchild=1)
        arg_dict = [{'tile':tile,'tmp_ref_dir':tmp_ref_dir,'tmp_aster_dir':tmp_aster_dir,'tmp_setsm_dir':tmp_setsm_dir,'out_dir':out_dir} for tile in tilelist]
        pool.map(stack_tile_wrapper,arg_dict,chunksize=1)
        pool.close()
        pool.join()

    print('>>>Fin stack.')


def fit_tile_wrapper(arg_dict):

    return fit_tile(**arg_dict)

def fit_tile(tile,tmp_ref_dir,out_dir):

    method = 'gpr'
    # subspat = [383000,400000,5106200,5094000]
    subspat = None
    ref_dem_date = np.datetime64('2013-01-01')
    gla_mask = '/calcul/santo/hugonnet/outlines/rgi60_merge.shp'
    inc_mask = '/calcul/santo/hugonnet/outlines/rgi60_buff_10.shp'
    write_filt = True
    clobber = True
    tstep = 1./12.
    time_filt_thresh=[-50,50]
    opt_gpr = False
    kernel = None
    filt_ref = 'both'
    filt_ls = False
    conf_filt_ls = 0.99
    # specify the exact temporal extent needed to be able to merge neighbouring stacks properly
    tlim = [np.datetime64('2000-01-01'), np.datetime64('2020-01-01')]

    lat, lon = SRTMGL1_naming_to_latlon(tile)
    epsg, utm = latlon_to_UTM(lat, lon)
    print('Fitting tile: ' + tile + ' in UTM zone ' + utm)

    # reference DEM
    ref_utm_dir = os.path.join(tmp_ref_dir, utm)
    ref_vrt = os.path.join(ref_utm_dir, 'tmp_' + utm + '.vrt')
    infile = os.path.join(out_dir, utm, tile + '.nc')
    outfile = os.path.join(out_dir, utm, tile + '_final.nc')

    if True:#not os.path.exists(outfile):
        ft.fit_stack(infile, fit_extent=subspat, fn_ref_dem=ref_vrt, ref_dem_date=ref_dem_date, gla_mask=gla_mask,
                     tstep=tstep, tlim=tlim, inc_mask=inc_mask, filt_ref=filt_ref, time_filt_thresh=time_filt_thresh,
                     write_filt=True,
                     outfile=outfile, method=method, filt_ls=filt_ls, conf_filt_ls=conf_filt_ls, nproc=nproc,
                     clobber=True)

        # write dh/dts for visualisation
        ds = xr.open_dataset(outfile)

        t0 = np.datetime64('2000-01-01')
        t1 = np.datetime64('2020-01-01')

        ft.get_full_dh(ds, os.path.join(os.path.dirname(outfile), os.path.splitext(os.path.basename(outfile))[0]),t0=t0, t1=t1)

    else:
        print('Tile already processed.')

if skip_fitting:
    print('Skipping fitting... end.')
    sys.exit()

if nfit == 1:
    for tile in tilelist:
        fit_tile(tile,tmp_ref_dir,out_dir)
else:
    pool = mp.Pool(nfit)
    arg_dict = [{'tile':tile,'tmp_ref_dir':tmp_ref_dir,'out_dir':out_dir} for tile in tilelist]
    pool.map(fit_tile_wrapper,arg_dict,chunksize=1)
    pool.close()
    pool.join()

print('>>>Fin fit.')











