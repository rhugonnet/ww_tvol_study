"""
@author: hugonnet
compute elevation differences between ICESat and elevation time series, store parameters of interest
"""
from __future__ import print_function
import os
import pyddem.tdem_tools as tt
from glob import glob
import pandas as pd
import gdal

icesat_dir = '/data/icesat/travail_en_cours/romain/data/dems/icesat/'
world_data_dir = '/calcul/santo/hugonnet/worldwide/'
list_regions = os.listdir(world_data_dir)
# list_regions = ['09_rgi60']

shp_dir = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_neighb_merged'

for region in list_regions:

    print('Working on region: '+region)

    gla_mask = glob(os.path.join(shp_dir,region+'*',region+'*.shp'),recursive=True)[0]

    print('Using glacier mask: ' +gla_mask)

    if region == '05_rgi60':
        exc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/gis/Greenland_IceMask.shp'
    elif region == '19_rgi60':
        exc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/ais/ais_glacier_ice_mask.shp'
    else:
        exc_mask = None
    # elif region == '12_rgi60':
    #     exc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/reg12/12_rgi60_CaucasusMiddleEast.shp'
    # else:
    #     exc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'

    inc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/land/simplified_land_polygons.shp'

    if region == '01_02_rgi60':
        list_fn_AK = [os.path.join(icesat_dir,fn_icesat) for fn_icesat in os.listdir(icesat_dir) if 'ICESat_01' in fn_icesat]
        list_fn_WNA = [os.path.join(icesat_dir,fn_icesat) for fn_icesat in os.listdir(icesat_dir) if 'ICESat_02' in fn_icesat]
        list_fn_icesat_region = list_fn_AK + list_fn_WNA
    elif region == '13_14_15_rgi60':
        list_fn_CA = [os.path.join(icesat_dir,fn_icesat) for fn_icesat in os.listdir(icesat_dir) if 'ICESat_13' in fn_icesat]
        list_fn_SAW = [os.path.join(icesat_dir,fn_icesat) for fn_icesat in os.listdir(icesat_dir) if 'ICESat_14' in fn_icesat]
        list_fn_SAE = [os.path.join(icesat_dir,fn_icesat) for fn_icesat in os.listdir(icesat_dir) if 'ICESat_15' in fn_icesat]
        list_fn_icesat_region = list_fn_CA + list_fn_SAW + list_fn_SAE
    else:
        list_fn_icesat_region = [os.path.join(icesat_dir,fn_icesat) for fn_icesat in os.listdir(icesat_dir) if 'ICESat_'+region[0:2] in fn_icesat]

    dir_stack = os.path.join(world_data_dir,region,'stacks')
    list_fn_stack = glob(os.path.join(dir_stack,'**/*_final.nc'),recursive=True)

    for fn_icesat in list_fn_icesat_region:

        print('Working on region file: '+fn_icesat)

        outfile = '/data/icesat/travail_en_cours/romain/results/valid_' + os.path.splitext(os.path.basename(fn_icesat))[0] + '.csv'


        # for utm in os.listdir(dir_stack):
        #
        #     ref_dir = os.path.join(world_data_dir, region, 'ref')
        #     ref_utm_dir = os.path.join(ref_dir, utm)
        #     ref_vrt = os.path.join(ref_utm_dir, 'tmp_' + utm + '.vrt')
        #     ref_list = glob(os.path.join(ref_utm_dir, '**/*.tif'), recursive=True)
        #     if not os.path.exists(ref_vrt):
        #         gdal.BuildVRT(ref_vrt, ref_list, resampleAlg='bilinear')
        #
        #     shift, stats = tt.shift_icesat_stack(ref_vrt, fn_icesat, fn_shp)

        if os.path.exists(outfile):
            continue

        try:
            h, dh, zsc, dt, pos, slp, t, lon, lat, dh_ref, curv, dh_tot = tt.comp_stacks_icesat(list_fn_stack,fn_icesat,gla_mask=gla_mask,inc_mask=inc_mask,exc_mask=exc_mask,read_filt=True,nproc=64)
        except ValueError:
            continue
        df = pd.DataFrame()
        df = df.assign(h=h, dh=dh, zsc=zsc, dt=dt, pos=pos, slp=slp, t=t, lon=lon, lat=lat, dh_ref=dh_ref, curv=curv,dh_tot=dh_tot)

        df.to_csv(outfile)

print('Fin.')