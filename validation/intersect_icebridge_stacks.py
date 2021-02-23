"""
@author: hugonnet
compute elevation differences between IceBridge and elevation time series, store parameters of interest
"""

import os
import pyddem.tdem_tools as tt
from glob import glob
import pandas as pd
import gdal

icebridge_dir = '/calcul/santo/hugonnet/icebridge'
world_data_dir = '/calcul/santo/hugonnet/worldwide'
list_regions = os.listdir(world_data_dir)
# list_regions = ['09_rgi60']

shp_dir = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_neighb_merged'

for region in list_regions:

    dir_iodem = os.path.join(icebridge_dir,'iodem3_point',region)
    dir_ilaks1b = os.path.join(icebridge_dir,'ilaks1b_point',region)

    list_csv = []
    if os.path.exists(dir_iodem):
        fn_csv = glob(os.path.join(dir_iodem,'*.csv'),recursive=True)[0]
        list_csv.append(fn_csv)

    if os.path.exists(dir_ilaks1b):
        fn_csv = glob(os.path.join(dir_ilaks1b,'*.csv'),recursive=True)[0]
        list_csv.append(fn_csv)

    if len(list_csv)>0:

        print('Working on region: '+region)

        gla_mask = glob(os.path.join(shp_dir,region+'*',region+'*.shp'),recursive=True)[0]

        print('Using glacier mask: ' +gla_mask)

        if region == '05_rgi60':
            exc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/gis/Greenland_IceMask.shp'
        elif region == '19_rgi60':
            exc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/ais/ais_glacier_ice_mask.shp'
        else:
            exc_mask = None

        inc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/land/simplified_land_polygons.shp'

        dir_stack = os.path.join(world_data_dir,region,'stacks')
        list_fn_stack = glob(os.path.join(dir_stack,'**/*_final.nc'),recursive=True)

        for fn_csv in list_csv:

            print('Working on region file: '+fn_csv)

            outfile = '/data/icesat/travail_en_cours/romain/results/valid_' + os.path.splitext(os.path.basename(fn_csv))[0] + '.csv'

            if os.path.exists(outfile):
                continue
            try:
                h, dh, zsc, dt, pos, slp, t, lon, lat, dh_ref, curv, dh_tot = tt.comp_stacks_icebridge(list_fn_stack,fn_csv,gla_mask=gla_mask,inc_mask=inc_mask,exc_mask=exc_mask,read_filt=True,nproc=64)
            except ValueError:
                continue
            df = pd.DataFrame()
            df = df.assign(h=h, dh=dh, zsc=zsc, dt=dt, pos=pos, slp=slp, t=t, lon=lon, lat=lat, dh_ref=dh_ref, curv=curv,dh_tot=dh_tot)

            df.to_csv(outfile)

print('Fin.')