from __future__ import print_function
import pyddem.fit_tools as ft
import xarray as xr
import numpy as np
from glob import glob
import os
from pybob.bob_tools import mkdir_p
# fn_stack='/calcul/santo/hugonnet/stacks/06_rgi60/27N/test2/N64W024_final.nc'

region = '11_rgi60'
dir_stacks = os.path.join('/calcul/santo/hugonnet/worldwide/',region)
list_fn_stacks = glob(os.path.join(dir_stacks,'**/*final.nc'),recursive=True)

out_dir = os.path.join('/data/icesat/produits/Global_DEM/AST_SRTM/',region)
mkdir_p(out_dir)

for fn_stack in list_fn_stacks:

    print('Working on ' +fn_stack)
    #middle of SRTM date
    t = np.datetime64('2000-02-15')

    tile_name= os.path.basename(fn_stack).split('_')[0]
    outname=os.path.join(out_dir,tile_name)
    ds = xr.open_dataset(fn_stack)
    ds_filt = xr.open_dataset(os.path.join(os.path.dirname(fn_stack),tile_name+'_filtered.nc'))

    ft.get_dem_date(ds,ds_filt,t,outname)


