from __future__ import print_function
import pyddem.fit_tools as ft
import xarray as xr
import numpy as np
from glob import glob
import os
from pybob.bob_tools import mkdir_p
# fn_stack='/calcul/santo/hugonnet/stacks/06_rgi60/27N/test2/N64W024_final.nc'

dir_stacks = '/calcul/santo/hugonnet/bbc_npispi/'
list_fn_stacks = glob(os.path.join(dir_stacks,'**/*final.nc'),recursive=True)
t = np.datetime64('2000-01-01')

out_dir = '/data/icesat/travail_en_cours/romain/extracts/ASTER_30m_npispi_'+str(t)+'/'
mkdir_p(out_dir)

for fn_stack in list_fn_stacks:

    print('Working on ' +fn_stack)
    #middle of SRTM date

    tile_name= os.path.basename(fn_stack).split('_')[0]
    outname=os.path.join(out_dir,tile_name)
    ds = xr.open_dataset(fn_stack)
    # ds_filt = xr.open_dataset(os.path.join(os.path.dirname(fn_stack),tile_name+'_filtered.nc'))

    ft.get_dem_date_exact(ds,t,outname)


