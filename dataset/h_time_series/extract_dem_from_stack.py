
import pyddem.fit_tools as ft
import xarray as xr
import numpy as np
from glob import glob
import os
from pybob.bob_tools import mkdir_p

# example to extract DEMs (.tif) closest to a giveb date from the elevation time series (.nc)

# single file
# fn_stack='/calcul/santo/hugonnet/stacks/06_rgi60/27N/test2/N64W024_final.nc'
# list_fn_stack = [fn_stack]

# all files of a region
dir_stacks = '/calcul/santo/hugonnet/worldwide/13_14_15_rgi60'
list_fn_stacks = glob(os.path.join(dir_stacks,'**/*final.nc'),recursive=True)

# date
t = np.datetime64('2000-01-01')

# output directory
out_dir = '/data/icesat/travail_en_cours/romain/extracts/Amaury_HMA_'+str(t)+'/'
mkdir_p(out_dir)

for fn_stack in list_fn_stacks:

    print('Working on ' +fn_stack)

    tile_name= os.path.basename(fn_stack).split('_')[0]
    outname=os.path.join(out_dir,tile_name)
    ds = xr.open_dataset(fn_stack)
    ft.get_dem_date_exact(ds,t,outname)


