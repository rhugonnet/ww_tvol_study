
import pyddem.fit_tools as ft
import xarray as xr
import numpy as np
from glob import glob
import os
import multiprocessing as mp
from pybob.bob_tools import mkdir_p

# example to extract DEMs (.tif) closest to a giveb date from the elevation time series (.nc)

# single file
# fn_stack='/calcul/santo/hugonnet/stacks/06_rgi60/27N/test2/N64W024_final.nc'
# list_fn_stack = [fn_stack]

# all files of a region
world_dir = '/calcul/santo/hugonnet/worldwide/'
# list_fn_stacks = glob(os.path.join(dir_stacks,'**/*final.nc'),recursive=True)

# date
list_t = [np.datetime64('2019-09-01')]

# output directory
out_pdir = '/data/icesat/travail_en_cours/romain/extracts/Amaury_AST-WV_h/'
nproc = 32

# wrapper for multi-processing
def wrapper_get_dem_stack(argsin):
    t0, fn_stack, out_dir, i, itot = argsin

    print('Working on stack: '+str(i+1)+' out of '+str(itot)+': ' +fn_stack)
    # extract tile name to write output file
    tile_name= os.path.basename(fn_stack).split('_')[0]
    outname=os.path.join(out_dir,tile_name)
    # open as xarray Dataset
    ds = xr.open_dataset(fn_stack)
    # integrate time series between the two time limits
    ft.get_dem_date_exact(ds,t0,outname)

# looping by region directory
list_regions = os.listdir(world_dir)

for region in list_regions:

    print('Working on region '+region)

    out_dir = os.path.join(out_pdir,region)
    # finding all stacks in the region
    list_fn_stacks = glob(os.path.join(world_dir,region,'stacks','**/*final.nc'), recursive=True)
    # all possibilities of stack and time period in the region
    list_tuples_region_tlims = [(list_t[i],list_fn_stacks[j]) for i in range(len(list_t)) for j in range(len(list_fn_stacks))]

    # run
    if nproc == 1:
        for tup in list_tuples_region_tlims:
            wrapper_get_dem_stack((tup[0], tup[1], out_dir,list_tuples_region_tlims.index(tup),len(list_tuples_region_tlims)))
    else:
        pool = mp.Pool(nproc)
        arg_dict = [(list_tuples_region_tlims[i][0],list_tuples_region_tlims[i][1],out_dir,i,len(list_tuples_region_tlims)) for i in range(len(list_tuples_region_tlims))]
        pool.map(wrapper_get_dem_stack, arg_dict, chunksize=1)
        pool.close()
        pool.join()


print('End')


