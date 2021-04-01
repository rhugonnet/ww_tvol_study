
import os
import pyddem.fit_tools as ft
import xarray as xr
import multiprocessing as mp
import numpy as np
from pybob.bob_tools import mkdir_p
from glob import glob

# example to elevation change maps (.tif) from the elevation time series (.nc)

# all files worldwide
world_dir = '/calcul/santo/hugonnet/worldwide'
out_pdir = '/data/icesat/travail_en_cours/romain/all_dhs'
# number of cores for processing
nproc = 32

# list of periods for which to derive elevation change
list_tlims = [(np.datetime64('2000-01-01'), np.datetime64('2020-01-01')),
              (np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
              (np.datetime64('2010-01-01'), np.datetime64('2020-01-01')),
              (np.datetime64('2000-01-01'), np.datetime64('2005-01-01')),
              (np.datetime64('2005-01-01'), np.datetime64('2010-01-01')),
              (np.datetime64('2010-01-01'), np.datetime64('2015-01-01')),
              (np.datetime64('2015-01-01'), np.datetime64('2020-01-01'))]


# wrapper for multi-processing
def wrapper_get_dh_stack(argsin):
    tlim, fn_stack, out_dir, i, itot = argsin

    out_dir_tlim = os.path.join(out_dir,str(tlim[0])+'_'+str(tlim[1]))

    print('Working on stack: '+str(i+1)+' out of '+str(itot)+': ' +fn_stack)
    # extract tile name to write output file
    tile_name= os.path.basename(fn_stack).split('_')[0]
    outname=os.path.join(out_dir_tlim,tile_name)
    # open as xarray Dataset
    ds = xr.open_dataset(fn_stack)
    # integrate time series between the two time limits
    ft.get_full_dh(ds,outname,t0=tlim[0],t1=tlim[1],rate=True)

# looping by region directory
list_regions = os.listdir(world_dir)

for region in list_regions:

    print('Working on region '+region)

    out_dir = os.path.join(out_pdir,region)
    # finding all stacks in the region
    list_fn_stacks = glob(os.path.join(world_dir,region, '**/*final.nc'), recursive=True)
    # all possibilities of stack and time period in the region
    list_tuples_region_tlims = [(list_tlims[i],list_fn_stacks[j]) for i in range(len(list_tlims)) for j in range(len(list_fn_stacks))]

    # run
    if nproc == 1:
        for tup in list_tuples_region_tlims:
            wrapper_get_dh_stack((tup[0], tup[1], out_dir))
    else:
        pool = mp.Pool(nproc)
        arg_dict = [(list_tuples_region_tlims[i][0],list_tuples_region_tlims[i][1],out_dir,i,len(list_tuples_region_tlims)) for i in range(len(list_tuples_region_tlims))]
        pool.map(wrapper_get_dh_stack, arg_dict, chunksize=1)
        pool.close()
        pool.join()
