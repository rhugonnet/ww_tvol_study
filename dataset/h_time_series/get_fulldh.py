from __future__ import print_function
import pyddem.fit_tools as ft
import xarray as xr
import numpy as np
from glob import glob
import os
from pybob.bob_tools import mkdir_p
import pyddem.vector_tools as ot

dir_stacks = '/calcul/santo/hugonnet/worldwide/01_02_rgi60/'
list_fn_stacks = glob(os.path.join(dir_stacks,'**/*final.nc'),recursive=True)
name = 'Southwest_Alaska'

t0 = np.datetime64('2000-01-01')
t1 = np.datetime64('2020-10-01')

outpdir = '/data/icesat/travail_en_cours/romain/data_extracts'
outdir = os.path.join(outpdir,name+'_'+str(t0)+'_'+str(t1))

extent = [-152,-148,58,62]
list_tile = [os.path.basename(fn).split('_')[0] for fn in list_fn_stacks]
print(list_tile)
list_lat = np.array([ot.SRTMGL1_naming_to_latlon(tile)[0] for tile in list_tile])
list_lon = np.array([ot.SRTMGL1_naming_to_latlon(tile)[1] for tile in list_tile])

ind = np.logical_and.reduce((list_lon>=extent[0],list_lon<extent[1],list_lat>=extent[2],list_lat<extent[3]))

list_fn_stacks = [fn for fn in list_fn_stacks if ind[list_fn_stacks.index(fn)]]

list_tile = [os.path.basename(fn).split('_')[0] for fn in list_fn_stacks]
list_lat = np.array([ot.SRTMGL1_naming_to_latlon(tile)[0] for tile in list_tile])
list_lon = np.array([ot.SRTMGL1_naming_to_latlon(tile)[1] for tile in list_tile])
_, utm_zone = ot.latlon_to_UTM(np.mean(list_lat),np.mean(list_lon))

mkdir_p(outdir)

print('Extent:')
print(extent)
print('List of tiles falling in extent:')
print(list_tile)

for fn_stack in list_fn_stacks:

    print('Working on ' +fn_stack)
    tile_name= os.path.basename(fn_stack).split('_')[0]
    outname=os.path.join(outdir,tile_name)
    ds = xr.open_dataset(fn_stack)
    ft.get_full_dh(ds,t0,t1,outname)


list_dh = glob(os.path.join(outdir, '**/*_dh.tif'),recursive=True)
ft.reproj_build_vrt(list_dh, utm_zone, os.path.join(outdir, 'dh_'+name + '_' +str(t0) +'_'+ str(t1)+ '.vrt'))

list_err = glob(os.path.join(outdir, '**/*_err.tif'),recursive=True)
ft.reproj_build_vrt(list_err, utm_zone, os.path.join(outdir, 'err_'+name + '_' +str(t0) +'_'+ str(t1)+ '.vrt'))

for dh in list_dh:
    os.remove(dh)
for err in list_err:
    os.remove(err)