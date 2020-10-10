from __future__ import print_function
import os
import pyddem.tdem_tools as tt
from glob import glob
import numpy as np

world_data_dir = '/calcul/santo/hugonnet/worldwide/'
list_regions = os.listdir(world_data_dir)
results_dir = '/data/icesat/travail_en_cours/romain/results/vol4'
# results_dir = '/home/atom/ongoing/work_worldwide/vol/vol4'
feat_id='RGIId'
tlim = None
nproc=64
dir_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_neighb_renamed'
# fn_base = '/home/atom/data/inventory_products/RGI/base_rgi.csv'
# fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'

fn_base = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/base_rgi.csv'
fn_tarea='/data/icesat/travail_en_cours/romain/data/outlines/rgi60/tarea_zemp.csv'

# list_regions = ['01_02_rgi60','03_rgi60','04_rgi60','05_rgi60','06_rgi60','07_rgi60','08_rgi60','09_rgi60','10_rgi60','11_rgi60','12_rgi60','13_14_15_rgi60','16_rgi60','17_rgi60','18_rgi60','19_rgi60']
# list_regions = ['06_rgi60']

for region in list_regions:

    print('Working on region: '+region)

    #integrate glaciers globally
    fn_shp = glob(os.path.join(dir_shp,'**/*'+region+'*.shp'),recursive=True)[0]
    dir_stack = os.path.join(world_data_dir, region, 'stacks')
    list_fn_stack = glob(os.path.join(dir_stack, '**/*_final.nc'), recursive=True)
    outfile = os.path.join(results_dir,'dh_'+region+'.csv')
    tt.hypsocheat_postproc_stacks_tvol(list_fn_stack,fn_shp,nproc=nproc,outfile=outfile)

    # add info from base glacier dataframe (missing glaciers, identify region, etc...)
    infile = os.path.join(results_dir,'dh_'+region+'_int.csv')
    tt.df_int_to_base(infile, fn_base=fn_base)

list_fn_base = glob(os.path.join(results_dir, 'dh_*_base.csv'))
print(list_fn_base)
list_fn_base = [fn_base for fn_base in list_fn_base if '13_14_15_rgi60' not in fn_base]
for fn_base in list_fn_base:

    # aggregate by region
    infile = fn_base
    infile_reg = os.path.join(os.path.dirname(fn_base),os.path.splitext(os.path.basename(fn_base))[0]+'_reg.csv')
    if not os.path.exists(infile_reg):
        tt.df_int_to_reg(infile,nproc=nproc)

    # aggregate by periods
    tt.df_region_to_periods(infile_reg,fn_tarea=fn_tarea)

#aggregate worldwide by tile
list_fn_base = glob(os.path.join(results_dir,'dh_*_base.csv'))
print(list_fn_base)
tt.df_all_base_to_tile(list_fn_base,fn_base,tile_size=1,nproc=nproc)
tt.df_all_base_to_tile(list_fn_base,fn_base,tile_size=0.25,nproc=nproc)
