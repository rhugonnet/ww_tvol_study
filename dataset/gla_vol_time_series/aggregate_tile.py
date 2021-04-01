import os
import pyddem.tdem_tools as tt
import numpy as np

# example to aggregate the per-glacier estimates over a tiling

int_dir = '/data/icesat/travail_en_cours/romain/results/vol_final'

# here we input only the estimates from Alaska, so the tiling will only use those; we can input a list with any desired estimates
list_fn_int = [os.path.join(int_dir,'dh_01_rgi60_int_base.csv')]
# file containing RGI 6.0 (updated for this study) metadata
fn_base = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/base_rgi.csv'
# number of processing cores
nproc=32
# we want the cumulative series to be integrate into rates for those specific periods
list_tlim = [(np.datetime64('2002-01-01'),np.datetime64('2020-01-01')),(np.datetime64('2002-01-01'),np.datetime64('2008-01-01'))
    ,(np.datetime64('2008-01-01'),np.datetime64('2014-01-01')),(np.datetime64('2014-01-01'),np.datetime64('2020-01-01'))]


# 1°x1° tiling (by default sort_tw = sort tidewater/non-tidewater glaciers is False)
print('Working on 1x1 tiling...')
fn_out_1 = '/data/icesat/travail_en_cours/romain/extracts/Gulf_of_Alaska_tile_1x1.csv'
tt.df_all_base_to_tile(list_fn_int,fn_base,list_tlim=list_tlim,fn_out=fn_out_1,tile_size=1,nproc=nproc)

# 0.5°x0.5° tiling
print('Working on 0.5x0.5 tiling...')
fn_out_05 = '/data/icesat/travail_en_cours/romain/extracts/Gulf_of_Alaska_tile_0_5x0_5.csv'
tt.df_all_base_to_tile(list_fn_int,fn_base,list_tlim=list_tlim,fn_out=fn_out_05,tile_size=0.5,nproc=nproc)

print('End.')
