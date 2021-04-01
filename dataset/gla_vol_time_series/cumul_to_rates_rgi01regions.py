import os, sys
import pandas as pd
import numpy as np
import pyddem.tdem_tools as tt

# example to integrate the RGI-O1 cumulative time series into rates with time-varying glacier areas

# file with time-varying areas for RGI regions
fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'

# list of regional cumulative series
reg_dir = '/home/atom/ongoing/work_worldwide/vol/final'
list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
# periods of interest
tlim_00_10 = [np.datetime64('2000-01-01'),np.datetime64('2010-01-01')]
tlim_10_20 = [np.datetime64('2010-01-01'),np.datetime64('2020-01-01')]
tlim_ar6 = [np.datetime64('2006-01-01'),np.datetime64('2019-01-01')]
tlim_00_20 = [np.datetime64('2000-01-01'),np.datetime64('2020-01-01')]
list_tlim = [tlim_00_10,tlim_10_20,tlim_ar6,tlim_00_20]
list_tag = ['decad1','decad2','ar6','full']

# integrate the cumulative series into rates for each period and region
list_df = []
for fn_reg in list_fn_reg:
    df_reg = pd.read_csv(fn_reg)
    df_agg = tt.aggregate_all_to_period(df_reg,list_tlim=list_tlim,fn_tarea=fn_tarea,frac_area=1,list_tag=list_tag)
    list_df.append(df_agg)

# concatenate results
df = pd.concat(list_df)

# convert m w.e. yr-1 into kg m-2 yr-1
df.dmdtda *= 1000
df.err_dmdtda *= 1000

# keep only variables of interests
df = df[['reg','period','dmdt','err_dmdt','dmdtda','err_dmdtda']]
df.to_csv('/home/atom/ongoing/work_ipcc_ar6/table_hugonnet_regions_10yr_20yr_ar6period_final.csv',float_format='%.2f',index=None)
