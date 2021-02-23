"""
@author: hugonnet
derive values from this study for Table 1 (values for AIS/GIS are from IMBIE)
"""

import os, sys
import numpy as np
import pandas as pd
from glob import glob
import pyddem.fit_tools as ft
import pyddem.tdem_tools as tt

reg_dir = '/home/atom/ongoing/work_worldwide/vol/reg'
fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'
list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]

periods = ['2000-01-01_2005-01-01','2005-01-01_2010-01-01','2010-01-01_2015-01-01','2015-01-01_2019-01-01','2003-01-01_2019-01-01']
tlims = [(np.datetime64('2000-01-01'),np.datetime64('2005-01-01')),(np.datetime64('2005-01-01'),np.datetime64('2010-01-01'))
    ,(np.datetime64('2010-01-01'),np.datetime64('2015-01-01')),(np.datetime64('2015-01-01'),np.datetime64('2019-01-01')),
         (np.datetime64('2003-01-01'),np.datetime64('2019-01-01'))]


list_df = []
for fn_reg in list_fn_reg:
    for period in periods:
        df_tmp = tt.aggregate_all_to_period(pd.read_csv(fn_reg),[tlims[periods.index(period)]],fn_tarea=fn_tarea,frac_area=1)

        list_df.append(df_tmp)
df = pd.concat(list_df)

list_df_all = []
for period in periods:

    df_p = df[df.period == period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['reg']='global'
    df_global['period'] = period

    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['reg']='global_noperiph'
    df_noperiph['period'] =period

    df_full_p = pd.concat([df_p,df_noperiph,df_global])

    list_df_all.append(df_full_p)

df_all = pd.concat(list_df_all)
df_g = df_all[df_all.reg=='global']
df_05 = df_all[df_all.reg==5]
df_19 = df_all[df_all.reg==19]

