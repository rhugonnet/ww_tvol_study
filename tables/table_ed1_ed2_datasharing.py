
import os, sys
import numpy as np
import pandas as pd
import pyddem.tdem_tools as tt
from glob import glob

reg_dir = '/home/atom/ongoing/work_worldwide/vol/final'
list_fn_reg_multann = [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg_subperiods.csv') for i in np.arange(1,20)]
out_csv = '/home/atom/ongoing/work_worldwide/tables/final/ED_Table_1.csv'
df = pd.DataFrame()
for fn_reg_multann in list_fn_reg_multann:
    df= df.append(pd.read_csv(fn_reg_multann,index_col=0))

df_out = pd.DataFrame()

region_names = ['01 Alaska (ALA)','02 Western Canada & USA (WNA)','03 Arctic Canada North (ACN)'
    ,'04 Arctic Canada South (ACS)','05 Greenland (GRL)', '06 Iceland (ISL)','07 Svalbard and Jan Mayen (SJM)'
    , '08 Scandinavia (SCA)','09 Russian Arctic (RUA)', '10 North Asia (ASN)','11 Central Europe (CEU)'
    , '12 Caucasus and Middle East (CAU)', '13 Central Asia (ASC)','14 South Asia West (ASW)', '15 South Asia East (ASE)',
    '16 Low Latitudes (TRP)','17 Southern Andes (SAN)','18 New Zealand (NZL)','19 Antarctic and Subantarctic (ANT)', 'Total, excl. Greenland and Antarctic','Global total']

periods = ['2000-01-01_2005-01-01','2005-01-01_2010-01-01','2010-01-01_2015-01-01','2015-01-01_2020-01-01','2000-01-01_2020-01-01']
# periods = ['2000-01-01_2020-01-01']
# columns_names = ['2000-2005','2005-2010','2010-2015','2015-2020','2000-2020']
# column_names=['2000-2020']
list_df = []
df_out = pd.DataFrame()

# tmp for Niko
# periods = np.unique(df.period)

for period in periods:
    df_p = df[df.period==period]

    df_global = tt.aggregate_indep_regions_rates(df_p)
    df_global['reg']=21

    df_noperiph = tt.aggregate_indep_regions_rates(df_p[~df_p.reg.isin([5, 19])])
    df_noperiph['reg']=20

    df_full_p = pd.concat([df_p,df_noperiph,df_global])
    df_full_p['period'] = period

    list_df.append(df_full_p)

df_all = pd.concat(list_df)
df_all = df_all.sort_values(by=['reg','period'])

df_all = df_all.round(
    {'dhdt': 3, 'err_dhdt': 3, 'dvoldt': 0, 'err_dvoldt': 0, 'dmdt': 4, 'err_dmdt': 4, 'dmdtda': 3, 'err_dmdtda': 3,
     'perc_area_meas': 3, 'perc_area_res': 3,
     'valid_obs': 2, 'valid_obs_py': 2, 'area': 0, 'area_nodata': 0, 'tarea': 0})

# df_all = df_all[['reg','period','tarea','dhdt','err_dhdt','dmdt','err_dmdt','dmdtda','err_dmdtda']]
df_all.to_csv('/home/atom/ongoing/work_worldwide/tables/final/ed_table_1_and_2.csv',index=None)
