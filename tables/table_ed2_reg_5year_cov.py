
import os, sys
import numpy as np
import pandas as pd
import pyddem.tdem_tools as tt
from glob import glob

reg_dir = '/home/atom/ongoing/work_worldwide/vol/final'
list_fn_reg_multann = [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg_subperiods.csv') for i in np.arange(1,20)]
out_csv = '/home/atom/ongoing/work_worldwide/tables/final/ED_Table_2.csv'
df = pd.DataFrame()
for fn_reg_multann in list_fn_reg_multann:
    df= df.append(pd.read_csv(fn_reg_multann))

df_out = pd.DataFrame()

region_names = ['01 Alaska (ALA)','02 Western Canada & USA (WNA)','03 Arctic Canada North (ACN)'
    ,'04 Arctic Canada South (ACS)','05 Greenland (GRL)', '06 Iceland (ISL)','07 Svalbard and Jan Mayen (SJM)'
    , '08 Scandinavia (SCA)','09 Russian Arctic (RUA)', '10 North Asia (ASN)','11 Central Europe (CEU)'
    , '12 Caucasus and Middle East (CAU)', '13 Central Asia (ASC)','14 South Asia West (ASW)', '15 South Asia East (ASE)',
    '16 Low Latitudes (TRP)','17 Southern Andes (SAN)','18 New Zealand (NZL)','19 Antarctic and Subantarctic (ANT)', 'Total, excl. Greenland and Antarctic','Global total']

periods = ['2000-01-01_2005-01-01','2005-01-01_2010-01-01','2010-01-01_2015-01-01','2015-01-01_2020-01-01','2000-01-01_2020-01-01']
# periods = ['2000-01-01_2020-01-01']
columns_names = ['2000-2004','2005-2009','2010-2014','2015-2019','2000-2019']
# column_names=['2000-2020']
list_dh = []
list_dh_err = []
list_dm = []
list_dm_err =[]
df_out = pd.DataFrame()
for period in periods:
    df_p = df[df.period==period]

    df_global = tt.aggregate_indep_regions_rates(df_p)
    df_global['reg']='global'

    df_noperiph = tt.aggregate_indep_regions_rates(df_p[~df_p.reg.isin([5, 19])])
    df_noperiph['reg']='global_noperiph'

    df_full_p = pd.concat([df_p,df_noperiph,df_global])

    column_dh = []
    for i in range(len(df_full_p)):
        dh = '{:.1f}'.format(df_full_p.valid_obs.values[i])
        column_dh.append(dh)

    list_dh.append(column_dh)


column_str_dh = []
for j in range(len(list_dh[0])):
    list_str_dh = []
    for i in range(4):
        list_str_dh.append(list_dh[i][j])
    final_str_dh = '/'.join(list_str_dh)
    column_str_dh.append(final_str_dh)

df_out['valid_obs_sub'] = column_str_dh
df_out['valid_obs_tot'] = list_dh[4]

perc_area_nodata = ['{:.2f}'.format(df_full_p.area_nodata.values[i]/df_full_p.area.values[i]*100) for i in range(len(df_full_p))]
perc_area_res = ['{:.2f}'.format(df_full_p.perc_area_res.values[i]*100) for i in range(len(df_full_p))]

df_out.insert(loc=0,column='region',value=region_names)
df_out.insert(loc=1,column='glacier area not covered',value=perc_area_nodata)
df_out.insert(loc=2,column='spatial coverage',value=perc_area_res)
df_out.to_csv(out_csv)


