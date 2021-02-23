
import os, sys
import pandas as pd
import numpy as np
from pyddem.tdem_tools import aggregate_indep_regions, aggregate_all_to_period
# in_ext = '/home/atom/ongoing/work_worldwide/tables/table_man_gard_zemp_wout.csv'
# df_ext = pd.read_csv(in_ext)
fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'

reg_dir = '/home/atom/ongoing/work_worldwide/vol/final'
list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,16,17,18,19]] + [os.path.join(reg_dir,'dh_13_14_15_rgi60_int_base_reg.csv')]

tlim_srocc = [np.datetime64('2006-01-01'),np.datetime64('2016-01-01')]
tlim_full = [np.datetime64('2000-01-01'),np.datetime64('2020-01-01')]

list_df = []
for fn_reg in list_fn_reg:
    df_reg = pd.read_csv(fn_reg)
    df_srocc = aggregate_all_to_period(df_reg,[tlim_srocc],fn_tarea=fn_tarea,frac_area=1)
    df_srocc['comp'] = 'srocc'
    df_full = aggregate_all_to_period(df_reg, [tlim_full], fn_tarea=fn_tarea, frac_area=1)
    df_full['comp'] = 'full'
    df_tmp = pd.concat([df_srocc,df_full])
    list_df.append(df_tmp)

df = pd.concat(list_df)

df_s = df[df.comp=='srocc']
df_global_srocc = aggregate_indep_regions(df_s)
df_global_srocc['reg'] = 22
df_noperiph_srocc = aggregate_indep_regions(df_s[~df_s.reg.isin([5, 19])])
df_noperiph_srocc['reg'] = 23
df_a_srocc = aggregate_indep_regions(df_s[df_s.reg.isin([1,3,4,5,6,7,8,9])])
df_a_srocc['reg'] = 24
df_m_srocc = aggregate_indep_regions(df_s[df_s.reg.isin([1,2,6,8,10,11,12,21,16,17,18])])
df_m_srocc['reg'] = 25
df_srocc_total = pd.concat([df_s, df_global_srocc, df_noperiph_srocc,df_a_srocc,df_m_srocc])
df_srocc_total['comp'] = 'srocc'
df_srocc_total.period = df[df.comp=='srocc'].period.values[0]

df_f = df[df.comp=='full']
df_global_full = aggregate_indep_regions(df_f)
df_global_full['reg'] = 22
df_noperiph_full = aggregate_indep_regions(df_f[~df_f.reg.isin([5, 19])])
df_noperiph_full['reg'] = 23
df_a_full = aggregate_indep_regions(df_f[df_f.reg.isin([1,3,4,5,6,7,8,9])])
df_a_full['reg'] = 24
df_m_full = aggregate_indep_regions(df_f[df_f.reg.isin([1,2,6,8,10,11,12,21,16,17,18])])
df_m_full['reg'] = 25
df_full_total = pd.concat([df_f, df_global_full, df_noperiph_full,df_a_full,df_m_full])
df_full_total['comp'] = 'full'
df_full_total.period = df[df.comp=='full'].period.values[0]


df = pd.concat([df_srocc_total,df_full_total])

# df_tmp_acce = sum_regions(df_f[~df_f.reg.isin([5,6,8,19])])

df.to_csv('/home/atom/ongoing/work_worldwide/tables/final/Supp_Table_1_srocc.csv')