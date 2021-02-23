import os
import pandas as pd
import numpy as np
import pyddem.tdem_tools as tt

fn_csv = '/data/icesat/travail_en_cours/romain/results/vol_final/dh_19_rgi60_int_base.csv'
nproc=64
fn_tarea='/data/icesat/travail_en_cours/romain/data/outlines/rgi60/tarea_zemp.csv'
fn_out = '/data/icesat/travail_en_cours/romain/results/vol_final/subreg_antarctica.csv'

# tlim = [np.datetime64('2000-01-01'), np.datetime64('2020-01-01')]
# df = tt.aggregate_df_int_time(fn_csv,tlim=tlim,rate=True)
df = pd.read_csv(fn_csv)

exc_subantarctic = df.lat<-60
df = df[exc_subantarctic]

regional_area = 132867*10**6

df_west = df[df.lon < -85]
df_east = df[df.lon > -25]
df_peninsula = df[np.logical_and(df.lon<=-25,df.lon>=-85)]

list_df = [df_west,df_east,df_peninsula]
name_df = ['west','east','peninsula']

list_df_out = []
for i, df_tmp in enumerate(list_df):
    df_subreg_reg = tt.aggregate_int_to_all(df_tmp,nproc=nproc,get_corr_err=True)
    df_subreg_reg['time'] = df_subreg_reg.index.values
    subreg_area = df_subreg_reg.area.values[0]
    frac_area = subreg_area/regional_area
    list_df_mult = []
    for mult_ann in [1, 2, 4, 5, 10, 20]:
        df_mult = tt.aggregate_all_to_period(df_subreg_reg, mult_ann=mult_ann,fn_tarea=fn_tarea,frac_area=frac_area)
        list_df_mult.append(df_mult)
    df_subperiods = pd.concat(list_df_mult)
    df_subperiods['subreg'] = name_df[i]

    list_df_out.append(df_subperiods)

df_all = pd.concat(list_df_out)
df_all.to_csv(fn_out)


#
# mean_east = np.nansum(df_east.dhdt.values*df_east.area.values)/np.nansum(df_east.area.values)
# mean_west = np.nansum(df_west.dhdt.values*df_west.area.values)/np.nansum(df_west.area.values)
# mean_peninsula = np.nansum(df_peninsula.dhdt.values*df_peninsula.area.values)/np.nansum(df_peninsula.area.values)