"""
@author: hugonnet
extract volume changes from GP time series for the same glaciers and periods as high-res differences of DEMs
"""

import os, sys
import pandas as pd
import numpy as np

in_csv='/home/atom/data/other/Hugonnet_2020/dhdt_int_HR.csv'
# in_csv = '/data/icesat/travail_en_cours/romain/results/hr_dems/dhdt_int_HR.csv'
out_csv = '/data/icesat/travail_en_cours/romain/results/hr_dems/dhdt_int_GP.csv'
df = pd.read_csv(in_csv)


def aggregate_df_int_time(df,tlim=(np.datetime64('2000-01-01'),np.datetime64('2020-01-01'))):

    #not the cleanest closest time search you can write, but works
    times = sorted(list(set(list(df['time']))))
    df_time = pd.DataFrame()
    df_time = df_time.assign(time=times)
    df_time.index = pd.DatetimeIndex(pd.to_datetime(times))
    time_start = df_time.iloc[df_time.index.get_loc(pd.to_datetime(tlim[0]),method='nearest')][0]
    time_end = df_time.iloc[df_time.index.get_loc(pd.to_datetime(tlim[1]),method='nearest')][0]

    df_start = df[df.time == time_start]
    df_end = df[df.time == time_end]

    date_diff = (np.datetime64(time_end)-np.datetime64(time_start)).astype(int)/365.2524


    df_gla = pd.DataFrame()
    df_gla['rgiid'] = df_start['rgiid']
    df_gla['area'] = df_start['area']
    df_gla['perc_area_meas'] = df_start['perc_area_meas']
    df_gla['lat'] = df_start['lat']
    df_gla['lon'] = df_start['lon']
    df_gla['dh'] = df_end['dh'].values - df_start['dh'].values
    df_gla['err_dh'] = np.sqrt(df_end['err_dh'].values**2+df_start['err_dh'].values**2)
    df_gla['err_cont'] = df_start['perc_err_cont'] * df_start['area']

    df_gla['dhdt'] = df_gla['dh'] / date_diff
    df_gla['err_dhdt'] = df_gla['err_dh'] /date_diff

    #get the volume per glacier
    # df_gla['dvol'] = df_gla['dh'] * df_gla['area']
    #
    # #correct systematic seasonal biases (snow-covered terrain)
    # # df_tot['dh'] =
    #
    # #propagate error to volume change
    # df_gla['err_dvol'] = np.sqrt((df_gla['err_dh']*df_gla['area'])**2 + (df_gla['dh']*df_gla['err_cont'])**2)
    #
    # #convert to mass change (Huss, 2012)
    # df_gla['dm'] = df_gla['dvol'] * 0.85 / 10 ** 9
    #
    # #propagate error to mass change (Huss, 2012)
    # df_gla['err_dm'] = np.sqrt((df_gla['err_dvol'] * 0.85 / 10 ** 9) ** 2 + (
    #             df_gla['dvol'] * 0.06 / 10 ** 9) ** 2)

    return df_gla

vec_reg = []
for rgiid in list(df.rgiid):
    vec_reg.append(int(rgiid[6:8]))
df['reg'] = vec_reg

list_reg = list(set(list(df.reg)))
dir_csv = '/data/icesat/travail_en_cours/romain/results/vol4'
list_df_rgiid_int = []
for reg in list_reg:

    df_base = pd.read_csv(os.path.join(dir_csv,'dh_'+str(reg).zfill(2)+'_rgi60_int_base.csv'))

    list_rgiid = list(df[df.reg==reg].rgiid)

    for rgiid in list_rgiid:

        print('Working on '+rgiid)

        df_rgiid = df[df.rgiid==rgiid]
        df_base_rgiid = df_base[df_base.rgiid==rgiid]
        df_rgiid_int = aggregate_df_int_time(df_base_rgiid,tlim=(np.datetime64(df_rgiid.date_early.values[0]),np.datetime64(df_rgiid.date_late.values[0])))

        list_df_rgiid_int.append(df_rgiid_int)

df_out = pd.concat(list_df_rgiid_int)
df_out.to_csv(out_csv)
