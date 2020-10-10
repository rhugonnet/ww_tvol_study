from __future__ import print_function
import os, sys
import pyddem.tdem_tools as tt
import pandas as pd
import numpy as np

dir_vol = '/data/icesat/travail_en_cours/romain/results/vol4/'
fn_csv_out = '/data/icesat/travail_en_cours/romain/results/df_pergla_global_10yr_20yr.csv'
list_fn_int_base=[os.path.join(dir_vol,'dh_'+str(i).zfill(2)+'_rgi60_int_base.csv') for i in np.arange(1,20)]

list_tlim = [[np.datetime64('2000-01-01'), np.datetime64('2010-01-01')],
              [np.datetime64('2010-01-01'), np.datetime64('2020-01-01')],
              [np.datetime64('2000-01-01'), np.datetime64('2020-01-01')]]

list_df = []
for fn_int_base in list_fn_int_base:
    print('Working on file: '+fn_int_base)
    for tlim in list_tlim:
        print('Working on period: '+str(tlim[0])+' to '+str(tlim[1]))
        df = tt.aggregate_df_int_time(fn_int_base,tlim=tlim,rate=True)
        list_df.append(df)

print('Concatenating and writing to file...')
df_tot = pd.concat(list_df)
df_tot = df_tot.sort_values(by=['rgiid'])
df_tot.to_csv(fn_csv_out, index=None, na_rep='NaN')


