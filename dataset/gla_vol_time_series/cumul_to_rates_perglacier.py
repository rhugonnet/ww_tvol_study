
import os, sys
import pyddem.tdem_tools as tt
import pandas as pd
import numpy as np

# example to integrate per-glacier cumulative time series into rates for specific periods

# listing per-glacier estimates files for all regions
dir_vol = '/data/icesat/travail_en_cours/romain/results/vol_final/'
fn_csv_out = '/data/icesat/travail_en_cours/romain/results/df_pergla_global_10yr_20yr.csv'
list_fn_int_base=[os.path.join(dir_vol,'dh_'+str(i).zfill(2)+'_rgi60_int_base.csv') for i in np.arange(1,20)]

# defining all possible successive 1-, 2-, 4-, 5-, 10- and 20-year time periods in 2000-2019 to derive rates
list_tlim = []
for mult_ann in [1,2,4,5,10,20]:
    nb_period = int(np.floor(20 / mult_ann))
    list_tlim += [
        (np.datetime64(str(2000 + mult_ann * i) + '-01-01'), np.datetime64(str(2000 + mult_ann * (i + 1)) + '-01-01')) for i
        in range(nb_period)]

# integrate into rates for each glacier of each region, and each possible time period
list_df = []
for fn_int_base in list_fn_int_base:
    print('Working on file: '+fn_int_base)
    for tlim in list_tlim:
        print('Working on period: '+str(tlim[0])+' to '+str(tlim[1]))
        df = tt.aggregate_df_int_time(fn_int_base,tlim=tlim,rate=True)
        list_df.append(df)

# concatenate, sort, and write to file
print('Concatenating and writing to file...')
df_tot = pd.concat(list_df)
df_tot = df_tot.sort_values(by=['rgiid'])
df_tot.to_csv(fn_csv_out, index=None, na_rep='NaN')


