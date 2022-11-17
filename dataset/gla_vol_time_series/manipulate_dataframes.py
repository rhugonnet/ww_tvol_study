# Short examples to show how to manipulate the datasets

import numpy as np
import pandas as pd

df = pd.read_csv('/home/atom/ongoing/work_Rounce/df_pergla_global_10yr_20yr.csv')

# look at the glaciers larger than 10 km² with the highest thinning rates of 2000-2019
df = df[np.logical_and(df.period=='2010-01-01_2020-01-01',df.area>10.)]
df_tmp = df.sort_values(by='dhdt')

# show the first 10
print(df_tmp[['rgiid','area','dhdt','perc_area_meas']].iloc[0:10,:])

#let's remove Alaska, Southern Andes and Antarctic
df_tmp = df_tmp[~df_tmp.reg.isin([1,17,19])]
# show the first 10 again
print(df_tmp[['rgiid','area','dhdt','perc_area_meas']].iloc[0:10,:])
