import pandas as pd
import numpy as np

fn_subreg_fig1 = '/home/atom/ongoing/work_worldwide/vol/final/subreg_fig1.csv'

df = pd.read_csv(fn_subreg_fig1)

df['area_valid_obs_py'] = df['valid_obs_py'] * df['area']
df_world = df.groupby(['period','category'])[['dvoldt','dmdt','tarea','area','area_valid_obs_py']].sum()
df_world['valid_obs_py'] = df_world['area_valid_obs_py'] / df_world['area']
df_world['dhdt'] = df_world['dvoldt'] / df_world['tarea']
df_world['period'] = df_world.index.get_level_values(0)
df_world['category'] = df_world.index.get_level_values(1)
df_world['subreg'] = 'Global'

df = pd.concat([df,df_world])

period_yearly = ['20'+str(i).zfill(2)+'-01-01_20'+str(i+1).zfill(2)+'-01-01' for i in range(20)]
period_full = ['2000-01-01_2020-01-01']

ind_yearly = np.logical_and(df.period.isin(period_yearly),df.category=='all')
ind_full = df.period.isin(period_full)

ind = np.logical_or(ind_yearly,ind_full)

df = df[ind]

df = df.drop(columns=[df.columns[0],'err_dhdt','dvoldt','err_dvoldt','err_dmdt','dmdtda','err_dmdtda','valid_obs','area_nodata','frac_area','area_valid_obs_py','reg','perc_area_meas','perc_area_res'])

df = df.round(2)

df = df[['subreg','period','category','dhdt','dmdt','valid_obs_py','area','tarea','lon_reg','lat_reg']]

df.to_csv('/home/atom/ongoing/work_worldwide/tables/final/Source_Data_Fig1.csv',index=False)


