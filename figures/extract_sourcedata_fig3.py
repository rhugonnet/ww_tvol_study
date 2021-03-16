import os
import pandas as pd
import numpy as np
import pyddem.tdem_tools as tt

in_ext = '/home/atom/ongoing/work_worldwide/tables/table_man_gard_zemp_wout.csv'
df_ext = pd.read_csv(in_ext)
reg_dir = '/home/atom/ongoing/work_worldwide/vol/final'

fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'

list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,16,17,18,19]] + [os.path.join(reg_dir,'dh_13_14_15_rgi60_int_base_reg.csv')] + [os.path.join(reg_dir,'dh_01_02_rgi60_int_base_reg.csv')]

tlim_zemp = [np.datetime64('2006-01-01'),np.datetime64('2016-01-01')]
tlim_wouters = [np.datetime64('2002-01-01'),np.datetime64('2017-01-01')]
tlim_cira = [np.datetime64('2002-01-01'),np.datetime64('2020-01-01')]
tlim_gardner = [np.datetime64('2003-01-01'),np.datetime64('2010-01-01')]
# tlim_shean = [np.datetime64('2000-01-01'),np.datetime64('2018-01-01')]
# tlim_braun = [np.datetime64('2000-01-01'),np.datetime64('2013-01-01')]

list_tlim = [tlim_zemp,tlim_wouters,tlim_cira,tlim_gardner]
list_tag = ['hugonnet_2021_period_zemp','hugonnet_2021_period_wout','hugonnet_2021_period_cira','hugonnet_2021_period_gard']

list_df = []
for fn_reg in list_fn_reg:
    df_reg = pd.read_csv(fn_reg)
    df_agg = tt.aggregate_all_to_period(df_reg,list_tlim=list_tlim,fn_tarea=fn_tarea,frac_area=1,list_tag=list_tag)
    list_df.append(df_agg)

df = pd.concat(list_df)


list_fn_reg_multann = [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg_subperiods.csv') for i in np.arange(1,20)]
df_all = pd.DataFrame()
for fn_reg_multann in list_fn_reg_multann:
    df_all= df_all.append(pd.read_csv(fn_reg_multann))

tlims = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = tt.aggregate_indep_regions_rates(df_p)
    df_global['period']=period
    df_noperiph = tt.aggregate_indep_regions_rates(df_p[~df_p.reg.isin([5, 19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
df_glob['reg']=23
df_per['reg']=22
df_glob['tag']='hugonnet_2021_yearly'
df_per['tag']='hugonnet_2021_yearly'

df = pd.concat([df,df_glob,df_per])

tlims = [np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = tt.aggregate_indep_regions_rates(df_p)
    df_global['period']=period
    df_noperiph = tt.aggregate_indep_regions_rates(df_p[~df_p.reg.isin([5, 19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
df_glob['reg']=23
df_per['reg']=22
df_glob['tag']='hugonnet_2021_5year'
df_per['tag']='hugonnet_2021_5year'

df = pd.concat([df,df_glob,df_per])

df = df.drop(columns=['dhdt','err_dhdt','dvoldt','err_dvoldt','valid_obs','valid_obs_py','perc_area_meas','perc_area_res','area_nodata'])

#put all to 2-sigma level
df['err_dmdt'] *= 2
df['err_dmdtda'] *= 2

df_gar = df_ext[['reg','gar','gar_err']]
df_gar.columns = ['reg','dmdtda','err_dmdtda']
df_gar['tag']= 'gardner_2013'
df_gar['period'] = str(tlim_gardner[0])+'_'+str(tlim_gardner[1])

df_zemp = df_ext[['reg','zemp','zemp_err']]
df_zemp.columns = ['reg','dmdtda','err_dmdtda']
df_zemp['tag']= 'zemp_2019'
df_zemp['period'] = str(tlim_zemp[0])+'_'+str(tlim_zemp[1])

df_wout = df_ext[['reg','wout','wout_err']]
df_wout.columns = ['reg','dmdtda','err_dmdtda']
df_wout['tag']= 'wouters_2019'
df_wout['period'] = str(tlim_wouters[0])+'_'+str(tlim_wouters[1])

df_cir = df_ext[['reg','cira','cira_err']]
df_cir.columns = ['reg','dmdtda','err_dmdtda']
df_cir['tag']= 'ciraci_2020'
df_cir['period'] = str(tlim_cira[0])+'_'+str(tlim_cira[1])

df = pd.concat([df,df_gar,df_zemp,df_wout,df_cir])


list_names = ['Northwestern America (01,02)','Alaska (01)','Western Canada and USA (02)','Arctic Canada North (03)','Arctic Canada South (04)','Greenland Periphery (05)', 'Iceland (06)','Svalbard and Jan Mayen (07)', 'Scandinavia (08)','Russian Arctic (09)','North Asia (10)','Central Europe (11)','Caucasus and Middle East (12)','Central Asia (13)','South Asia West (14)','South Asia East (15)','High Mountain Asia (13-15)','Low Latitudes (16)','Southern Andes (17)','New Zealand (18)','Antarctic and Subantarctic (19)','Global excl. regions 05 and 19','Global total']
list_reg = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,16,17,18,19,22,23]

df['reg_name']='NA'
for reg in list_reg:
    df.loc[df.reg==reg,'reg_name']=list_names[list_reg.index(reg)]

df = df[['reg_name','reg','period','tag','dmdtda','err_dmdtda','dmdt','err_dmdt','area','tarea']]

df['area'] /= 1000000
df['tarea'] /= 1000000

df = df.round(3)

df.to_csv('/home/atom/ongoing/work_worldwide/tables/final/Source_Data_Fig3_tmp.csv',index=False)

