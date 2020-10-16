"""
@author: hugonnet
derive all values present in the text of the manuscript: accelerations, SLR contributions, etc..
"""
from __future__ import print_function
import os, sys
import numpy as np
import pandas as pd
from glob import glob
import pyddem.fit_tools as ft
import pyddem.tdem_tools as tt

reg_dir = '/home/atom/ongoing/work_worldwide/vol/reg'
fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'
list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]

periods = ['2000-01-01_2005-01-01','2005-01-01_2010-01-01','2010-01-01_2015-01-01','2015-01-01_2020-01-01','2000-01-01_2020-01-01']
tlims = [(np.datetime64('2000-01-01'),np.datetime64('2005-01-01')),(np.datetime64('2005-01-01'),np.datetime64('2010-01-01')),(np.datetime64('2010-01-01'),np.datetime64('2015-01-01')),(np.datetime64('2015-01-01'),np.datetime64('2020-01-01')),(np.datetime64('2000-01-01'),np.datetime64('2020-01-01'))]


list_df = []
for fn_reg in list_fn_reg:
    for period in periods:
        df_tmp = tt.aggregate_all_to_period(pd.read_csv(fn_reg),[tlims[periods.index(period)]],fn_tarea=fn_tarea,frac_area=1)

        list_df.append(df_tmp)
df = pd.concat(list_df)

list_df_all = []
for period in periods:

    df_p = df[df.period == period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['reg']='global'
    df_global['period'] = period

    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['reg']='global_noperiph'
    df_noperiph['period'] =period

    df_full_p = pd.concat([df_p,df_noperiph,df_global])

    list_df_all.append(df_full_p)

df_all = pd.concat(list_df_all)
df_g = df_all[df_all.reg=='global']
df_np = df_all[df_all.reg=='global_noperiph']

#CONTRIBUTION TO SLR

# from M. Ablain: AVISO-based sea-level rise trend for 2000.0-2020.0 and 1-sigma errors
gmsl_trend = 3.56
gmsl_trend_err = 0.2

gmsl_acc = 0.15
gmsl_acc_err = 0.04

glac_trend = df_g[df_g.period == '2000-01-01_2020-01-01'].dmdt.values[0]/361.8
glac_trend_err = df_g[df_g.period == '2000-01-01_2020-01-01'].err_dmdt.values[0]/361.8

print('Glacier mass loss totalled '+'{:.2f}'.format(df_g[df_g.period == '2000-01-01_2020-01-01'].dmdt.values[0])+' ± '+'{:.2f}'.format(2*df_g[df_g.period == '2000-01-01_2020-01-01'].err_dmdt.values[0])+ ' Gt yr-1')

contr_trend = -glac_trend/gmsl_trend*100
contr_trend_err = -glac_trend/gmsl_trend*np.sqrt((gmsl_trend_err/gmsl_trend)**2+(glac_trend_err/glac_trend)**2)*100
print('Glacier contribution to SLR is '+'{:.2f}'.format(contr_trend)+' % ± '+'{:.2f}'.format(2*contr_trend_err)+' %')

#GLACIER ACCELERATION

beta1_t, beta0, incert_slope, _, _ = ft.wls_matrix(x=np.arange(0,16,5),y=df_g.dhdt.values[:-1],w=1/df_g.err_dhdt.values[:-1]**2)
print('Global thinning acceleration is '+'{:.5f}'.format(beta1_t)+' ± '+'{:.5f}'.format(2*incert_slope)+ ' m yr-2')
beta1, beta0, incert_slope, _, _ = ft.wls_matrix(x=np.array([0,5,10,15]),y=df_np.dhdt.values[:-1],w=1/df_np.err_dhdt.values[:-1]**2)
print('Global excl. GRL and ANT thinning acceleration is '+'{:.5f}'.format(beta1)+' ± '+'{:.5f}'.format(2*incert_slope)+ ' m yr-2')


beta1_g, beta0, incert_slope_g, _, _ = ft.wls_matrix(x=np.arange(0,16,5),y=df_g.dmdt.values[:-1],w=1/df_g.err_dmdt.values[:-1]**2)
print('Global mass loss acceleration is '+'{:.5f}'.format(beta1_g)+' ± '+'{:.5f}'.format(2*incert_slope_g)+ ' Gt yr-2')
beta1, beta0, incert_slope, _, _ = ft.wls_matrix(x=np.array([0,5,10,15]),y=df_np.dmdt.values[:-1],w=1/df_np.err_dmdt.values[:-1]**2)
print('Global excl. GRL and ANT mass loss acceleration is '+'{:.5f}'.format(beta1)+' ± '+'{:.5f}'.format(2*incert_slope)+ ' Gt yr-2')

#CONTRIBUTION TO ACCELERATION OF SLR
glac_acc = -beta1_g/361.8
glac_acc_err = incert_slope_g/361.8
contr_acc = glac_acc/gmsl_acc*100

# error is not symmetrial, error of acceleration of SLR is 20 times larger than glacier error
rss_gmsl_acc_err = np.sqrt(glac_acc_err**2+gmsl_acc_err**2)
upper_bound = glac_acc/(gmsl_acc-2*rss_gmsl_acc_err)*100
lower_bound = glac_acc/(gmsl_acc+2*rss_gmsl_acc_err)*100

print('Glacier contribution to acceleration of SLR is '+'{:.2f}'.format(contr_acc)+' % with 95% confidence interval of '+'{:.1f}'.format(lower_bound)+'-'+'{:.1f}'.format(upper_bound)+' %')

#YEARLY VALUES

periods = ['20'+str(i).zfill(2)+'-01-01_'+'20'+str(i+1).zfill(2)+'-01-01' for i in np.arange(0,20,1)]
tlims = [(np.datetime64('20'+str(i).zfill(2)+'-01-01'),np.datetime64('20'+str(i+1).zfill(2)+'-01-01')) for i in np.arange(0,20,1)]

list_df_yrly = []
for fn_reg in list_fn_reg:
    for period in periods:
        df_tmp = tt.aggregate_all_to_period(pd.read_csv(fn_reg),[tlims[periods.index(period)]],fn_tarea=fn_tarea,frac_area=1)

        list_df_yrly.append(df_tmp)
df_yrly = pd.concat(list_df_yrly)

list_df_all_yrly = []
for period in periods:

    df_p = df_yrly[df_yrly.period == period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['reg']='global'
    df_global['period'] = period

    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['reg']='global_noperiph'
    df_noperiph['period'] =period

    df_full_p = pd.concat([df_p,df_noperiph,df_global])

    list_df_all_yrly.append(df_full_p)

df_all_yrly = pd.concat(list_df_all_yrly)

dhdt_2000_global = df_all_yrly[np.logical_and(df_all_yrly.period=='2000-01-01_2001-01-01',df_all_yrly.reg=='global_noperiph')].dhdt.values[0]
dhdt_2000_global_err = df_all_yrly[np.logical_and(df_all_yrly.period=='2000-01-01_2001-01-01',df_all_yrly.reg=='global_noperiph')].err_dhdt.values[0]

dhdt_2019_global = df_all_yrly[np.logical_and(df_all_yrly.period=='2019-01-01_2020-01-01',df_all_yrly.reg=='global_noperiph')].dhdt.values[0]
dhdt_2019_global_err = df_all_yrly[np.logical_and(df_all_yrly.period=='2019-01-01_2020-01-01',df_all_yrly.reg=='global_noperiph')].err_dhdt.values[0]

print('Global excl. GRL and ANT thinning rates in 2000: '+'{:.3f}'.format(dhdt_2000_global)+' ± '+'{:.3f}'.format(2*dhdt_2000_global_err)+' m yr-1')
print('Global excl. GRL and ANT thinning rates in 2019: '+'{:.3f}'.format(dhdt_2019_global)+' ± '+'{:.3f}'.format(2*dhdt_2019_global_err)+' m yr-1')

# REGIONAL PERCENTAGES

df_tot = df_all[df_all.period == '2000-01-01_2020-01-01']

list_cont_perc = []
for i in range(19):
    cont = df_tot[df_tot.reg==i+1].dmdt.values[0]/df_tot[df_tot.reg=='global'].dmdt.values[0]*100
    list_cont_perc.append(cont)

print('Contribution of Alaska: '+'{:.1f}'.format(list_cont_perc[0])+' %')
print('Contribution of Greenland Periphery: '+'{:.1f}'.format(list_cont_perc[4])+' %')
print('Contribution of Arctic Canada North: '+'{:.1f}'.format(list_cont_perc[2])+' %')
print('Contribution of Arctic Canada South: '+'{:.1f}'.format(list_cont_perc[3])+' %')
print('Contribution of Antarctic Periphery: '+'{:.1f}'.format(list_cont_perc[18])+' %')
print('Contribution of High Moutain Asia: '+'{:.1f}'.format(list_cont_perc[12]+list_cont_perc[13]+list_cont_perc[14])+' %')
print('Contribution of Southern Andes: '+'{:.1f}'.format(list_cont_perc[16])+' %')


#separate contribution from North Greenland and South: done manually

print('Iceland specific rate: '+'{:.2f}'.format(df_tot[df_tot.reg==6].dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_tot[df_tot.reg==6].err_dmdtda.values[0])+' m w.e yr-1')

df_nonpolar = tt.aggregate_indep_regions(df_tot[df_tot.reg.isin([10,11,12,16,17,18])])

print('Non-polar specific rate: '+'{:.2f}'.format(df_nonpolar.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_nonpolar.err_dmdtda.values[0])+' m w.e yr-1')

#for HMA, account for correlated error all at once:
fn_hma=os.path.join(reg_dir,'dh_13_14_15_rgi60_int_base_reg.csv')

df_hma =  tt.aggregate_all_to_period(pd.read_csv(fn_hma),[(np.datetime64('2000-01-01'),np.datetime64('2020-01-01'))],fn_tarea=fn_tarea,frac_area=1)
print('HMA specific rate: '+'{:.2f}'.format(df_hma.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_hma.err_dmdtda.values[0])+' m w.e yr-1')

print('Antarctic and Subantartic specific rate: '+'{:.2f}'.format(df_tot[df_tot.reg==19].dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_tot[df_tot.reg==19].err_dmdtda.values[0])+' m w.e yr-1')

#corresponding period for comparison to Shean et al., 2019
df_hma =  tt.aggregate_all_to_period(pd.read_csv(fn_hma),[(np.datetime64('2000-01-01'),np.datetime64('2018-01-01'))],fn_tarea=fn_tarea,frac_area=1)
print('Shean comparison: HMA specific rate: '+'{:.2f}'.format(df_hma.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_hma.err_dmdtda.values[0])+' m w.e yr-1')

#corresponding period for comparison to Braun et al., 2019
df_sa =  tt.aggregate_all_to_period(pd.read_csv(list_fn_reg[16]),[(np.datetime64('2000-01-01'),np.datetime64('2013-01-01'))],fn_tarea=fn_tarea,frac_area=1)
print('Braun comparison: Southern Andes specific rate: '+'{:.2f}'.format(df_sa.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_sa.err_dmdtda.values[0])+' m w.e yr-1')

df_trp =  tt.aggregate_all_to_period(pd.read_csv(list_fn_reg[15]),[(np.datetime64('2000-01-01'),np.datetime64('2013-01-01'))],fn_tarea=fn_tarea,frac_area=1)
print('Braun comparison: Tropics specific rate: '+'{:.2f}'.format(df_trp.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_trp.err_dmdtda.values[0])+' m w.e yr-1')

#TEMPORAL VARIABILITIES

df_ant_grl_00_04 = tt.aggregate_indep_regions(df_all[np.logical_and(df_all.reg.isin([5,19]),df_all.period=='2000-01-01_2005-01-01')])
df_ant_grl_15_19 = tt.aggregate_indep_regions(df_all[np.logical_and(df_all.reg.isin([5,19]),df_all.period=='2015-01-01_2020-01-01')])

print('GRL and ANT mass loss in 2000-2004:'+'{:.2f}'.format(df_ant_grl_00_04.dmdt.values[0])+' ± '+'{:.2f}'.format(2*df_ant_grl_00_04.err_dmdt.values[0])+' Gt yr-1')
print('GRL and ANT mass loss in 2015-2019:'+'{:.2f}'.format(df_ant_grl_15_19.dmdt.values[0])+' ± '+'{:.2f}'.format(2*df_ant_grl_15_19.err_dmdt.values[0])+' Gt yr-1')

print('Iceland thinning rates 2000-2004:'+'{:.2f}'.format(df_all[np.logical_and(df_all.reg==6,df_all.period=='2000-01-01_2005-01-01')].dhdt.values[0])+' ± '+'{:.2f}'.format(2*df_all[np.logical_and(df_all.reg==6,df_all.period=='2000-01-01_2005-01-01')].err_dhdt.values[0])+' m yr-1')
print('Iceland thinning rates 2015-2019:'+'{:.2f}'.format(df_all[np.logical_and(df_all.reg==6,df_all.period=='2015-01-01_2020-01-01')].dhdt.values[0])+' ± '+'{:.2f}'.format(2*df_all[np.logical_and(df_all.reg==6,df_all.period=='2015-01-01_2020-01-01')].err_dhdt.values[0])+' m yr-1')

df_acc_00_04 = tt.aggregate_indep_regions(df_all[np.logical_and(df_all.reg.isin([1,2,3,4,7,9,10,11,12,13,14,15,16,17,18]),df_all.period=='2000-01-01_2005-01-01')])
df_acc_15_19 = tt.aggregate_indep_regions(df_all[np.logical_and(df_all.reg.isin([1,2,3,4,7,9,10,11,12,13,14,15,16,17,18]),df_all.period=='2015-01-01_2020-01-01')])

print('Accelerating regions mass loss in 2000-2004:'+'{:.2f}'.format(df_acc_00_04.dmdt.values[0])+' ± '+'{:.2f}'.format(2*df_acc_00_04.err_dmdt.values[0])+' Gt yr-1')
print('Accelerating regions mass loss in 2015-2019:'+'{:.2f}'.format(df_acc_15_19.dmdt.values[0])+' ± '+'{:.2f}'.format(2*df_acc_15_19.err_dmdt.values[0])+' Gt yr-1')

contr_alaska = (df_all[np.logical_and(df_all.reg==1,df_all.period=='2015-01-01_2020-01-01')].dmdt.values[0]-df_all[np.logical_and(df_all.reg==1,df_all.period=='2000-01-01_2005-01-01')].dmdt.values[0])/(df_acc_15_19.dmdt.values[0]-df_acc_00_04.dmdt.values[0])*100
contr_hma = (df_all[np.logical_and(df_all.reg==13,df_all.period=='2015-01-01_2020-01-01')].dmdt.values[0]+df_all[np.logical_and(df_all.reg==14,df_all.period=='2015-01-01_2020-01-01')].dmdt.values[0]+df_all[np.logical_and(df_all.reg==15,df_all.period=='2015-01-01_2020-01-01')].dmdt.values[0]-df_all[np.logical_and(df_all.reg==13,df_all.period=='2000-01-01_2005-01-01')].dmdt.values[0]-df_all[np.logical_and(df_all.reg==14,df_all.period=='2000-01-01_2005-01-01')].dmdt.values[0]-df_all[np.logical_and(df_all.reg==15,df_all.period=='2000-01-01_2005-01-01')].dmdt.values[0])/(df_acc_15_19.dmdt.values[0]-df_acc_00_04.dmdt.values[0])*100
contr_wna = (df_all[np.logical_and(df_all.reg==2,df_all.period=='2015-01-01_2020-01-01')].dmdt.values[0]-df_all[np.logical_and(df_all.reg==2,df_all.period=='2000-01-01_2005-01-01')].dmdt.values[0])/(df_acc_15_19.dmdt.values[0]-df_acc_00_04.dmdt.values[0])*100

print('Contribution of Alaska to the accelerating regions: '+'{:.1f}'.format(contr_alaska)+' %')
print('Contribution of HMA to the accelerating regions: '+'{:.1f}'.format(contr_hma)+' %')
print('Contribution of WNA to the accelerating regions: '+'{:.1f}'.format(contr_wna)+' %')

#SENSIBILITY

temp = 0.0305 #K yr-1, from ERA 5 data
mwe = beta1_t * 0.85 #convert to m w.e. yr-1

sensi = mwe/temp

print('Mass balance sensitivity to temperature: '+'{:.2f}'.format(sensi)+' m w.e. yr-1 K-1')
