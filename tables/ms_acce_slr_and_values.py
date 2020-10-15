from __future__ import print_function
import os, sys
import numpy as np
import pandas as pd
from glob import glob
import pyddem.fit_tools as ft

reg_dir = '/home/atom/ongoing/work_worldwide/vol/reg'
# list_fn_reg_multann = [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg_subperiods.csv') for i in np.arange(1,20)]
# out_csv = '/home/atom/ongoing/work_worldwide/tables/revised/ED_Table_1.csv'
fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'

list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]] #+ [os.path.join(reg_dir,'dh_13_14_15_rgi60_int_base_reg.csv')] #+ [os.path.join(reg_dir,'dh_01_02_rgi60_int_base_reg.csv')]

def integrate_periods(df,tlim,fn_tarea,frac_area):

    times = sorted(list(set(list(df['time']))))
    df_time = pd.DataFrame()
    df_time = df_time.assign(time=times)
    df_time.index = pd.DatetimeIndex(pd.to_datetime(times))
    closest_dates = []
    for tl in tlim:
        time_clos = df_time.iloc[df_time.index.get_loc(pd.to_datetime(tl), method='nearest')][0]
        closest_dates.append(time_clos)

    reg = df.reg.values[0]

    if fn_tarea is not None:
        df_tarea = pd.read_csv(fn_tarea)

        # get regionally evolving areas
        if reg == 20:
            tmp_tarea = df_tarea['RGI1'].values + df_tarea['RGI2'].values
        elif reg == 21:
            tmp_tarea = df_tarea['RGI13'].values + df_tarea['RGI14'].values + df_tarea['RGI15'].values
        else:
            tmp_tarea = df_tarea['RGI' + str(int(reg))].values

        if frac_area is not None:
            tmp_tarea = frac_area * tmp_tarea

        tarea = np.zeros(len(tlim))
        for i in range(len(tlim)):
            # getting years 2000 to 2020
            ind = df_tarea['YEAR'] == tlim[i].astype(object).year
            tarea[i] = tmp_tarea[ind][0] * 1000000
    else:
        tarea = np.repeat(df.area.values[0], len(tlim))

    mult_ann = tlim[1].astype(object).year - tlim[0].astype(object).year

    list_tarea, list_dhdt, list_err_dhdt, list_dvoldt, list_err_dvoldt, list_dmdt, list_err_dmdt, list_valid_obs, list_dt, list_valid_obs_py, list_dmdtda, list_err_dmdtda = (
    [] for i in range(12))
    for i in range(len(tlim) - 1):
        # derive volume change for subperiod
        area = df.area.values[0]
        dvol = (df[df.time == closest_dates[i + 1]].dvol.values - df[df.time == closest_dates[i]].dvol.values)[0]
        dh = dvol / area

        err_dh = np.sqrt(
            df[df.time == closest_dates[i + 1]].err_dh.values[0] ** 2 + df[df.time == closest_dates[i]].err_dh.values[
                0] ** 2)
        err_dvol = np.sqrt((err_dh * area) ** 2 + (dh * df.perc_err_cont.values[0] / 100. * area) ** 2)

        dvoldt = dvol / mult_ann
        err_dvoldt = err_dvol / mult_ann

        dmdt = dvol * 0.85 / 10 ** 9 / mult_ann

        err_dmdt = np.sqrt((err_dvol * 0.85 / 10 ** 9) ** 2 + (
                dvol * 0.06 / 10 ** 9) ** 2) / mult_ann

        linear_area = (tarea[i] + tarea[i + 1]) / 2
        dhdt = dvol / linear_area / mult_ann
        perc_err_linear_area = 1. / 100
        err_dhdt = np.sqrt((err_dvol / linear_area) ** 2 \
                           + (perc_err_linear_area * linear_area * dvol / linear_area ** 2) ** 2) / mult_ann

        dmdtda = dmdt / linear_area * 10**9
        err_dmdtda = np.sqrt((err_dmdt * 10**9 / linear_area) ** 2 \
                           + (perc_err_linear_area * linear_area * dmdt * 10**9/ linear_area ** 2) ** 2)


        ind = np.logical_and(df.time >= closest_dates[i], df.time < closest_dates[i + 1])
        valid_obs = np.nansum(df.valid_obs[ind].values)
        valid_obs_py = np.nansum(df.valid_obs_py[ind].values)

        list_tarea.append(linear_area)
        list_dhdt.append(dhdt)
        list_err_dhdt.append(err_dhdt)
        list_dvoldt.append(dvoldt)
        list_err_dvoldt.append(err_dvoldt)
        list_dmdt.append(dmdt)
        list_err_dmdt.append(err_dmdt)
        list_dmdtda.append(dmdtda)
        list_err_dmdtda.append(err_dmdtda)
        list_dt.append(str(tlim[i]) + '_' + str(tlim[i + 1]))
        list_valid_obs.append(valid_obs)
        list_valid_obs_py.append(valid_obs_py)

    df_mult_ann = pd.DataFrame()
    df_mult_ann = df_mult_ann.assign(period=list_dt, tarea=list_tarea, dhdt=list_dhdt, err_dhdt=list_err_dhdt,
                                     dvoldt=list_dvoldt, err_dvoldt=list_err_dvoldt, dmdt=list_dmdt,
                                     err_dmdt=list_err_dmdt, dmdtda=list_dmdtda,err_dmdtda=list_err_dmdtda,valid_obs=list_valid_obs, valid_obs_py=list_valid_obs_py)
    df_mult_ann['reg'] = reg
    df_mult_ann['perc_area_meas'] = df.perc_area_meas.values[0]
    df_mult_ann['perc_area_res'] = df.perc_area_res.values[0]
    df_mult_ann['area_nodata'] = df.area_nodata.values[0]
    df_mult_ann['area'] = df.area.values[0]

    return df_mult_ann

def sum_regions(df_p):

    # GLOBAL TOTAL
    area_global = np.nansum(df_p.area.values)
    area_nodata_global = np.nansum(df_p.area_nodata.values)
    tarea_global = np.nansum(df_p.tarea.values)

    dmdt_global = np.nansum(df_p.dmdt.values)
    err_dmdt_global = np.sqrt(np.nansum(df_p.err_dmdt.values ** 2))

    dvoldt_global = np.nansum(df_p.dvoldt.values)
    err_dvoldt_global = np.sqrt(np.nansum(df_p.err_dvoldt.values ** 2))

    err_tarea_global = np.sqrt(np.nansum((1 / 100. * df_p.tarea.values) ** 2))
    dhdt_global = np.nansum(df_p.dvoldt.values) / tarea_global
    err_dhdt_global = np.sqrt(
        (err_dvoldt_global / tarea_global) ** 2 + (err_tarea_global * dvoldt_global / (tarea_global ** 2)) ** 2)

    dmdtda_global = dmdt_global * 10**9/tarea_global
    err_dmdtda_global = np.sqrt(
        (err_dmdt_global *10**9/ tarea_global) ** 2 + (err_tarea_global * dmdt_global*10**9 / (tarea_global ** 2)) ** 2)

    perc_area_res_global = np.nansum(df_p.perc_area_res.values * df_p.area.values) / np.nansum(
        df_p.perc_area_res.values)
    perc_area_meas_global = np.nansum(df_p.perc_area_meas.values * df_p.area.values) / np.nansum(
        df_p.perc_area_meas.values)

    valid_obs_global = np.nansum(df_p.valid_obs.values * df_p.area.values) / np.nansum(df_p.perc_area_res.values)
    valid_obs_py_global = np.nansum(df_p.valid_obs_py.values * df_p.area.values) / np.nansum(df_p.perc_area_res.values)

    df_sum = pd.DataFrame()
    df_sum = df_sum.assign(area=[area_global],area_nodata=[area_nodata_global],dhdt=[dhdt_global],err_dhdt=[err_dhdt_global],dvoldt=[dvoldt_global],err_dvoldt=[err_dvoldt_global]
                                 ,dmdt=[dmdt_global],err_dmdt=[err_dmdt_global],dmdtda=[dmdtda_global],err_dmdtda=[err_dmdtda_global],tarea=[tarea_global],perc_area_res=[perc_area_res_global],perc_area_meas=[perc_area_meas_global],valid_obs=[valid_obs_global]
                                 ,valid_obs_py=[valid_obs_py_global])

    return df_sum

periods = ['2000-01-01_2005-01-01','2005-01-01_2010-01-01','2010-01-01_2015-01-01','2015-01-01_2020-01-01','2000-01-01_2020-01-01']
tlims = [(np.datetime64('2000-01-01'),np.datetime64('2005-01-01')),(np.datetime64('2005-01-01'),np.datetime64('2010-01-01')),(np.datetime64('2010-01-01'),np.datetime64('2015-01-01')),(np.datetime64('2015-01-01'),np.datetime64('2020-01-01')),(np.datetime64('2000-01-01'),np.datetime64('2020-01-01'))]


list_df = []
for fn_reg in list_fn_reg:
    for period in periods:
        df_tmp = integrate_periods(pd.read_csv(fn_reg),tlims[periods.index(period)],fn_tarea=fn_tarea,frac_area=1)

        list_df.append(df_tmp)
df = pd.concat(list_df)

list_df_all = []
for period in periods:

    df_p = df[df.period == period]
    df_global = sum_regions(df_p)
    df_global['reg']='global'
    df_global['period'] = period

    df_noperiph = sum_regions(df_p[~df_p.reg.isin([5,19])])
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
        df_tmp = integrate_periods(pd.read_csv(fn_reg),tlims[periods.index(period)],fn_tarea=fn_tarea,frac_area=1)

        list_df_yrly.append(df_tmp)
df_yrly = pd.concat(list_df_yrly)

list_df_all_yrly = []
for period in periods:

    df_p = df_yrly[df_yrly.period == period]
    df_global = sum_regions(df_p)
    df_global['reg']='global'
    df_global['period'] = period

    df_noperiph = sum_regions(df_p[~df_p.reg.isin([5,19])])
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

df_nonpolar = sum_regions(df_tot[df_tot.reg.isin([10,11,12,16,17,18])])

print('Non-polar specific rate: '+'{:.2f}'.format(df_nonpolar.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_nonpolar.err_dmdtda.values[0])+' m w.e yr-1')

#for HMA, account for correlated error all at once:
fn_hma=os.path.join(reg_dir,'dh_13_14_15_rgi60_int_base_reg.csv')

df_hma =  integrate_periods(pd.read_csv(fn_hma),(np.datetime64('2000-01-01'),np.datetime64('2020-01-01')),fn_tarea=fn_tarea,frac_area=1)
print('HMA specific rate: '+'{:.2f}'.format(df_hma.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_hma.err_dmdtda.values[0])+' m w.e yr-1')

print('Antarctic and Subantartic specific rate: '+'{:.2f}'.format(df_tot[df_tot.reg==19].dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_tot[df_tot.reg==19].err_dmdtda.values[0])+' m w.e yr-1')

#corresponding period for comparison to Shean et al., 2019
df_hma =  integrate_periods(pd.read_csv(fn_hma),(np.datetime64('2000-01-01'),np.datetime64('2018-01-01')),fn_tarea=fn_tarea,frac_area=1)
print('Shean comparison: HMA specific rate: '+'{:.2f}'.format(df_hma.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_hma.err_dmdtda.values[0])+' m w.e yr-1')

#corresponding period for comparison to Braun et al., 2019
df_sa =  integrate_periods(pd.read_csv(list_fn_reg[16]),(np.datetime64('2000-01-01'),np.datetime64('2013-01-01')),fn_tarea=fn_tarea,frac_area=1)
print('Braun comparison: Southern Andes specific rate: '+'{:.2f}'.format(df_sa.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_sa.err_dmdtda.values[0])+' m w.e yr-1')

df_trp =  integrate_periods(pd.read_csv(list_fn_reg[15]),(np.datetime64('2000-01-01'),np.datetime64('2013-01-01')),fn_tarea=fn_tarea,frac_area=1)
print('Braun comparison: Tropics specific rate: '+'{:.2f}'.format(df_trp.dmdtda.values[0])+' ± '+'{:.2f}'.format(2*df_trp.err_dmdtda.values[0])+' m w.e yr-1')

#TEMPORAL VARIABILITIES

df_ant_grl_00_04 = sum_regions(df_all[np.logical_and(df_all.reg.isin([5,19]),df_all.period=='2000-01-01_2005-01-01')])
df_ant_grl_15_19 = sum_regions(df_all[np.logical_and(df_all.reg.isin([5,19]),df_all.period=='2015-01-01_2020-01-01')])

print('GRL and ANT mass loss in 2000-2004:'+'{:.2f}'.format(df_ant_grl_00_04.dmdt.values[0])+' ± '+'{:.2f}'.format(2*df_ant_grl_00_04.err_dmdt.values[0])+' Gt yr-1')
print('GRL and ANT mass loss in 2015-2019:'+'{:.2f}'.format(df_ant_grl_15_19.dmdt.values[0])+' ± '+'{:.2f}'.format(2*df_ant_grl_15_19.err_dmdt.values[0])+' Gt yr-1')

print('Iceland thinning rates 2000-2004:'+'{:.2f}'.format(df_all[np.logical_and(df_all.reg==6,df_all.period=='2000-01-01_2005-01-01')].dhdt.values[0])+' ± '+'{:.2f}'.format(2*df_all[np.logical_and(df_all.reg==6,df_all.period=='2000-01-01_2005-01-01')].err_dhdt.values[0])+' m yr-1')
print('Iceland thinning rates 2015-2019:'+'{:.2f}'.format(df_all[np.logical_and(df_all.reg==6,df_all.period=='2015-01-01_2020-01-01')].dhdt.values[0])+' ± '+'{:.2f}'.format(2*df_all[np.logical_and(df_all.reg==6,df_all.period=='2015-01-01_2020-01-01')].err_dhdt.values[0])+' m yr-1')

df_acc_00_04 = sum_regions(df_all[np.logical_and(df_all.reg.isin([1,2,3,4,7,9,10,11,12,13,14,15,16,17,18]),df_all.period=='2000-01-01_2005-01-01')])
df_acc_15_19 = sum_regions(df_all[np.logical_and(df_all.reg.isin([1,2,3,4,7,9,10,11,12,13,14,15,16,17,18]),df_all.period=='2015-01-01_2020-01-01')])

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
