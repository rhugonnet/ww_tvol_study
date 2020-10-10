from __future__ import print_function
import os, sys
import pandas as pd
import numpy as np

in_ext = '/home/atom/ongoing/work_worldwide/tables/table_man_gard_zemp_wout.csv'
df_ext = pd.read_csv(in_ext)
fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'

reg_dir = '/home/atom/ongoing/work_worldwide/vol/reg'
list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,16,17,18,19]] + [os.path.join(reg_dir,'dh_13_14_15_rgi60_int_base_reg.csv')]

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
        list_dhdt.append(dhdt*1000.)
        list_err_dhdt.append(err_dhdt*1000.)
        list_dvoldt.append(dvoldt)
        list_err_dvoldt.append(err_dvoldt)
        list_dmdt.append(dmdt)
        list_err_dmdt.append(err_dmdt)
        list_dmdtda.append(dmdtda*1000.)
        list_err_dmdtda.append(err_dmdtda*1000.)
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

    err_tarea_global = np.sqrt(np.nansum((3 / 100. * df_p.tarea.values) ** 2))
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
    df_sum = df_sum.assign(area=[area_global],area_nodata=[area_nodata_global],dhdt=[dhdt_global*1000],err_dhdt=[err_dhdt_global*1000],dvoldt=[dvoldt_global],err_dvoldt=[err_dvoldt_global]
                                 ,dmdt=[dmdt_global],err_dmdt=[err_dmdt_global],dmdtda=[dmdtda_global*1000],err_dmdtda=[err_dmdtda_global*1000],tarea=[tarea_global],perc_area_res=[perc_area_res_global],perc_area_meas=[perc_area_meas_global],valid_obs=[valid_obs_global]
                                 ,valid_obs_py=[valid_obs_py_global])

    return df_sum

tlim_srocc = [np.datetime64('2006-01-01'),np.datetime64('2016-01-01')]
tlim_full = [np.datetime64('2010-01-01'),np.datetime64('2020-01-01')]

list_df = []
for fn_reg in list_fn_reg:
    df_reg = pd.read_csv(fn_reg)
    df_srocc = integrate_periods(df_reg,tlim_srocc,fn_tarea=fn_tarea,frac_area=1)
    df_srocc['comp'] = 'srocc'
    df_full = integrate_periods(df_reg, tlim_full, fn_tarea=fn_tarea, frac_area=1)
    df_full['comp'] = 'full'
    df_tmp = pd.concat([df_srocc,df_full])
    list_df.append(df_tmp)

df = pd.concat(list_df)

df_s = df[df.comp=='srocc']
df_global_srocc = sum_regions(df_s)
df_global_srocc['reg'] = 22
df_noperiph_srocc = sum_regions(df_s[~df_s.reg.isin([5, 19])])
df_noperiph_srocc['reg'] = 23
df_a_srocc = sum_regions(df_s[df_s.reg.isin([1,3,4,5,6,7,8,9])])
df_a_srocc['reg'] = 24
df_m_srocc = sum_regions(df_s[df_s.reg.isin([1,2,6,8,10,11,12,21,16,17,18])])
df_m_srocc['reg'] = 25
df_srocc_total = pd.concat([df_s, df_global_srocc, df_noperiph_srocc,df_a_srocc,df_m_srocc])
df_srocc_total['comp'] = 'srocc'
df_f = df[df.comp=='full']
df_global_full = sum_regions(df_f)
df_global_full['reg'] = 22
df_noperiph_full = sum_regions(df_f[~df_f.reg.isin([5, 19])])
df_noperiph_full['reg'] = 23
df_a_full = sum_regions(df_f[df_f.reg.isin([1,3,4,5,6,7,8,9])])
df_a_full['reg'] = 24
df_m_full = sum_regions(df_f[df_f.reg.isin([1,2,6,8,10,11,12,21,16,17,18])])
df_m_full['reg'] = 25
df_full_total = pd.concat([df_f, df_global_full, df_noperiph_full,df_a_full,df_m_full])
df_full_total['comp'] = 'full'
df = pd.concat([df_srocc_total,df_full_total])

# df_tmp_acce = sum_regions(df_f[~df_f.reg.isin([5,6,8,19])])

# df.to_csv('/home/atom/ongoing/work_worldwide/tables/table_srocc.csv')