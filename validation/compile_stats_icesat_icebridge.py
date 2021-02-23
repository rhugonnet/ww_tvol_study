"""
@author: hugonnet
compile the differences to IceBridge and ICESat into elevation biases, standardized uncertainties, and elevation change biases for all regions and parameters of interest
"""

import os
import pandas as pd
import numpy as np
from pybob.ddem_tools import nmad
from glob import glob
import scipy.interpolate
from sklearn.linear_model import LinearRegression
from pybob.bob_tools import mkdir_p
import pyddem.fit_tools as ft

# dir_valid = '/home/atom/ongoing/work_worldwide/validation/icesat'
# dir_valid_out = '/home/atom/ongoing/work_worldwide/validation/compiled'

dir_valid = '/data/icesat/travail_en_cours/romain/results/valid'
dir_valid_out = '/data/icesat/travail_en_cours/romain/results/valid_compil_stable'

mkdir_p(dir_valid_out)

list_fn_valid = glob(os.path.join(dir_valid,'*.csv'),recursive=True)

print('Found validation file list:')
print(list_fn_valid)

print('Concatenating data...')
df = pd.DataFrame()
for fn_valid in list_fn_valid:
    tmp_df = pd.read_csv(fn_valid)
    reg = int(os.path.basename(fn_valid).split('_')[2])
    if os.path.basename(fn_valid).split('_')[1] == 'ICESat':
        sensor = 'ICS'
    else:
        sensor = 'IB'
    tmp_df = tmp_df.assign(reg=reg,sensor=sensor)
    df = df.append(tmp_df)

#we want time series minus validation, easier to conceptualize
df.zsc = -df.zsc
df.dh = -df.dh
df.dh_ref = -df.dh_ref

#glacier only
df = df[np.abs(df.dh_ref)<300]
df = df[df.pos==1]

#remove very large outliers
nmad_gla = nmad(df.zsc)
df=df[np.abs(df.zsc-np.nanmedian(df.zsc))<10*nmad_gla]

def bin_valid_df_by_vals(df,bins,bins_val,list_var=['dh','zsc'],ls_dvardt=True,weight_ib=1./40,return_ls=False):

    mid_bin, med, std, dvardt, dvardt_2std, ns_ics, ns_ib = ([] for i in range(7))
    for i in range(len(bins)-1):
        ind = np.logical_and(bins_val >= bins[i],bins_val < bins[i+1])
        df_ind = df[ind]
        nics = np.count_nonzero(df_ind.sensor == 'ICS')
        nib=np.count_nonzero(df_ind.sensor == 'IB')
        ns_ics.append(nics)
        ns_ib.append(nib)
        mid_bin.append(bins[i] + 0.5*(bins[i+1]-bins[i]))

        sub_med = []
        sub_std = []
        sub_dvardt = []
        sub_dvardt_2std = []
        sub_mu = []
        sub_w = []
        sub_t = []
        for var in list_var:
            if weight_ib is not None:
                if nics != 0 or nib !=0:
                    sub_med.append(np.nansum((np.nanmedian(df_ind[df_ind.sensor=='ICS'][var])*nics,np.nanmedian(df_ind[df_ind.sensor=='IB'][var])*nib*weight_ib))/(nics+nib*weight_ib))
                    sub_std.append(np.nansum((nmad(df_ind[df_ind.sensor == 'ICS'][var]) * nics,nmad(df_ind[df_ind.sensor == 'IB'][var]) * nib * weight_ib)) / (nics + nib * weight_ib))
                else:
                    sub_med.append(np.nan)
                    sub_std.append(np.nan)
            else:
                sub_med.append(np.nanmedian(df_ind[var]))
                sub_std.append(nmad(df_ind[var].values))

            if ls_dvardt:
                list_t = sorted(list(set(list(df_ind.t.values))))
                ftime_delta = np.array(
                    [(np.datetime64(t) - np.datetime64('{}-01-01'.format(int(2000)))).astype(int) / 365.2422 for t in list_t])
                mu = []
                w = []
                for val_t in list_t:
                    ind_t = df_ind.t.values == val_t
                    df_ind_t = df_ind[ind_t]
                    nics_t = np.count_nonzero(df_ind_t.sensor == 'ICS')
                    nib_t = np.count_nonzero(df_ind_t.sensor == 'IB')
                    if np.count_nonzero(ind_t) > 20:
                        med_t = np.nansum((np.nanmedian(df_ind_t[df_ind_t.sensor=='ICS'][var])*nics_t,np.nanmedian(df_ind_t[df_ind_t.sensor=='IB'][var])*nib_t*weight_ib))/(nics_t+nib_t*weight_ib)
                        mu.append(med_t)
                        std_t = np.nansum((nmad(df_ind_t[df_ind_t.sensor == 'ICS'][var]) * nics_t,nmad(df_ind_t[df_ind_t.sensor == 'IB'][var]) * nib_t * weight_ib)) / (nics_t + nib_t * weight_ib)
                        w.append(std_t/np.sqrt(nics_t+nib_t*weight_ib))
                    else:
                        mu.append(np.nan)
                        w.append(np.nan)
                mu = np.array(mu)
                w = np.array(w)
                if np.count_nonzero(~np.isnan(mu)) > 5:
                    # reg = LinearRegression().fit(ftime_delta[~np.isnan(mu)].reshape(-1, 1),
                    #                              mu[~np.isnan(mu)].reshape(-1, 1))

                    beta1, _ , incert_slope, _, _ = ft.wls_matrix(ftime_delta[~np.isnan(mu)], mu[~np.isnan(mu)], 1. / w[~np.isnan(mu)]**2,
                                                                  conf_slope=0.95)
                    # fig = plt.figure()
                    # plt.scatter(ftime_delta,mu_dh,color='red')
                    # plt.plot(np.arange(0,10,0.1),reg.predict(np.arange(0,10,0.1).reshape(-1,1)),color='black',label=reg)
                    # plt.ylim([-20,20])
                    # plt.text(5,0,str(reg.coef_[0]))
                    # plt.legend()
                    # coef = reg.coef_[0][0]
                    coef = beta1
                    sub_dvardt.append(coef)
                    sub_dvardt_2std.append(incert_slope)
                else:
                    sub_dvardt.append(np.nan)
                    sub_dvardt_2std.append(np.nan)

                sub_mu.append(mu)
                sub_w.append(w)
                sub_t.append(ftime_delta)
        med.append(sub_med)
        std.append(sub_std)
        dvardt.append(sub_dvardt)
        dvardt_2std.append(sub_dvardt_2std)



    df_out = pd.DataFrame()
    df_out = df_out.assign(mid_bin=mid_bin, ns_ics=ns_ics, ns_ib=ns_ib)
    for var in list_var:
        df_out['med_' + var] = list(zip(*med))[list_var.index(var)]
        df_out['nmad_' + var] = list(zip(*std))[list_var.index(var)]
        if ls_dvardt:
            df_out['d'+var+'_dt'] = list(zip(*dvardt))[list_var.index(var)]
            df_out['d'+var+'_dt_2std'] = list(zip(*dvardt_2std))[list_var.index(var)]

    if return_ls and ls_dvardt:
        df_ls = pd.DataFrame()
        for var in list_var:
            # print(len(sub_mu))
            df_ls['mu_' + var] = sub_mu[list_var.index(var)]
            df_ls['w_' + var] = sub_w[list_var.index(var)]
            df_ls['t_' + var] = sub_t[list_var.index(var)]
        return df_out, df_ls
    else:
        return df_out


def bin_valid_df_by_season(df,var='dh',weight_ib=1./40):

    date=df.t
    season_month_bins = np.arange(1,13,1)
    mon = pd.DatetimeIndex(date).month.values

    med, std, mid_bin, ns_ics, ns_ib = ([] for i in range(5))
    for i in range(len(season_month_bins)):
        ind = (mon == season_month_bins[i])
        df_ind = df[ind]
        nics = np.count_nonzero(df_ind.sensor == 'ICS')
        nib = np.count_nonzero(df_ind.sensor == 'IB')
        ns_ics.append(nics)
        ns_ib.append(nib)
        # med.append(np.nanmedian(df_ind[var].values))
        # std.append(nmad(df_ind[var].values))
        if nics != 0 or nib != 0:
            med.append(np.nansum((np.nanmedian(df_ind[df_ind.sensor == 'ICS'][var]) * nics,
                   np.nanmedian(df_ind[df_ind.sensor == 'IB'][var]) * nib * weight_ib)) / (nics + nib * weight_ib))
            std.append(np.nansum((nmad(df_ind[df_ind.sensor == 'ICS'][var]) * nics,
                                      nmad(df_ind[df_ind.sensor == 'IB'][var]) * nib * weight_ib)) / (
                                       nics + nib * weight_ib))
        else:
            med.append(np.nan)
            std.append(np.nan)

        mid_bin.append(season_month_bins[i])

    df_out = pd.DataFrame()
    df_out = df_out.assign(seas_dec=mid_bin,ns_ics=ns_ics,ns_ib=ns_ib)
    df_out['med_'+var]=med
    df_out['nmad_'+var]=std

    return df_out

#1/ BEFORE SEASONAL CORRECTIONS

print('Deriving statistics without seasonal corrections')

#normalize elevation by region
list_reg = sorted(list(set(list(df.reg))))
for reg in list_reg:
    min_elev = np.nanpercentile(df[df.reg == reg].h,95)
    max_elev = np.nanpercentile(df[df.reg == reg].h,5)
    df.loc[df.reg == reg,'h'] = (df.loc[df.reg == reg,'h'] - min_elev)/(max_elev-min_elev)
    ind_0 = np.logical_and(df.reg==reg,df.h<0)
    df.loc[ind_0,'h']=np.nan
    ind_1 = np.logical_and(df.reg==reg,df.h>1)
    df.loc[ind_1,'h']=np.nan

bin_dt = [0,60,120,180,240,300,360,540,720,900,1080]
dt = bin_valid_df_by_vals(df, bin_dt, np.abs(df.dt))
dt['type'] = 'dt'

bin_t = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]
t = bin_valid_df_by_vals(df,bin_t,pd.to_datetime(df.t))
t['type'] = 't'

bin_h = np.arange(0,1.1,0.1)
h = bin_valid_df_by_vals(df,bin_h,df.h)
h['type'] = 'h'

bin_dh_tot = [-150,-100,-50,-35,-15,-10,-5,0,5,10,15]
dh_tot = bin_valid_df_by_vals(df, bin_dh_tot, df.dh_tot)
dh_tot['type'] = 'dh_tot'

bin_reg = np.arange(1, 21)
r = bin_valid_df_by_vals(df, bin_reg, df.reg)
r['type'] = 'reg'

bin_dh = np.arange(-12,13,2)
dh = bin_valid_df_by_vals(df, bin_dh, df.dh)
dh['type']  ='dh'

bin_zsc = np.arange(-3,3.1,0.5)
zsc = bin_valid_df_by_vals(df, bin_zsc, df.zsc)
zsc['type']  ='zsc'

bin_all = [min(df.zsc),max(df.zsc)]
a, a_ls = bin_valid_df_by_vals(df,bin_all,df.zsc,return_ls=True)
a['type'] = 'all'

df_north = df[df.reg <=15]
bin_months = np.arange(1, 14, 2)
months = pd.DatetimeIndex(df_north.t).month.values
m_n = bin_valid_df_by_vals(df_north,bin_months,months)
m_n['type'] = 'seas_north'

df_south = df[df.reg > 15]
bin_months = np.arange(1, 14, 2)
months = pd.DatetimeIndex(df_south.t).month.values
m_s = bin_valid_df_by_vals(df_south,bin_months,months)
m_s['type'] = 'seas_south'

df_init = pd.concat([dt,t,h,dh_tot,r,dh,zsc,a,m_n,m_s])
df_init['seas_corr'] = 0

fn_out = os.path.join(dir_valid_out,'valid_ICS_IB_all_bins_all_ls_init.csv')
a_ls.to_csv(fn_out)

#2/ COMPUTE SEASONAL BIASES BY REGION

print('Computing and applying seasonal corrections')

list_s = []
list_s2 = []
for reg in list(set(list(df.reg))):
    df_reg = df[df.reg == reg]
    # df_reg = df_reg[df_reg.sensor=='ICS']
    s = bin_valid_df_by_season(df_reg)
    coefs1, _ = scipy.optimize.curve_fit(lambda t, a, b, c: a ** 2 * np.sin(t * 2 * np.pi / 12 + c) + b, s.seas_dec[~np.isnan(s.med_dh)].values,
                                     s.med_dh[~np.isnan(s.med_dh)].values)
    s2 = bin_valid_df_by_season(df_reg,var='zsc')
    coefs2, _ = scipy.optimize.curve_fit(lambda t, a, b, c: a ** 2 * np.sin(t * 2 * np.pi / 12 + c) + b, s2.seas_dec[~np.isnan(s2.med_zsc)].values,
                                     s2.med_zsc[~np.isnan(s2.med_zsc)].values)
    season_month_bins = np.arange(1, 13, 1)
    mon = pd.DatetimeIndex(df.t).month.values
    for i in range(len(season_month_bins)):
        ind = np.logical_and(mon == season_month_bins[i],df.reg==reg)
        df.loc[ind,'dh'] -= coefs1[0] ** 2 * np.sin(season_month_bins[i] * 2 * np.pi / 12 + coefs1[2]) + coefs1[1]
        df.loc[ind,'zsc'] -= coefs2[0] ** 2 * np.sin(season_month_bins[i] * 2 * np.pi / 12 + coefs2[2]) + coefs2[1]
    s['reg'] = reg
    s['var'] = 'dh'
    s2['reg']=reg
    s2['var']='zsc'
    s['amp'] = coefs1[0]**2
    s['phase'] = coefs1[2]*12/(2*np.pi) % 12
    s['h_shift'] = coefs1[1]
    s2['amp_zsc'] = coefs2[0]**2
    s2['phase_zsc'] = coefs2[2]*12/(2*np.pi) % 12
    s2['h_shift_zsc'] = coefs2[1]
    list_s.append(s)
    list_s2.append(s2)
#
# df_north = df[df.reg <=15]
# df_south = df[df.reg > 15]
#
# s_n_dh = bin_valid_df_by_season(df_north)
# s_n_dh['hemi'] = 'north'
# s_n_dh['var'] = 'dh'
# s_n_zsc = bin_valid_df_by_season(df_north,var='zsc')
# s_n_zsc['hemi'] = 'north'
# s_n_zsc['var'] = 'zsc'
#
# s_s_dh = bin_valid_df_by_season(df_south)
# s_s_dh['hemi'] = 'south'
# s_s_dh['var'] = 'dh'
# s_s_zsc = bin_valid_df_by_season(df_south,var='zsc')
# s_s_zsc['hemi'] = 'south'
# s_s_zsc['var'] = 'zsc'
#
# s_ns = pd.concat([s_n_dh,s_n_zsc,s_s_dh,s_s_zsc])
# fn_seas_ns = os.path.join(dir_valid_out,'valid_ICS_IB_seas_NS.csv')
# s_ns.to_csv(fn_seas_ns)

df_seas = pd.concat(list_s+list_s2)
fn_seas = os.path.join(dir_valid_out,'valid_ICS_IB_seas_corr_final_weight.csv')
df_seas.to_csv(fn_seas)
#
# #3/ AFTER SEASONAL CORRECTIONS

print('Deriving statistics after seasonal corrections')

bin_dt = [0,60,120,180,240,300,360,540,720,900,1080]
dt = bin_valid_df_by_vals(df, bin_dt, np.abs(df.dt))
dt['type'] = 'dt'

bin_t = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]
t = bin_valid_df_by_vals(df,bin_t,pd.to_datetime(df.t))
t['type'] = 't'

bin_h = np.arange(0,1.1,0.1)
h = bin_valid_df_by_vals(df,bin_h,df.h)
h['type'] = 'h'

bin_dh_tot = [-150,-100,-50,-35,-15,-10,-5,0,5,10,15]
dh_tot = bin_valid_df_by_vals(df, bin_dh_tot, df.dh_tot)
dh_tot['type'] = 'dh_tot'

bin_reg = np.arange(1, 21)
r = bin_valid_df_by_vals(df, bin_reg, df.reg)
r['type'] = 'reg'

bin_dh = np.arange(-12,13,2)
dh = bin_valid_df_by_vals(df, bin_dh, df.dh)
dh['type']  ='dh'

bin_zsc = np.arange(-3,3.1,0.5)
zsc = bin_valid_df_by_vals(df, bin_zsc, df.zsc)
zsc['type']  ='zsc'

bin_all = [min(df.zsc),max(df.zsc)]
a, a_ls = bin_valid_df_by_vals(df,bin_all,df.zsc,return_ls=True)
a['type'] = 'all'

df_north = df[df.reg <=15]
bin_months = np.arange(1, 14, 2)
months = pd.DatetimeIndex(df_north.t).month.values
m_n = bin_valid_df_by_vals(df_north,bin_months,months)
m_n['type'] = 'seas_north'

df_south = df[df.reg > 15]
bin_months = np.arange(1, 14, 2)
months = pd.DatetimeIndex(df_south.t).month.values
m_s = bin_valid_df_by_vals(df_south,bin_months,months)
m_s['type'] = 'seas_south'

df_end = pd.concat([dt,t,h,dh_tot,r,dh,zsc,a,m_n,m_s])
df_end['seas_corr'] = 1

df_out = pd.concat([df_init,df_end])
fn_out = os.path.join(dir_valid_out,'valid_ICS_IB_all_bins_final_weight.csv')
df_out.to_csv(fn_out)

fn_a_ls = os.path.join(dir_valid_out,'valid_ICS_IB_all_bins_final_weight_all_ls.csv')
a_ls.to_csv(fn_a_ls)