
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybob.ddem_tools import nmad
import scipy
from pybob.bob_tools import mkdir_p
import pyddem.fit_tools as ft

dir_valid = '/data/icesat/travail_en_cours/romain/results/valid'
dir_valid_out = '/data/icesat/travail_en_cours/romain/results/valid_summary_random_sys'

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
df_tot = df[np.abs(df.dh_ref)<200]

# df = df[np.logical_and(df.reg != 5,df.reg!=19)]

print('Finished loading data')

def bin_dh_zsc_by_vals(dh,zsc,bins,bins_val):

    med_dh = []
    nmad_dh = []
    med_zsc = []
    nmad_zsc = []
    mid_bin = []
    for i in range(len(bins)-1):
        ind = np.logical_and(bins_val >= bins[i],bins_val < bins[i+1])
        if len(ind)>100:
            med_dh.append(np.nanmedian(dh[ind]))
            nmad_dh.append(nmad(dh[ind]))
            # nmad_dh.append(np.nanstd(dh[ind]))
            med_zsc.append(np.nanmedian(zsc[ind]))
            nmad_zsc.append(nmad(zsc[ind]))
            # nmad_zsc.append(np.nanstd(zsc[ind]))

            mid_bin.append(bins[i] + 0.5*(bins[i+1]-bins[i]))

    return [np.array(mid_bin), np.array(med_dh), np.array(nmad_dh), np.array(med_zsc), np.array(nmad_zsc)]


def bin_dh_zsc_by_season(dh,zsc,date):

    season_month_bins = np.arange(1,13,1)

    mon = pd.DatetimeIndex(date).month.values

    med_dh = []
    nmad_dh = []
    med_zsc = []
    nmad_zsc = []
    mid_bin = []
    for i in range(len(season_month_bins)):
        ind = (mon == season_month_bins[i])
        if np.count_nonzero(ind)>0:
        # ind = np.logical_and(mon >= season_month_bins[i], mon < season_month_bins[i + 1])
            med_dh.append(np.nanmedian(dh[ind]))
            nmad_dh.append(nmad(dh[ind]))
            med_zsc.append(np.nanmedian(zsc[ind]))
            nmad_zsc.append(nmad(zsc[ind]))

            mid_bin.append(season_month_bins[i])

    return [np.array(mid_bin), np.array(med_dh), np.array(nmad_dh), np.array(med_zsc), np.array(nmad_zsc)]


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
    df_out = df_out.assign(mid_bin=mid_bin,ns_ics=ns_ics,ns_ib=ns_ib)
    df_out['med_'+var]=med
    df_out['nmad_'+var]=std

    return df_out


fig = plt.figure(figsize=(9,12))

plt.subplots_adjust(hspace=0.3)

ax = fig.add_subplot(3, 1, 1)

df = df_tot[df_tot.sensor=='ICS']
nmad_gla = nmad(df[df.pos==2].zsc)
nmad_stable = nmad(df[df.pos==1].zsc)

ax.text(0.025, 0.965, 'a', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax.hist(df[df.pos==1].zsc,np.arange(-5,5,0.1),label='Stable',alpha=0.5,color='tab:red')
ax.hist(df[df.pos==2].zsc,np.arange(-5,5,0.1),label='Glacier',alpha=0.5,color='tab:blue')
ax.vlines(np.nanmedian(df[df.pos==2].zsc),0,600000,color='tab:blue',lw=2)
ax.vlines(np.nanmedian(df[df.pos==1].zsc),0,600000,color='tab:red',lw=2)
ax.vlines(np.nanmedian(df[df.pos==2].zsc),0,0,color='black',label='Median')
ax.vlines(np.nanmedian(df[df.pos==2].zsc),0,0,color='black',label='Median ± NMAD',linestyles='dotted')
ax.vlines(np.nanmedian(df[df.pos==2].zsc) + nmad_gla,0,600000,linestyles='dotted',color='tab:blue',lw=2)
ax.vlines(np.nanmedian(df[df.pos==2].zsc) - nmad_gla,0,600000,linestyles='dotted',color='tab:blue',lw=2)
ax.vlines(np.nanmedian(df[df.pos==1].zsc) + nmad_stable,0,600000,linestyles='dotted',color='tab:red',lw=2)
ax.vlines(np.nanmedian(df[df.pos==1].zsc) - nmad_stable,0,600000,linestyles='dotted',color='tab:red',lw=2)
ax.text(0.25,0.5,'NMAD glacier:\n' +str(np.round(nmad_gla,2)),transform=ax.transAxes,color='tab:blue',fontsize=12,ha='center',fontweight='bold')
ax.text(0.75,0.5,'NMAD stable:\n' +str(np.round(nmad_stable,2)),transform=ax.transAxes,color='tab:red',fontsize=12,ha='center',fontweight='bold')
# ax.text(-4,100000,'$z = \\frac{h_{ICESat} - h_{GPR}}{\\sigma_{h_{GPR}}}$',fontsize=20)
ax.set_xlabel('Z-scores of ICESat')
ax.set_ylabel('Count of ICESat validation points')
# plt.ylim([0,500000])
ax.grid()
ax.legend(loc='upper right',ncol=2)

plt.subplots_adjust(hspace=0.3)

ax = fig.add_subplot(3, 1, 2)

df = df_tot[df_tot.sensor=='IB']
nmad_gla = nmad(df[df.pos==2].zsc)
nmad_stable = nmad(df[df.pos==1].zsc)

ax.text(0.025, 0.965, 'b', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax.hist(df[df.pos==1].zsc,np.arange(-5,5,0.1),label='Stable',alpha=0.5,color='tab:red')
ax.hist(df[df.pos==2].zsc,np.arange(-5,5,0.1),label='Glacier',alpha=0.5,color='tab:blue')
ax.vlines(np.nanmedian(df[df.pos==2].zsc),0,2500000,color='tab:blue',lw=2)
ax.vlines(np.nanmedian(df[df.pos==1].zsc),0,2500000,color='tab:red',lw=2)
ax.vlines(np.nanmedian(df[df.pos==2].zsc),0,0,color='black',label='Median')
ax.vlines(np.nanmedian(df[df.pos==2].zsc),0,0,color='black',label='Median ± NMAD',linestyles='dotted')
ax.vlines(np.nanmedian(df[df.pos==2].zsc) + nmad_gla,0,2500000,linestyles='dotted',color='tab:blue',lw=2)
ax.vlines(np.nanmedian(df[df.pos==2].zsc) - nmad_gla,0,2500000,linestyles='dotted',color='tab:blue',lw=2)
ax.vlines(np.nanmedian(df[df.pos==1].zsc) + nmad_stable,0,2500000,linestyles='dotted',color='tab:red',lw=2)
ax.vlines(np.nanmedian(df[df.pos==1].zsc) - nmad_stable,0,2500000,linestyles='dotted',color='tab:red',lw=2)
ax.text(0.25,0.5,'NMAD glacier:\n' +str(np.round(nmad_gla,2)),transform=ax.transAxes,color='tab:blue',fontsize=12,ha='center',fontweight='bold')
ax.text(0.75,0.5,'NMAD stable:\n' +str(np.round(nmad_stable,2)),transform=ax.transAxes,color='tab:red',fontsize=12,ha='center',fontweight='bold')
# ax.text(-4,100000,'$z = \\frac{h_{ICESat} - h_{GPR}}{\\sigma_{h_{GPR}}}$',fontsize=20)
ax.set_xlabel('Z-scores of IceBridge')
ax.set_ylabel('Count of IceBridge validation points')
# plt.ylim([0,500000])
ax.grid()
ax.legend(loc='upper right',ncol=2)

ax = fig.add_subplot(3, 1, 3)

df_corr = df_tot
dt2 = bin_valid_df_by_vals(df_corr[df_corr.pos==2], [0,50,100,150,200,250,300,350,450,550,650,800,1000,1200,1400,1600], np.abs(df_corr[df_corr.pos==2].dt),list_var=['zsc'])
dt = bin_valid_df_by_vals(df_corr[df_corr.pos==1], [0,50,100,150,200,250,300,350,450,550,650,800,1000,1200,1400,1600], np.abs(df_corr[df_corr.pos==1].dt),list_var=['zsc'])
ax.text(0.025, 0.965, 'c', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax.scatter(dt.mid_bin.values, dt.nmad_zsc.values, color='tab:red',label='Stable')
ax.scatter(dt2.mid_bin.values, dt2.nmad_zsc.values, color='tab:blue',label='Glacier')
# ax.axhline(np.nanmedian(dt[4]),0,2000,linestyle=':',color='tab:red')
# ax.axhline(np.nanmedian(dt2[4]),0,2000,linestyle=':',color='tab:blue')
# ax.axhline([],[],[],linestyle=':',color='black',label='median')
ax.set_xlabel('Days to closest observation')
ax.set_ylabel('NMAD of z-scores')
ax.axhline(1,0,2000,linestyle='dashed',color='black',label='Unit variance')
ax.set_ylim([-0.2,2.5])
ax.legend(loc='upper center')
ax.grid()

plt.savefig(os.path.join(dir_valid_out,'Figure_S6.png'),dpi=400)

print('Done with random figure.')
# FIGURE 9

fig = plt.figure(figsize=(12,10))

ax3 = fig.add_subplot(2,2,1)

df = df_tot[df_tot.sensor=='ICS']

list_camp = sorted(list(set(list(df.t))))

vec_gla, vec_sta, vec_tdx = ([] for i in range(3))

for i in np.arange(len(list_camp)):
    df_tmp = df[df.t==list_camp[i]]
    vec_gla.append(np.nanmedian(df_tmp[df_tmp.pos==2].dh.values))
    vec_sta.append(np.nanmedian(df_tmp[df_tmp.pos==1].dh.values))
    vec_tdx.append(np.nanmedian(df_tmp[df_tmp.pos==1].dh_ref.values))

list_camp = np.array(list_camp,dtype='datetime64[D]')

laser_op_name = ['1AB\n(2003-02-20)', '2A', '2B', '2C', '3A', '3B', '3C', '3D', '3E', '3F', '3G', '3H', '3I', '3J', '3K', '2D',
                 '2E', '2F\n(2009-10-11)']

ax3.scatter(laser_op_name,vec_gla,color='tab:blue',marker='x')
ax3.scatter(laser_op_name,vec_sta,color='tab:red',marker='x')
ax3.scatter(laser_op_name,vec_tdx,color='black',marker='x')

a=ax3.scatter([],[],color='tab:red',marker='x',label='$h_{GP} - h_{ICS}$ (stable)')
b=ax3.scatter([],[],color='tab:blue',marker='x',label='$h_{GP} - h_{ICS}$ (glacier)')
c=ax3.scatter([],[],color='black',marker='x',label='$h_{TDX} - h_{ICS}$ (stable)')
d,=ax3.plot([],[],color='black',linestyle=':',label='Median')

hh = [a,b,c,d]

ax3.hlines(np.median(vec_gla),'1AB\n(2003-02-20)','2F\n(2009-10-11)',color='tab:blue',linestyles=':')
ax3.hlines(np.median(vec_sta),'1AB\n(2003-02-20)','2F\n(2009-10-11)',color='tab:red',linestyles=':')
ax3.hlines(np.median(vec_tdx),'1AB\n(2003-02-20)','2F\n(2009-10-11)',color='black',linestyles=':')

ax3.set_xlabel('ICESat campaign')
ax3.set_ylabel('Elevation difference (m)')
ax3.set_ylim([-1,3])
ax3.legend(hh,[H.get_label() for H in hh],loc='upper right')
ax3.grid()
ax3.text(0.05, 0.95, 'a', transform=ax3.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

ax1 = fig.add_subplot(2,2,2)

df = df_tot[df_tot.sensor=='IB']

print(len(df))

bin_t = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in np.arange(9,21)]

vec_gla, vec_sta, vec_tdx = ([] for i in range(3))

for i in np.arange(len(bin_t)-1):
    df_tmp = df[np.logical_and(pd.to_datetime(df.t)>=bin_t[i],pd.to_datetime(df.t)<bin_t[i+1])]
    print(len(df_tmp))
    vec_gla.append(np.nanmedian(df_tmp[df_tmp.pos==2].dh.values))
    vec_sta.append(np.nanmedian(df_tmp[df_tmp.pos==1].dh.values))
    vec_tdx.append(np.nanmedian(df_tmp[df_tmp.pos==1].dh_ref.values))

ax1.scatter(np.arange(2009,2020),vec_gla,color='tab:blue',marker='x')
ax1.scatter(np.arange(2009,2020),vec_sta,color='tab:red',marker='x')
ax1.scatter(np.arange(2009,2020),vec_tdx,color='black',marker='x')

print(vec_gla)
print(vec_sta)
print(vec_tdx)

a=ax1.scatter([],[],color='tab:red',marker='x',label='$h_{GP} - h_{IB}$ (stable)')
b=ax1.scatter([],[],color='tab:blue',marker='x',label='$h_{GP} - h_{IB}$ (glacier)')
c=ax1.scatter([],[],color='black',marker='x',label='$h_{TDX} - h_{IB}$ (stable)')
d,=ax1.plot([],[],color='black',linestyle=':',label='Median')

hh = [a,b,c,d]

ax1.hlines(np.median(vec_gla),2009,2019,color='tab:blue',linestyles=':')
ax1.hlines(np.median(vec_sta),2009,2019,color='tab:red',linestyles=':')
ax1.hlines(np.median(vec_tdx),2009,2019,color='black',linestyles=':')

ax1.set_xlabel('Year of ICEBridge campaign')
ax1.set_ylabel('Elevation difference (m)')
ax1.set_ylim([-1,3])
ax1.set_xlim([2008.5,2019.5])
ax1.legend(hh,[H.get_label() for H in hh],loc='upper right')
ax1.grid()
ax1.text(0.05, 0.95, 'b', transform=ax1.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')


ax0 = fig.add_subplot(2,2,3)

df = df_tot[df_tot.sensor=='ICS']

ind_sn = np.logical_and.reduce((df.pos==1,df.reg>15,df.reg!=19))
ind_gn = np.logical_and.reduce((df.pos==2,df.reg>15,df.reg!=19))

ind_ss = np.logical_and(df.pos==1,df.reg<=15)
ind_gs = np.logical_and(df.pos==2,df.reg<=15)

t0_n = bin_valid_df_by_season(df[ind_sn],var='dh')
t0_s = bin_valid_df_by_season(df[ind_ss],var='dh')
t0_n_ref = bin_valid_df_by_season(df[ind_sn],var='dh_ref')
t0_s_ref = bin_valid_df_by_season(df[ind_ss],var='dh_ref')

ax0.plot([],[],color='tab:red',label='$h_{GP} - h_{ICS}$ (stable)')
ax0.plot([],[],color='tab:blue',label='$h_{GP} - h_{ICS}$ (glacier)')
ax0.plot([],[],color='black',label='$h_{TDX} - h_{ICS}$ (stable)')
ax0.plot([],[],marker='o',linestyle=':',color='black',label='Northern\nhemisphere')
ax0.plot([],[],marker='^',linestyle='--',color='black',label='Southern\nhemisphere')

ax0.scatter(t0_s.mid_bin.values, t0_s.med_dh.values, marker='o',color='tab:red')
ax0.scatter(t0_n.mid_bin.values, t0_n.med_dh.values, marker='^',color='tab:red')
# ax0.errorbar(t0_s[0][:-1], t0_s[1][:-1], t0_s[2][:-1],fmt='o',color='red')
# ax0.errorbar(t0_n[0][:-1], t0_n[1][:-1], t0_n[2][:-1], fmt='^',color='red')
ax0.scatter(t0_n_ref.mid_bin.values,t0_n_ref.med_dh_ref.values,marker='^',color='black')
ax0.scatter(t0_s_ref.mid_bin.values,t0_s_ref.med_dh_ref.values,marker='o',color='black')

x = np.arange(0,12.1,0.1)
coefs , _ = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.sin(t*2*np.pi/12+c)+b, t0_n_ref.mid_bin.values[~np.isnan(t0_n_ref.med_dh_ref.values)], t0_n_ref.med_dh_ref.values[~np.isnan(t0_n_ref.med_dh_ref.values)])
y_500 = coefs[0]*np.sin(x*2*np.pi/12+coefs[2])+coefs[1]
ax0.plot(x,y_500,color='black',linestyle='--')
x = np.arange(0,12.1,0.1)
coefs , _ = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.sin(t*2*np.pi/12+c)+b, t0_s_ref.mid_bin.values[~np.isnan(t0_s_ref.med_dh_ref.values)], t0_s_ref.med_dh_ref.values[~np.isnan(t0_s_ref.med_dh_ref.values)])
y_500 = coefs[0]*np.sin(x*2*np.pi/12+coefs[2])+coefs[1]
ax0.plot(x,y_500,color='black',linestyle=':')

x = np.arange(0,12.1,0.1)
coefs , _ = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.sin(t*2*np.pi/12+c)+b, t0_n.mid_bin.values[~np.isnan(t0_n.med_dh.values)], t0_n.med_dh.values[~np.isnan(t0_n.med_dh.values)])
y_500 = coefs[0]*np.sin(x*2*np.pi/12+coefs[2])+coefs[1]
ax0.plot(x,y_500,color='tab:red',linestyle='--')

x = np.arange(0,12.1,0.1)
coefs , _ = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.sin(t*2*np.pi/12+c)+b, t0_s.mid_bin.values[~np.isnan(t0_s.med_dh.values)], t0_s.med_dh.values[~np.isnan(t0_s.med_dh.values)])
y_500 = coefs[0]*np.sin(x*2*np.pi/12+coefs[2])+coefs[1]
ax0.plot(x,y_500,color='tab:red',linestyle=':')

# ax0.plot(t0[0], t0[1],linestyle=':')

t1_n = bin_valid_df_by_season(df[ind_gn],var='dh')
t1_s = bin_valid_df_by_season(df[ind_gs],var='dh')

ax0.scatter(t1_s.mid_bin.values, t1_s.med_dh.values, marker='o',color='tab:blue')
ax0.scatter(t1_n.mid_bin.values, t1_n.med_dh.values, marker='^',color='tab:blue')

# ax0.errorbar(t1_s[0][:-1], t1_s[1][:-1], t1_s[2][:-1],fmt='o',color='blue')
# ax0.errorbar(t1_n[0][:-1], t1_n[1][:-1], t1_s[2][:-1], fmt='^',color='blue')

coefs , _ = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.sin(t*2*np.pi/12+c)+b, t1_n.mid_bin.values[~np.isnan(t1_n.med_dh.values)], t1_n.med_dh.values[~np.isnan(t1_n.med_dh.values)])
y_500 = coefs[0]*np.sin(x*2*np.pi/12+coefs[2])+coefs[1]
ax0.plot(x,y_500,color='tab:blue',linestyle='--')

x = np.arange(0,12.1,0.1)
coefs , _ = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.sin(t*2*np.pi/12+c)+b, t1_s.mid_bin.values[~np.isnan(t1_s.med_dh.values)], t1_s.med_dh.values[~np.isnan(t1_s.med_dh.values)])
y_500 = coefs[0]*np.sin(x*2*np.pi/12+coefs[2])+coefs[1]
ax0.plot(x,y_500,color='tab:blue',linestyle=':')

# ax0.plot(t1[0], t1[1],linestyle=':')
ax0.set_xlabel('Month of the year (decimal)')
ax0.set_ylabel('Elevation difference (m)')
ax0.set_ylim([-1,3])
ax0.legend(loc='upper right',ncol=2)
ax0.text(0.05, 0.95, 'c', transform=ax0.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax0.grid()

ax2 = fig.add_subplot(2,2,4)

df = df_tot

curv_s = bin_valid_df_by_vals(df[df.pos==1],np.arange(-0.02,0.05,0.001),df[df.pos==1].curv,list_var=['dh'])
curv_g = bin_valid_df_by_vals(df[df.pos==2],np.arange(-0.02,0.05,0.001),df[df.pos==2].curv,list_var=['dh'])
curv_r = bin_valid_df_by_vals(df[df.pos==1],np.arange(-0.02,0.05,0.001),df[df.pos==1].curv,list_var=['dh_ref'])

ax2.plot(curv_s.mid_bin.values,curv_s.med_dh.values,color='tab:red',label='$h_{GP} - h_{IC}$ (stable)')
ax2.plot(curv_g.mid_bin.values,curv_g.med_dh.values,color='tab:blue',label='$h_{GP} - h_{IC}$ (glacier)')
ax2.plot(curv_r.mid_bin.values,curv_r.med_dh_ref.values,color='black',label='$h_{TDX} - h_{IC}$ (stable)')
ax2.hlines(0,-0.03,0.1,color='black',linestyle='dotted',lw=2)
ax2.set_xlabel('Curvature (10$^{-3}$ m$^{-2}$)')
ax2.set_ylabel('Elevation difference (m)')
ax2.set_xlim((-0.025,0.055))
ax2.set_ylim((-20,35))
ax2.legend()
ax2.text(0.05, 0.95, 'd', transform=ax2.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax2.grid()

plt.savefig(os.path.join(dir_valid_out,'Figure_S4.png'),dpi=400)
print('Done with systematic figure.')
