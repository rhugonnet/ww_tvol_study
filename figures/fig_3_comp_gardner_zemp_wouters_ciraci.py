from __future__ import print_function
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.legend_handler import HandlerErrorbar, HandlerPatch, HandlerBase
import matplotlib.patches as mpatches
from glob import glob

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

tlim_zemp = [np.datetime64('2006-01-01'),np.datetime64('2016-01-01')]
tlim_wouters = [np.datetime64('2002-01-01'),np.datetime64('2016-01-01')]
tlim_gardner = [np.datetime64('2003-01-01'),np.datetime64('2009-01-01')]
tlim_shean = [np.datetime64('2000-01-01'),np.datetime64('2018-01-01')]
tlim_braun = [np.datetime64('2000-01-01'),np.datetime64('2013-01-01')]

list_df = []
for fn_reg in list_fn_reg:
    df_reg = pd.read_csv(fn_reg)
    df_zemp = integrate_periods(df_reg,tlim_zemp,fn_tarea=fn_tarea,frac_area=1)
    df_zemp['comp'] = 'zemp'
    df_gard = integrate_periods(df_reg,tlim_gardner,fn_tarea=fn_tarea,frac_area=1)
    df_gard['comp'] = 'gard'
    df_wout = integrate_periods(df_reg,tlim_wouters,fn_tarea=fn_tarea,frac_area=1)
    df_wout['comp'] = 'wout'
    df_shean = integrate_periods(df_reg,tlim_shean,fn_tarea=fn_tarea,frac_area=1)
    df_shean['comp'] = 'shean'
    df_braun = integrate_periods(df_reg,tlim_braun,fn_tarea=fn_tarea,frac_area=1)
    df_braun['comp'] = 'braun'
    df_tmp = pd.concat([df_gard,df_wout,df_zemp,df_shean,df_braun])
    list_df.append(df_tmp)

df = pd.concat(list_df)

fig = plt.figure(figsize=(15,5.5))
grid = plt.GridSpec(20, 17, wspace=0.25, hspace=0)

x_axis = np.arange(0,17*4)
#
# ax0 = fig.add_subplot(grid[:4, :10])
# ax0.set_xticks([])
# # ax1=ax0.twinx()
#
r_list = [1,2,3,4,5,6,7,8,9,10,11,12,21,16,17,18,19]
#
# shift_x=0
# for i in range(len(r_list)):
#
#     if r_list[i] == 21 or r_list[i] == 17:
#         shift_x += 0.5
#     area = df[df.reg==r_list[i]].area.values[0]/1000000
#     ax0.fill_between([shift_x+x_axis[4*i]-0.25,shift_x+x_axis[4*i]+2.25],[0]*2,[area]*2,color=plt.cm.Greys(0.80),alpha=1)
#     # ax1.fill_between([x_axis[4*i]+1,x_axis[4*i]+2],[0]*2,[vol_reg[i]]*2,color='dimgrey',alpha=0.8)
#     if r_list[i] == 21 or r_list[i] == 17:
#         shift_x += 0.5
#
# ax0.vlines(np.concatenate((np.arange(-1,12*4,4),47+np.array([5,9,14,18,22]))),0,1000000,colors='grey',alpha=0.7,linewidth=0.75,linestyles='dashed')
#
# ax0.set_xlim([-1,17*4+1])
# ax0.set_ylim([0,max(df.area.values)/1000000+10000])
# # ax0.yaxis.tick_right()
# # ax0.yaxis.set_label_position('right')
# # ax0.set_yscale('log')
# # ax1.set_yscale('log')
# # ax0.spines['top'].set_visible(False)
# # ax0.spines['right'].set_visible(False)
# ax0.set_ylabel('Glacier area (km$^{2}$)')
# # ax1.set_ylabel('Volume (km$^{3}$)')


ax = fig.add_subplot(grid[:-2, :10])

shift_x=0
for i in range(len(r_list)):

    df_g = df[np.logical_and(df.comp=='gard',df.reg==r_list[i])]

    ax.fill_between([shift_x+x_axis[4*i]-0.5,shift_x+x_axis[4*i]+0.5],[df_ext.gar[i]+df_ext.gar_err[i]]*2,[df_ext.gar[i]-df_ext.gar_err[i]]*2,color='tab:red',alpha=0.45)
    ax.plot([shift_x+x_axis[4*i]-0.5,shift_x+x_axis[4*i]+0.5],[df_ext.gar[i]]*2,color='tab:red',lw=2)
    ax.errorbar(shift_x+x_axis[4*i],df_g.dmdtda.values[0],2*df_g.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)

    df_z = df[np.logical_and(df.comp=='zemp',df.reg==r_list[i])]

    ax.fill_between([shift_x+x_axis[4*i+2]-0.5,shift_x+x_axis[4*i+2]+0.5],[df_ext.zemp[i]+df_ext.zemp_err[i]]*2,[df_ext.zemp[i]-df_ext.zemp_err[i]]*2,color='tab:orange',alpha=0.45)
    ax.plot([shift_x+x_axis[4*i+2]-0.5,shift_x+x_axis[4*i+2]+0.5],[df_ext.zemp[i]]*2,color='tab:orange',lw=2)
    ax.errorbar(shift_x+x_axis[4*i+2],df_z.dmdtda.values[0],2*df_z.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)

    df_w = df[np.logical_and(df.comp=='wout',df.reg==r_list[i])]

    ax.fill_between([shift_x+x_axis[4*i+1]-0.5,shift_x+x_axis[4*i+1]+0.5],[df_ext.wout[i]+df_ext.wout_err[i]]*2,[df_ext.wout[i]-df_ext.wout_err[i]]*2,color='tab:blue',alpha=0.45)
    ax.plot([shift_x+x_axis[4*i+1]-0.5,shift_x+x_axis[4*i+1]+0.5],[df_ext.wout[i]]*2,color='tab:blue',lw=2)
    ax.errorbar(shift_x+x_axis[4*i+1], df_w.dmdtda.values[0],2*df_w.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)

    if r_list[i] == 21 or r_list[i] == 17 or r_list[i] == 16:

        if r_list[i]==21:
            df_t = df[np.logical_and(df.comp=='shean',df.reg==r_list[i])]
            col = 'tab:purple'
            dmdtda = -0.19
            err_dmdtda= 0.03
        elif r_list[i]==17:
            df_t = df[np.logical_and(df.comp=='braun',df.reg==r_list[i])]
            col = 'tab:green'
            dmdtda = -0.64
            err_dmdtda=0.02
        elif r_list[i]==16:
            df_t = df[np.logical_and(df.comp=='braun',df.reg==r_list[i])]
            col = 'tab:green'
            dmdtda = -0.23
            err_dmdtda=0.04

        ax.fill_between([shift_x + x_axis[4 * i + 3] - 0.5,shift_x+ x_axis[4 * i + 3] + 0.5],
                        [dmdtda + 2*err_dmdtda] * 2, [dmdtda -2*err_dmdtda] * 2,
                        color=col, alpha=0.45)
        ax.plot([shift_x + x_axis[4 * i + 3] - 0.5, shift_x+x_axis[4 * i + 3] + 0.5], [dmdtda] * 2, color=col,lw=2)
        ax.errorbar(shift_x + x_axis[4 * i + 3], df_t.dmdtda.values[0], 2 * df_t.err_dmdtda.values[0], fmt='o',
                         color=plt.cm.Greys(0.8), capsize=3, zorder=30)
        shift_x += 1



ax.vlines(np.concatenate((np.arange(-1,12*4,4),47+np.array([5,10,15,19,23]))),-3,3,colors='grey',alpha=0.7,linewidth=0.75,linestyles='dashed')


ax.hlines(0,-5,17*4+5,linestyles='dashed',colors=plt.cm.Greys(0.9))

ticks = ['ALA (01)','WNA (02)','ACN (03)','ACS (04)','GRL (05)', 'ISL (06)','SJM (07)', 'SCA (08)','RUA (09)','ASN (10)','CEU (11)','CAU (12)','HMA (13-15)','TRP (16)','SAN (17)','NZL (18)','ANT (19)']
ax.set_xticks(np.concatenate((np.arange(-1,12*4,4),47+np.array([5,9.5,14,18,22])))+2)
ax.set_xticklabels(ticks,rotation='vertical')
ax.set_ylabel('Specific mass change rate (m w.e yr$^{-1}$)')
ax.set_xlim([-1,17*4+2])
ax.set_ylim([-1.5,0.4])
ax.text(0.025, 0.95, 'A', transform=ax.transAxes, ha='left', va='top',fontweight='bold',fontsize=14)

p3 = ax.plot([], [], color='tab:red', linewidth=2)
p4 = ax.fill([], [], color='tab:red', alpha=0.45)
p5 = ax.plot([], [], color='tab:blue', linewidth=2)
p6 = ax.fill([], [], color='tab:blue', alpha=0.45)
p1 = ax.plot([], [], color='tab:orange', linewidth=2)
p2 = ax.fill([], [], color='tab:orange', alpha=0.45)
p7 = ax.plot([], [], color='tab:purple', linewidth=2)
p8 = ax.fill([], [], color='tab:purple', alpha=0.45)
p9 = ax.plot([], [], color='tab:green', linewidth=2)
p10 = ax.fill([], [], color='tab:green', alpha=0.45)
p0 = ax.errorbar([], [], [], fmt='o',
            color='black', capsize=3)

# ax.legend()

def make_legend_polygon(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    a=1.5*height
    p = mpatches.Polygon(np.array([[0,-a/2],[width,-a/2],[width,height+a/2],[0,height+a/2],[0,-a/2]]))
    return p


hm = {p0: HandlerErrorbar(xerr_size=0.9), p4[0]: HandlerPatch(patch_func=make_legend_polygon), p2[0]: HandlerPatch(patch_func=make_legend_polygon), p6[0]: HandlerPatch(patch_func=make_legend_polygon),p8[0]: HandlerPatch(patch_func=make_legend_polygon),p10[0]: HandlerPatch(patch_func=make_legend_polygon)}
ax.legend([(p3[0],p4[0]),(p5[0],p6[0]),(p1[0], p2[0]),(p7[0], p8[0]),(p9[0], p10[0]),p0], ['Gardner (2003-2009)','Wouters (2002-2016)','Zemp (2006-2016)','Shean (2000-2018)','Braun (2000-2013)','This study (same period)'],ncol=3,handlelength=1,framealpha=1,loc='upper right',labelspacing=1.2,handler_map=hm,borderpad=0.7)
# ax.yaxis.grid(True,linestyle='--')


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
    df_sum = df_sum.assign(area=[area_global],area_nodata=[area_nodata_global],dhdt=[dhdt_global],err_dhdt=[err_dhdt_global],dvoldt=[dvoldt_global],err_dvoldt=[err_dvoldt_global]
                                 ,dmdt=[dmdt_global],err_dmdt=[err_dmdt_global],dmdtda=[dmdtda_global],err_dmdtda=[err_dmdtda_global],tarea=[tarea_global],perc_area_res=[perc_area_res_global],perc_area_meas=[perc_area_meas_global],valid_obs=[valid_obs_global]
                                 ,valid_obs_py=[valid_obs_py_global])

    return df_sum

reg_dir = '/home/atom/ongoing/work_worldwide/vol/reg'
list_fn_reg_multann = [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg_subperiods.csv') for i in np.arange(1,20)]
out_csv = '/home/atom/ongoing/work_worldwide/tables/Table1_final.csv'
df_all = pd.DataFrame()
for fn_reg_multann in list_fn_reg_multann:
    df_all= df_all.append(pd.read_csv(fn_reg_multann))

tlims = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = sum_regions(df_p)
    df_global['period']=period
    df_noperiph = sum_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)


df_g = df[df.comp=='gard']
df_g_glo = sum_regions(df_g)
df_g_per = sum_regions(df_g[~df_g.reg.isin([5, 19])])

df_z = df[df.comp=='zemp']
df_z_glo = sum_regions(df_z)
df_z_per = sum_regions(df_z[~df_z.reg.isin([5, 19])])

dmdtda_gard = -0.35
err_gard = 0.04

dmdtda_zemp = -0.48
err_zemp = 0.2

ax1 = fig.add_subplot(grid[:-2, 14:])

tlims = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = sum_regions(df_p)
    df_global['period']=period
    df_noperiph = sum_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_glob[df_glob.period==str(tlims[i])+'_'+str(tlims[i+1])]
    # ax1.errorbar(tlims[i]+(tlims[i+1]-tlims[i])/2,df_p.dmdtda.values[0],2*df_p.err_dmdtda.values[0],fmt='x',color=plt.cm.Blues(0.9),capsize=3,zorder=3,lw=0.5)
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.3)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=1)

    # df_p = df_per[df_per.period==str(tlims[i])+'_'+str(tlims[i+1])]
    # ax1.errorbar(tlims[i]+(tlims[i+1]-tlims[i])/2,df_p.dmdtda.values[0],2*df_p.err_dmdtda.values[0],fmt='<',color=plt.cm.Purples(0.9),capsize=5,zorder=4)
    # ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Purples(0.9),alpha=0.5)
    # ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Purples(0.9),lw=2,linestyle='dotted')

tlims = [np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = sum_regions(df_p)
    df_global['period']=period
    df_noperiph = sum_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_glob[df_glob.period==str(tlims[i])+'_'+str(tlims[i+1])]
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.5)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=2)


ax1.fill_between(tlim_zemp,[dmdtda_zemp+err_zemp]*2,[dmdtda_zemp-err_zemp]*2,color='tab:orange',alpha=0.45)
ax1.plot(tlim_zemp, [dmdtda_zemp] * 2, color='tab:orange')
# ax1.errorbar(tlim_zemp[0]+(tlim_zemp[1]-tlim_zemp[0])/2,df_z_glo.dmdtda.values[0],2*df_z_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)
ax1.fill_between(tlim_gardner,[dmdtda_gard+err_gard]*2,[dmdtda_gard-err_gard]*2,color='tab:red',alpha=0.45)
ax1.plot(tlim_gardner, [dmdtda_gard] * 2, color='tab:red')
# ax1.errorbar(tlim_gardner[0]+(tlim_gardner[1]-tlim_gardner[0])/2,df_g_glo.dmdtda.values[0],2*df_g_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)


ax1.plot([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')],[0,0],linestyle='dashed',color=plt.cm.Greys(0.9))

ax1.set_xlim((np.datetime64('2000-01-01'),np.datetime64('2020-01-01')))
ax1.set_xticks([np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)])
ax1.set_ylim([-0.75,0.01])
ax1.set_yticks([])
ax.set_ylabel('Specific mass change rate (m w.e yr$^{-1}$)')
ax1.set_xlabel('Year\n(Global total)')


ax1 = fig.add_subplot(grid[:-2, 11:14])

dmdtda_gard = -0.42
err_gard = 0.05

dmdtda_zemp = -0.56
err_zemp = 0.04

dmdtda_wouters = -0.41
err_wouters = 0.07

tlims = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = sum_regions(df_p)
    df_global['period']=period
    df_noperiph = sum_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_per[df_per.period==str(tlims[i])+'_'+str(tlims[i+1])]
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.3)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=1)

tlims = [np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = sum_regions(df_p)
    df_global['period']=period
    df_noperiph = sum_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_per[df_per.period==str(tlims[i])+'_'+str(tlims[i+1])]
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.5)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=2)


ax1.fill_between(tlim_zemp,[dmdtda_zemp+err_zemp]*2,[dmdtda_zemp-err_zemp]*2,color='tab:orange',alpha=0.45)
ax1.plot(tlim_zemp, [dmdtda_zemp] * 2, color='tab:orange')

ax1.fill_between(tlim_wouters,[dmdtda_wouters+err_wouters]*2,[dmdtda_wouters-err_wouters]*2,color='tab:blue',alpha=0.45)
ax1.plot(tlim_wouters, [dmdtda_wouters] * 2, color='tab:blue')
# ax1.errorbar(tlim_zemp[0]+(tlim_zemp[1]-tlim_zemp[0])/2,df_z_glo.dmdtda.values[0],2*df_z_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)
ax1.fill_between(tlim_gardner,[dmdtda_gard+err_gard]*2,[dmdtda_gard-err_gard]*2,color='tab:red',alpha=0.45)
ax1.plot(tlim_gardner, [dmdtda_gard] * 2, color='tab:red')
# ax1.errorbar(tlim_gardner[0]+(tlim_gardner[1]-tlim_gardner[0])/2,df_g_glo.dmdtda.values[0],2*df_g_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)


ax1.plot([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')],[0,0],linestyle='dashed',color=plt.cm.Greys(0.9))

ax1.set_xlim((np.datetime64('2000-01-01'),np.datetime64('2020-01-01')))
ax1.set_ylim([-0.75,0.01])
ax1.set_xticks([np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(4)])
# ax.set_ylabel('Specific mass change (m w.e yr$^{-1}$)')
ax1.set_xlabel('Year\n(Global excl. GRL and ANT)')
ax1.set_ylabel('Specific mass change rate (m w.e yr$^{-1}$)')
ax1.text(0.075, 0.95, 'B', transform=ax1.transAxes, ha='left', va='top',fontweight='bold',fontsize=14)

p1 = ax.plot([], [], color=plt.cm.Greys(0.9), linewidth=1)
p2 = ax.fill([], [], color=plt.cm.Greys(0.9), alpha=0.3)

p3 = ax.plot([], [], color=plt.cm.Greys(0.9), linewidth=2)
p4 = ax.fill([], [], color=plt.cm.Greys(0.9), alpha=0.5)

hm = { p4[0]: HandlerPatch(patch_func=make_legend_polygon), p2[0]: HandlerPatch(patch_func=make_legend_polygon)}
ax1.legend([(p1[0],p2[0]),(p3[0],p4[0])], ['This study (yearly)','This study (quinquennial)'],handlelength=1,framealpha=1,loc=(0.4,0.82),ncol=1,labelspacing=1.2,handler_map=hm)

plt.savefig('/home/atom/ongoing/work_worldwide/figures/revised/Figure_3.png',dpi=400)