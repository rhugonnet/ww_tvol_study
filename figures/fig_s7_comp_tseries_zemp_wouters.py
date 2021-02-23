
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from matplotlib.dates import MonthLocator, YearLocator
from glob import glob
from matplotlib.legend_handler import HandlerErrorbar, HandlerPatch, HandlerBase
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

dir_wouters = '/home/atom/data/other/Wouters_2019/GRACE_w_GIA_err.txt'
dir_zemp = '/home/atom/data/other/Zemp_2019/correction/Zemp_etal_results_regions_global_v11'
dir_hugonnet = '/home/atom/ongoing/work_worldwide/vol/reg'

region_wouters = ['Alaska','Western_Canada_and_US','Arctic_Canada_North','Arctic_Canada_South',None,'Iceland','Svalbard','Scandinavia','Arctic_Russia','North_Asia','Central_Europe','Caucasus','HMA','Low_Latitudes','Southern_Andes','New_Zealand',None,None]
region_zemp = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20]
region_names = ['Alaska (01)','Western Canada and US (02)','Arctic Canada North (03)','Arctic Canada South (04)','Greenland (05)','Iceland (06)','Svalbard (07)','Scandinavia (08)','Russian Arctic (09)','North Asia (10)','Central Europe (11)','Caucasus (12)','HMA (13-15)','Low Latitudes (16)','Southern Andes (17)','New Zealand (18)','Antarctic (19)','Global']

def decyears_to_datetime(arr_dec):

    arr_dt = np.zeros(np.shape(arr_dec),dtype='datetime64[D]')
    for i, dec in enumerate(arr_dec):

        year = int(dec)
        rem = dec - year

        base = datetime(year, 1, 1)
        result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)

        arr_dt[i] = result

    return arr_dt

fig,axes = plt.subplots(6,3,sharex=True,figsize=(14,12))

for i in np.arange(len(region_zemp)):

    # fn_bert = '/home/atom/data/validation/Wouters_2019/Iceland_hydrocorr_upd/Iceland_no_hydro.txt'
    if region_wouters[i] is not None:
        fn_bert = glob(os.path.join(dir_wouters,region_wouters[i]+'*'))[0]
        df_w = pd.read_csv(fn_bert, sep=' ', header=None)
        df_w.columns = ['date', 'dm', 'dm_err']

    if region_zemp[i] != 20:
        fn_zemp = glob(os.path.join(dir_zemp,'*_'+str(region_zemp[i])+'_*'))[0]
        df_z = pd.read_csv(fn_zemp, header=26)
    else:
        fn_zemp = '/home/atom/data/other/Zemp_2019/correction/Zemp_etal_results_regions_global_v11/Zemp_etal_results_global.csv'
        df_z = pd.read_csv(fn_zemp, header=18)



    if region_zemp[i] not in [13,14,15,20]:
        fn_hugonnet = os.path.join(dir_hugonnet,'dh_'+str(region_zemp[i]).zfill(2)+'_rgi60_int_base_reg.csv')
        df_tot = pd.read_csv(fn_hugonnet)
        df_tot = df_tot[df_tot.reg==region_zemp[i]]
    elif region_zemp[i] in [13,14,15]:
        fn_hugonnet = os.path.join(dir_hugonnet,'dh_13_14_15_rgi60_int_base_reg.csv')
        df_tot = pd.read_csv(fn_hugonnet)
    else:
        list_df = []
        for j in np.arange(1,20):
            list_df.append(pd.read_csv(os.path.join(dir_hugonnet,'dh_'+str(j).zfill(2)+'_rgi60_int_base_reg.csv')))
        df_tot= pd.concat(list_df)
        dm_tot = np.zeros(len(df_tot[df_tot.reg==1]))
        for j in np.arange(1,20):
            dm_tot += df_tot[df_tot.reg==j].dm.values
        df_tot = df_tot[df_tot.reg==1]
        df_tot.dm = dm_tot

    ncol = i % 3
    nrow = int(np.floor(i/3))
    print('Plot: '+str(i))
    print(ncol)
    print(nrow)
    ax = axes[nrow,ncol]
    # fig, ax = plt.subplots()

    if region_wouters[i] is not None:
        ax.plot(decyears_to_datetime(df_w.date.values),df_w.dm-df_w.dm[0]-(df_w.dm.values[4]),label='Wouters',color='tab:blue')
        ax.fill_between(decyears_to_datetime(df_w.date.values),df_w.dm-df_w.dm[0]-df_w.dm.values[4] + df_w.dm_err, df_w.dm-df_w.dm[0]-df_w.dm.values[4]-df_w.dm_err,alpha=0.15,color='tab:blue',edgecolor=None)

        # ax.fill_between(df_tot.time.values.astype('datetime64[D]'),df_tot['dm']-df_tot['dm'][27]-2*df_tot['err_dm'],df_tot['dm']-df_tot['dm'][25]+2*df_tot['err_dm'],alpha=0.3,color='red')

    # plt.plot(decyears_to_datetime(df_h.date_utc.values/365.2422),df_h.dm-65,label='Hippel')

    ind_0_zemp = 2002-df_z.iloc[0,0]
    if i != 17:
        p4 = ax.errorbar(decyears_to_datetime(df_z.Year[ind_0_zemp:]+9/12.),np.cumsum(df_z.iloc[:,10][ind_0_zemp:])-np.cumsum(df_z.iloc[:,10][ind_0_zemp:]).values[0],2*np.sqrt(np.cumsum(df_z.iloc[:,18][ind_0_zemp:]**2)),label='Zemp',color='tab:red',linestyle='--')
    else:
        ax.errorbar(decyears_to_datetime(df_z.Year[ind_0_zemp:]+9/12.),np.cumsum(df_z.iloc[:,2][ind_0_zemp:])-np.cumsum(df_z.iloc[:,2][ind_0_zemp:]).values[0],2*np.sqrt(np.cumsum(df_z.iloc[:,8][ind_0_zemp:]**2)),label='Zemp',color='tab:red',linestyle='--')

    if os.path.exists(fn_hugonnet):
        p5 = ax.plot(df_tot.time.values.astype('datetime64[D]'), df_tot['dm'].values - df_tot['dm'].values[33],
                     color='black', alpha=0.8, label='This study (monthly)')
        p3 = ax.errorbar(df_tot.time.values.astype('datetime64[D]')[9::12],
                         df_tot['dm'].values[9::12] - df_tot['dm'].values[33],
                         2 * np.sqrt(df_tot['err_dm'].values[12::12] ** 2 + df_tot['err_dm'].values[24] ** 2),
                         label='This study (annual)', color='black', fmt='x',elinewidth=2)

    ax.set_xlim([np.datetime64('2001-12-01'),np.datetime64('2017-01-01')])
    if i+1 in [2,8,10,11,12,14,16]:
        if os.path.exists(fn_hugonnet):
            min_y = min(np.cumsum(df_z.iloc[:,10][ind_0_zemp:])-np.cumsum(df_z.iloc[:,10][ind_0_zemp:]).values[0])-10
            max_y = max(max(np.cumsum(df_z.iloc[:,10][ind_0_zemp:])-np.cumsum(df_z.iloc[:,10][ind_0_zemp:]).values[0])+10,max(df_tot['dm'].values[23:]-df_tot['dm'].values[24]+10))
            ax.set_ylim([min_y,max_y])

    yloc = YearLocator(2)
    # mloc = MonthLocator()
    ax.xaxis.set_major_locator(yloc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax.xaxis.set_minor_locator(mloc)
    ax.grid(True)
    if i==0:
        p1 = ax.plot([], [], color='tab:blue', linewidth=1)
        p2 = ax.fill([], [], color='tab:blue', alpha=0.15,edgecolor=None)


        def make_legend_polygon(legend, orig_handle,
                                xdescent, ydescent,
                                width, height, fontsize):
            a = 1.5 * height
            p = mpatches.Polygon(
                np.array([[0, -a / 2], [width, -a / 2], [width, height + a / 2], [0, height + a / 2], [0, -a / 2]]))
            return p
        hm = {p2[0]: HandlerPatch(patch_func=make_legend_polygon)}
        ax.legend([(p1[0], p2[0]),p4,p3], ['Wouters', 'Zemp','This study'],handlelength=1, framealpha=0.8, loc='lower left', ncol=3, labelspacing=0.1, columnspacing=0.3, handler_map=hm)

        ax.text(0.95, 0.95, region_names[i], horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, fontweight='bold',bbox= dict(facecolor='white', alpha=0.8))
        # ax.legend(loc='lower left',ncol=2,columnspacing=0.1)
    else:
        ax.text(0.05, 0.05, region_names[i], horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, fontweight='bold',bbox= dict(facecolor='white', alpha=0.8))

ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Year')
plt.ylabel('Cumulative mass change (Gt)\n ')

plt.savefig('/home/atom/ongoing/work_worldwide/figures/revised/Figure_S7_2.png',dpi=400)