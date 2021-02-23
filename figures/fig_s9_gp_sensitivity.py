import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import scipy.interpolate
import matplotlib as mpl
import matplotlib.ticker as ticker

list_regions = ['06_rgi60','08_rgi60']

period_1yr = ['20'+str(i).zfill(2)+'-01-01_20'+str(i+1).zfill(2)+'-01-01' for i in range(20)]
period_2yr = ['20'+str(2*i).zfill(2)+'-01-01_20'+str(2*(i+1)).zfill(2)+'-01-01' for i in range(10)]
period_5yr = ['20'+str(5*i).zfill(2)+'-01-01_20'+str(5*(i+1)).zfill(2)+'-01-01' for i in range(4)]
period_10yr = ['20'+str(10*i).zfill(2)+'-01-01_20'+str(10*(i+1)).zfill(2)+'-01-01' for i in range(2)]
period_20yr = ['20'+str(20*i).zfill(2)+'-01-01_20'+str(20*(i+1)).zfill(2)+'-01-01' for i in range(1)]
list_periods = [period_1yr,period_2yr,period_5yr,period_10yr,period_20yr]
length_periods = [1,2,5,10,20]

list_df_reg = []
for region in list_regions:

    sens_dir = os.path.join('/home/atom/ongoing/work_worldwide/sensitivity',region)
    list_fn_csv = glob(os.path.join(sens_dir,'*_subperiods.csv'),recursive=True)

    list_fn_test = [fn_csv for fn_csv in list_fn_csv if 'base' not in os.path.basename(fn_csv).split('_')[3]]
    fn_base = [fn_csv for fn_csv in list_fn_csv if 'base' in os.path.basename(fn_csv).split('_')[3]][0]

    df_base = pd.read_csv(fn_base)

    list_df_out = []
    for periods in list_periods:

        # print('Looking at periods of length '+str(length_periods[list_periods.index(periods)]))

        for fn_test in list_fn_test:

            df = pd.read_csv(fn_test)
            test= os.path.basename(fn_test).split('_')[3]
            fac=os.path.basename(fn_test).split('_')[4]

            # print('Working on test: '+test)

            diff = []
            diff_to_err = []
            diff_to_signal = []

            for period in periods:

                df_base_per = df_base[df_base.period==period]
                df_per = df[df.period==period]

                d= df_base_per.dhdt.values[0]-df_per.dhdt.values[0]
                d_t_e = d/(2*df_base_per.err_dhdt)*100
                d_t_s = d/(df_base_per.dhdt.values[0])*100

                diff.append(d)
                diff_to_err.append(d_t_e)
                diff_to_signal.append(d_t_s)

            df_out = pd.DataFrame()
            df_out['test'] = [test]
            df_out['fac'] = [fac]
            df_out['length_per']=[length_periods[list_periods.index(periods)]]
            df_out['mean_diff']=[np.nanmean(diff)]
            df_out['std_diff']=[np.nanstd(diff)]
            df_out['mean_to_err']=[np.nanmean(diff_to_err)]
            df_out['std_to_err']=[np.nanstd(diff_to_err)]
            df_out['mean_to_sig']=[np.nanmean(diff_to_signal)]
            df_out['std_to_sig']=[np.nanstd(diff_to_signal)]

            list_df_out.append(df_out)

    df_reg = pd.concat(list_df_out)
    df_reg['reg']=int(region.split('_')[0])
    list_df_reg.append(df_reg)

df_all = pd.concat(list_df_reg)

# fig, (ax1,ax2) = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
fig, ((ax0,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharey='row',figsize=(10,10))

# plt.subplots_adjust(hspace=0.3)
# grid = plt.GridSpec(20, 20, wspace=0.5, hspace=0.5)

list_test = ['bvar','blength','pvar','nlvar','nllength','nlalpha']
list_ntest = ['$\sigma_{l}^{2}$','$\Delta t_{l}$','$\sigma_{p}^{2}$','$\sigma_{nl}^{2}$','$\Delta t_{nl}$','$\\alpha_{nl}$']
col_per = ['tab:blue','tab:red','tab:orange','tab:pink','tab:grey','tab:brown']

# list_grid = [((0,10),(0,10)),((10,20),(0,10)),((0,10),(10,20)),((10,20),(10,20))]

reg = [8,8,6,6]
max_y = [4,40,4,40]
ax = [ax0,ax3,ax2,ax4]
lab = ['a','b','c','d']

for j in range(4):

    df_08 = df_all[df_all.reg == reg[j]]

    # ax1 = fig.add_subplot(grid[list_grid[j][0][0]:list_grid[j][0][1],list_grid[j][1][0]:list_grid[j][1][1]])
    ax1 = ax[j]
    shift_x = 0
    for test in list_test:
        sub_shift_x = 0

        for yr in length_periods:

            df_tmp = df_08[np.logical_and.reduce((df_08.test==test,df_08.length_per==yr,df_08.fac.isin(['twice','half'])))]


            val = np.mean(np.abs(df_tmp.mean_to_sig.values))
            val_perc = np.mean(np.abs(df_tmp.mean_to_err.values))

            if max_y[j] == 4:
                ax1.bar(shift_x+sub_shift_x-0.125*6/2+0.125,val,0.125,color=col_per[length_periods.index(yr)],alpha=0.7)
            else:
                ax1.bar(shift_x+sub_shift_x-0.125*6/2+0.125,val_perc,0.125,color=col_per[length_periods.index(yr)],alpha=0.7)

            sub_shift_x += 0.125
        shift_x += 1

    for i in range(len(length_periods)):
        p = ax1.bar([10],[1],[0.1],color=col_per[i],label=str(length_periods[i])+'-year')

    if j == 0:
        ax1.legend()
    # list_name = [test+'\n$\\times$'+'2\n'+'$\div$'+'2' for test in list_ntest]
    if max_y[j] == 4:
        if reg[j] == 8:
            ax1.set_ylabel('Mean absolute deviation to regional estimate (%)')
            ax1.set_title('Scandinavia (region 08)', loc='center', fontweight='bold')
        else:
            ax1.set_title('Iceland (region 06)', loc='center', fontweight='bold')
    else:
        if reg[j] == 8:
            ax1.set_ylabel('Mean absolute deviation relative\nto volume change uncertainties (%)')

    ax1.xaxis.set_major_locator(ticker.FixedLocator(-0.5+np.arange(0,len(list_test)+1)))
    ax1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,len(list_test))))
    ax1.xaxis.set_major_formatter(ticker.NullFormatter())
    ax1.xaxis.set_minor_formatter(ticker.FixedFormatter(list_ntest))
    for tick in ax1.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    ax1.set_xlim((-0.125*6/2-0.125,len(list_test)-0.125*6/2-0.125))
    print(max_y[j])
    ax1.set_ylim((0,max_y[j]))
    ax1.set_axisbelow(True)
    ax1.grid(color='gray', linestyle='dashed')
    ax1.text(0.05, 0.95, lab[j], transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('GP kernel parameter (variation: '+'$\\times$'+'2,'+'$\div$'+'2)')


plt.tight_layout()

# ax2 = fig.add_subplot(grid[6:,:])
plt.savefig('/home/atom/ongoing/work_worldwide/figures/final/Figure_S9.png',dpi=400)